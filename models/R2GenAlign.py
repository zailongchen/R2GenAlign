import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, BlipProcessor, BlipForConditionalGeneration, SwinModel, ViTMAEForPreTraining, AutoTokenizer, AutoModel
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
import torch.distributed as dist
from peft import get_peft_model, LoraConfig, TaskType


class LinearNorm(nn.Module):
    def __init__(self, in_dim=None,out_dim=None):
        super(LinearNorm,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim))
        
    def forward(self,x):
        return self.net(x)


class R2GenAlign(pl.LightningModule):
    """
    R2GenAlign model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        if args.vis_use_lora:
            peft_config_visual = LoraConfig(
                                    r=args.vis_r,
                                    lora_alpha=args.vis_alpha,
                                    target_modules=["query", "value"],
                                    lora_dropout=args.lora_dropout,
                                    bias="none",
                                    modules_to_save=["classifier"],
                                )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
            print('Loading vision encoder with LoRA -- Done')
        elif args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Loading Frozen vision encoder:{args.vision_model} -- Done')
        else:
            print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        self.processor = AutoTokenizer.from_pretrained(args.clinicalbert)
        self.text_encoder = AutoModel.from_pretrained(args.clinicalbert)
        print('Loading encoders Done')  

        print('Loading LLAMA')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, use_fast=False)
        self.llama_tokenizer.pad_token_id = 0
        if args.low_resource:
            self.llama_model = LlamaForCausalLM.from_pretrained(
                args.llama_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto")
        else:
            self.llama_model = LlamaForCausalLM.from_pretrained(args.llama_model,torch_dtype=torch.bfloat16,)
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                target_modules= [ "gate_proj", "down_proj", "up_proj"],#["q_proj", "v_proj"],  # , "k_proj", "gate_proj", "down_proj", "up_proj", "o_proj"
                inference_mode=False, 
                r=args.llm_r, 
                lora_alpha=args.llm_alpha, 
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')
        
        self.visual_llama_proj = LinearNorm(in_dim=self.visual_encoder.config.hidden_size,out_dim=self.llama_model.config.hidden_size)
        self.text_llama_proj = LinearNorm(in_dim=self.text_encoder.config.hidden_size,out_dim=self.llama_model.config.hidden_size)

        self.text_prompt = "<<SYS>>\nGenerate visual embedding sequence of the chest xray image for this radiology report. \n<</SYS>>\n\nThe report is"
        self.img_prompt = "<<SYS>>\nGenerate a comprehensive and detailed diagnosis report for this chest xray image. \n<</SYS>>\n\nThe image is"
        self.end_sym = args.end_sym
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location='cpu')['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')

    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            # (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores
    
    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            image_embed = self.visual_encoder(image)['last_hidden_state'].to(device) # for swin
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        inputs_llama = self.visual_llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama
    
    def encode_text(self, text, device):
        text_tokens = self.processor(
            text=text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(device)

        text_embeds = self.text_encoder(
            input_ids = text_tokens.input_ids,
            attention_mask = text_tokens.attention_mask,
            token_type_ids = text_tokens.token_type_ids  # no this parameter for clip text encoder
        ).last_hidden_state

        text_embeds = self.text_llama_proj(text_embeds)
        return text_embeds, text_tokens.attention_mask

    def text_prompt_wrap(self, text_embeds, atts_text):
        device = text_embeds.device
        prompt=f'[INST] {self.text_prompt} <Text><TextHere></Text>. [/INST] '
        batch_size = text_embeds.shape[0]
        p_before, p_after = prompt.split('<TextHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(text_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(text_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        atts_p_before = torch.ones(p_before_embeds.shape[:2]).to(device)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        atts_p_after = torch.ones(p_after_embeds.shape[:2]).to(device)
        wrapped_text_embeds = torch.cat([p_before_embeds, text_embeds, p_after_embeds], dim=1)
        wrapped_atts_text = torch.cat([atts_p_before, atts_text, atts_p_after], dim=1)
        return wrapped_text_embeds, wrapped_atts_text
    
    def img_prompt_wrap(self, img_embeds, atts_img):
        prompt=f'[INST] {self.img_prompt} <Img><ImageHere></Img>. [/INST] '
        batch_size = img_embeds.shape[0]
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples, batch_idx):
        image = samples["image"]
        batch_size = image[0].shape[0]
        img_embeds, atts_img = self.encode_img(image)

        wraped_img_embeds, wraped_atts_img = self.img_prompt_wrap(img_embeds[:,1:,:], atts_img[:,:-1]) # to generate report

        # CXR to report part
        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["report"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(img_embeds.device)

        targets = to_regress_tokens.input_ids.masked_fill(to_regress_tokens.input_ids == 0, -100)

        empty_targets = (torch.ones([wraped_atts_img.shape[0], wraped_atts_img.shape[1]+1],dtype=torch.long).to(img_embeds.device).fill_(-100))  # plus one for bos
        targets = torch.cat([empty_targets, targets], dim=1)

        bos = torch.ones([batch_size, 1],
                         dtype=wraped_atts_img.dtype,
                         device=wraped_atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = wraped_atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, wraped_img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, wraped_atts_img, to_regress_tokens.attention_mask], dim=1)

        # outputs = self.llama_model.base_model.model(
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        lm_loss = outputs.loss

        # report to visual embedding part
        text_embeds, atts_text = self.encode_text(samples['report'],img_embeds.device)

        # compute the mse loss between the generated pseudo-visual embeddings and the extracted visual embeddings
        wraped_text_embeds, wraped_atts_text = self.text_prompt_wrap(text_embeds, atts_text) # to generate CXR
        
        inputs_embeds = torch.cat([bos_embeds, wraped_text_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, wraped_atts_text, atts_img], dim=1)

        # hidden_states = self.llama_model.base_model.model.model(
        hidden_states = self.llama_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )[0]

        generated_img_embeds = hidden_states[:, -img_embeds.shape[1]-1:-1, :] # Shift so that embeds < n predict n
        align_loss = self.mse_loss(generated_img_embeds, img_embeds)

        all_loss = lm_loss + align_loss 
        return {"loss": all_loss, "lm_loss": lm_loss, "align_loss": align_loss}

    def training_step(self, batch, batch_idx):
        result = self(batch, batch_idx)
        self.log_dict(result, prog_bar=True)
        
        optimizer = self.optimizers()
        self.logger.log_hyperparams({"learning_rate": optimizer.state_dict()['param_groups'][0]['lr']})
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}_ori.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['report'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        wraped_img_embeds, wraped_atts_img = self.img_prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, wraped_img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, wraped_atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=0,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})

        return hypo, ref

    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text[15:] # remove the view information
        return output_text

    def on_validation_epoch_end(self):
        self.val_step_outputs = self.all_gather(self.val_step_outputs)
        ref, hypo, ids = [], [], []
        for cnt in range(len(self.val_step_outputs)):
            for i in self.val_step_outputs[cnt]:
                ref.extend(i['ref'])
                hypo.extend(i['hypo'])
                ids.extend(i['id'])
        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)
        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()

    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['report'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds, atts_img = self.img_prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=0,
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})

        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        self.test_step_outputs = self.all_gather(self.test_step_outputs)
        ref, hypo, ids = [], [], []
        for cnt in range(len(self.test_step_outputs)):
            for i in self.test_step_outputs[cnt]:
                ref.extend(i['ref'])
                hypo.extend(i['hypo'])
                ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

    @torch.no_grad()
    def all_gather(self, data):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        dist.barrier()
        gather_data = [None for _ in range(torch.distributed.get_world_size())]
        dist.all_gather_object(gather_data, data)
        return gather_data
