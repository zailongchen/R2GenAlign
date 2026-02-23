# Analyzing and Enhancing Visual Learning in LLM-based Radiology Report Generation

## Introduction
![overview](https://github.com/zailongchen/R2GenAlign/blob/main/images/framework.jpg)

## Getting Started
### Installation

**1. Prepare the code and the environment**

Git clone our repository and install the requirements.

```bash
https://github.com/zailongchen/R2GenAlign.git
cd R2GenAlign
pip install -r requirements.txt
```


**2. Prepare the training dataset**

IU-xray: you can download our preprocess annotation file from [here](https://drive.google.com/file/d/1OXETn7goaYFyFuCaXyfQ6pFC77XXX9EV/view?usp=sharing) and download the dataset from [here](https://drive.google.com/file/d/1c0BXEuDy8Cmm2jfN0YYGkQxFZd2ZIoLg/view)

Mimic-cxr: you can download our preprocess annotation file from [here](https://drive.google.com/file/d/1D1BbsKd9R5npeDDkKDAV06mMYnLHwX2B/view?usp=sharing) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training 

For MIMIC-CXR

```bash
bash scripts/mimic/r2genalign_train.sh
```

For IU-Xray

```bash
bash scripts/iuxray/r2genalign_train.sh
```

### Testing

For MIMIC-CXR

```bash
bash scripts/mimic/r2genalign_test.sh
```

For IU-Xray

```bash
bash scripts/iuxray/r2genalign_test.sh
```

## Acknowledgement

+ [R2GenGPT](https://github.com/wang-zhanyu/R2GenGPT) This repo is mainly built upon R2GenGPT. We sincerely appreciate the authors' contributions to the original implementation.
+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
