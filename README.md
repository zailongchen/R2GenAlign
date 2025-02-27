# R2GenAlign: Bidirectional Visual-Textual Alignment for LLM-based Radiology Report Generation

## Introduction
![overview](https://github.com/zailongchen/R2GenAlign/blob/main/images/frame.png)

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

Mimic-cxr: you can download our preprocess annotation file from [here](https://drive.google.com/file/d/14689ztodTtrQJYs--ihB_hgsPMMNHX-H/view?usp=sharing) and download the images from [official website](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

After downloading the data, place it in the ./data folder.

### Training (for MIMIC-CXR)

```bash
bash scripts/mimic/r2genalign_train.sh
```

### Testing (For MIMIC-CXR)

```bash
bash scripts/mimic/r2genalign_test.sh
```

## Acknowledgement

+ [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4) Some codes of this repo are based on MiniGPT-4.
+ [Llama2](https://github.com/facebookresearch/llama) The fantastic language ability of Llama-2 with only 7B parameters is just amazing.


## License
This repository is under [BSD 3-Clause License](LICENSE.md).
