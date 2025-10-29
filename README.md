#### Purpose
Ted talk is hugedataset ,in TPS model, only frames are needed not audio ,so language portability is possible . Im working Hindi video dataset HAV-Df ,which is 7GB sizw with 180 videos. It is far less for any video generation ,now here TPS is trained to generated keypoints which further used as building block for training audio to video or text to video with other principle deep learning. Now using diffted im training HAV-Df dataset to get video from audio using keypoints in condition with audio . Pretrained checkpoints are taken from TPS model (which is firtsly prtrained with TED talk dataset but in this I had finetuned it with my specific wanted dataset to get perfect keypoints ). so it will be be a better approach if you want to train  your model in other language using like this technique

# [CVPR2022] Thin-Plate Spline Motion Model for Image Animation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![stars](https://img.shields.io/github/stars/yoyo-nb/Thin-Plate-Spline-Motion-Model.svg?style=flat)
![GitHub repo size](https://img.shields.io/github/repo-size/yoyo-nb/Thin-Plate-Spline-Motion-Model.svg)

Source code of the CVPR'2022 paper "Thin-Plate Spline Motion Model for Image Animation"

[**Paper**](https://arxiv.org/abs/2203.14367) **|** [**Supp**](https://cloud.tsinghua.edu.cn/f/f7b8573bb5b04583949f/?dl=1)

**PS**: The paper trains the model for 100 epochs for a fair comparison. You can use more data and train for more epochs to get better performance.


### Pre-trained models
- [Google Drive](https://drive.google.com/drive/folders/1pNDo1ODQIb5HVObRtCmubqJikmR7VVLT?usp=sharing)


### Installation

We support ```python3```.(Recommended version is Python 3.9).
To install the dependencies run:
```bash
pip install -r requirements.txt
```
### YAML configs

See description of the parameters in the ```config/ted-384.yaml```. I changed root directory to my HAV-DF dataset path.

### Datasets
 "naman3007/hav-df-dataset"

 ### frames_datset 
 I changed framedataset as per my dataset structure and correctly match with prtratined checkpoints of tedt alk dataset. so just in case compare with original frames_dataset and modify as per you needs
### finetune_tpsmm.py
I created a file importing classes from original files and gradually unfreezed each network and trained upto 100 epochs
### RUN
run finetune_tpsmm.py file ,if you see any mismatch with already checkpoints with currrent using dataset change the config to match.

#### Fine-tuning the TPS Motion Model

We fine-tuned the Thin-Plate Spline Motion Model (TPSMM) using a custom frame-based dataset.
Here’s what was done step by step:

#### Integrated Dataset Loader

We replaced the old dataset.py with a new frames_dataset.py that directly loads frame folders from
/content/drive/MyDrive/DiffTED_project/DiffTED/dataset/frames.

Each folder represents a video sequence — containing source and driving frames.

#### Rebuilt the Training Script (finetune_tpsmm.py)

Clean, modular structure with full PyTorch training loop.

Config loaded from YAML (ted-384.yaml).

Pretrained model loaded from ted.pth.tar.

Automatic checkpoint saving, progress bar, and AMP mixed precision added.

#### Model Components Initialized

✅ Keypoint Detector — extracts motion landmarks from frames.

✅ Dense Motion Network — estimates pixel-level motion using TPS deformation fields.

✅ Inpainting Network — reconstructs realistic images after motion warping.

### Freezing & Fine-tuning Strategy

The Keypoint Detector and Dense Motion Network were kept mostly frozen (used in eval() mode).

The Inpainting Network was actively trained using L1 loss between predicted and driving frames.

This approach fine-tunes only the appearance generation, not the motion representation — faster, more stable convergence.

### Training Run

Loaded ~196 frame sequences.

Forward pass computed keypoints → motion field → warped source → predicted target.

Optimized only the generator (inpainting) using AdamW and mixed precision.

Checkpoints were saved regularly in /content/drive/MyDrive/DiffTED_project/TPS/checkpoints.

### Result

The model started fine-tuning successfully using your dataset.

The only fix needed was handling tensor output (pred.get("prediction")), since the model returns a dictionary with multiple maps — we adjusted that in the next version
# Acknowledgments
The main code is based upon [[FOMM](https://github.com/AliaksandrSiarohin/first-order-model) and [MRAA](https://github.com/snap-research/articulated-animation)](https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model.git)

Thanks for the excellent works!

And Thanks to:

- [@chenxwh](https://github.com/chenxwh): Add Web Demo & Docker environment [![Replicate](https://replicate.com/yoyo-nb/thin-plate-spline-motion-model/badge)](https://replicate.com/yoyo-nb/thin-plate-spline-motion-model) 

- [@TalkUHulk](https://github.com/TalkUHulk): The C++/Python demo is provided in [Image-Animation-Turbo-Boost](https://github.com/TalkUHulk/Image-Animation-Turbo-Boost)

- [@AK391](https://github.com/AK391): Add huggingface web demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/CVPR/Image-Animation-using-Thin-Plate-Spline-Motion-Model)
