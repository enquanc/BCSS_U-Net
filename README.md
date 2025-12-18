# BCSS Semantic Segmentation (U-Net & Attention U-Net)

This repository contains the source code for the **Deep Learning Final Project**. It implements **U-Net** and **Attention U-Net** for semantic segmentation on the Breast Cancer Semantic Segmentation (BCSS) dataset.

## 📂 Dataset
The dataset used in this project is available on Kaggle:
- **Link:** [BCSS - Breast Cancer Semantic Segmentation](https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss)

## 🛠️ Environment Setup

### 1. Docker Environment
To set up the container with the required NVIDIA PyTorch image:

```bash
# Run the container
docker run --gpus all -itd \
    --shm-size 128gb \
    --name BCSS_DL \
    -v /DATA2/user/docker/BCSS_DL/:/code \
    -v /DATA2/user/docker/BCSS_DL/dataset:/dataset \
    nvcr.io/nvidia/pytorch:23.10-py3

# Enter the container
docker exec -it BCSS_DL /bin/bash
(Note: Please adjust the volume paths /DATA2/user/... according to your local machine configuration.)

### 2. Dependencies
Install the required Python packages inside the container:

# Basic utilities
pip install h5py numpy==1.26.0 numba==0.62.1 albumentations==1.3.1

# PyTorch (CUDA 11.8)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# OpenCV (Headless version for server environments)
pip uninstall opencv opencv-python opencv-python-headless -y
pip install "opencv-python-headless==4.7.0.72"


## 🚀 Usage
Replace cuda:x with your specific GPU ID (e.g., cuda:0).
Please Check dataset path.

### Training
Option A: 3 Classes
# Train U-Net
python train.py --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 \
    --model Unet --save_dir ./checkpoints/wd-2/1217_64n --n_filters 64 --device cuda:x

# Train Attention U-Net
python train.py --n_epochs 50 --batch_size 32 --learning_rate 1e-3 --lr_decay_factor 0.7 \
    --model Attention-Unet --save_dir ./checkpoints/wd-2/1217_64n --n_filters 64 --device cuda:x


Option B: 22 Classes
# Train U-Net
python train.py --n_epochs 50 --batch_size 16 --learning_rate 1e-3 --lr_decay_factor 0.7 \
    --model Unet --save_dir ./checkpoints/512/wd-2/1217_64n \
    --n_filters 64 --output_channels 22 --device cuda:x

# Train Attention U-Net
python train.py --n_epochs 50 --batch_size 4 --learning_rate 1e-3 --lr_decay_factor 0.7 \
    --model Attention-Unet --save_dir ./checkpoints/512/wd-2/1217_64n \
    --n_filters 64 --output_channels 22 --device cuda:x


### Testing
Make sure the checkpoint paths (--ckpt) exist before running.
Option A: 3 Classes
python test.py --model Unet --ckpt "checkpoints/Unet-weights.pth" --device cuda:1 --n_filter 32
python test.py --model Attention-Unet --ckpt "checkpoints/Attention-Unet-weights.pth" --device cuda:1 --n_filter 32

Option B: 22 Classes
python test.py --model Unet --ckpt "checkpoints/512/Unet-weights.pth" \
    --device cuda:1 --out_channels 22 --n_filter 32

python test.py --model Attention-Unet --ckpt "checkpoints/512/Attention-Unet-weights.pth" \
    --device cuda:1 --out_channels 22 --n_filter 32