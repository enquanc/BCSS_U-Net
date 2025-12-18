### Docker
docker run --gpus all -itd --shm-size 128gb --name NCU_DL -v /DATA2/user/docker/NCU_DL/:/code -v /DATA2/user/docker/NCU_DL/dataset:/dataset nvcr.io/nvidia/pytorch:23.10-py3
docker exec -it enquanc_DL /bin/bash

### pip package
pip install h5py
pip install numpy==1.26.0
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install numba==0.62.1
pip uninstall opencv opencv-python opencv-python-headless -y
pip install "opencv-python-headless==4.7.0.72"
pip install albumentations==1.3.1



### CMD
python train.py  --n_epochs 20 --batch_size 32 --learning_rate 5e-3 --lr_decay_factor 0.9 --model Attention-Unet --save_dir ./checkpoints/1214_ --n_filters 32 --device cuda:4 
python train.py  --n_epochs 20 --batch_size 32 --learning_rate 5e-3 --lr_decay_factor 0.9 --model Unet --save_dir ./checkpoints/1214_ --n_filters 64 --device cuda:4 