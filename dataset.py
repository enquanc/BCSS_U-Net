from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2


class BCSS_Dataset(Dataset):
    def __init__(self, images_list, images_dir, labels_dir=None, transform=None):
        self.images_dir = images_dir
        self.images = images_list
        self.labels_dir = labels_dir
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.labels_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        mask = mask.long()

        return image, mask
    

def create_dataset(train_image_path = 'archive/BCSS/train/', val_image_path = 'archive/BCSS/val/', test_image_path = 'archive/BCSS/test/',
                      train_mask_path = 'archive/BCSS/train_mask/', val_mask_path = 'archive/BCSS/val_mask/'):

    train_dir = train_image_path

    # 1. Read ALL file name
    file_names = [f for f in os.listdir(train_dir) if f.endswith('.png')]
    df = pd.DataFrame({'filename': file_names})

    # 2. Analyze Patient ID (關鍵步驟)
    # BCSS 的檔名通常源自 TCGA，formate like: "TCGA-AR-A1AS-01Z-00-DX1.png"
    # 我們需要取前幾個區段作為 Group ID (通常前三個區段代表一位病患)
    # 例如: TCGA-AR-A1AS
    df['patient_id'] = df['filename'].apply(lambda x: "-".join(x.split('-')[:3]))

    # 3. 進行 Group Split (依病患分組，而非依圖片)
    # test_size=0.1 表示切 10% 出來當作新的 Validation
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=33)

    train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # 4. 驗證拆分結果
    print("-" * 30)
    print(f"New Train 數量: {len(train_df)} (包含 {train_df['patient_id'].nunique()} 位病患)")
    print(f"New Valid 數量: {len(val_df)} (包含 {val_df['patient_id'].nunique()} 位病患)")

    # 確保沒有重疊
    overlap = set(train_df['patient_id']) & set(val_df['patient_id'])
    print(f"病患重疊數: {len(overlap)} (必須為 0)")


    train_imgs_list = train_df['filename'].to_list()
    val_imgs_list = val_df['filename'].to_list()
    test_imgs_list = os.listdir(val_image_path)

    # Define transformations using Albumentations
    transforms_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-10,10), shear=(-5,5), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    transforms_val = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_dataset = BCSS_Dataset(train_imgs_list, train_image_path, train_mask_path, transform=transforms_train)
    val_dataset = BCSS_Dataset(val_imgs_list, train_image_path, train_mask_path, transform=transforms_val)
    test_dataset = BCSS_Dataset(test_imgs_list, val_image_path, val_mask_path, transform=transforms_val)

    return train_dataset, val_dataset, test_dataset