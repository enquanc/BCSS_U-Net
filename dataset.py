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

    # 2. Analyze Patient ID 
    # BCSS file name from TCGA，formate like: "TCGA-AR-A1AS-01Z-00-DX1.png"
    # We need get the first few section as Group ID (Usually first three section as one patient)
    # Ex: TCGA-AR-A1AS
    df['patient_id'] = df['filename'].apply(lambda x: "-".join(x.split('-')[:3]))

    # 3. Group Split (Splitting based on patient , not only image)
    # 10% for Validation from Original Training dataset
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=33)

    train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    # 4. Check the validation result
    print("-" * 30)
    print(f"New Train counts: {len(train_df)} (Include {train_df['patient_id'].nunique()} patients)")
    print(f"New Valid counts: {len(val_df)} (Include {val_df['patient_id'].nunique()} patients)")

    # Check no over lap
    overlap = set(train_df['patient_id']) & set(val_df['patient_id'])
    print(f"The counts of duplicate patients: {len(overlap)} (Must be 0 !!!)")


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