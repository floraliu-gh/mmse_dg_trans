import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

class DeepGlobeDataset(Dataset):
    def __init__(self, root_dir, split='train', img_size=(64, 64)):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        
        metadata_path = os.path.join(root_dir, 'metadata.csv')
        self.df = pd.read_csv(metadata_path)
        self.df = self.df.dropna()
        if split is not None:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)
            
        self.color_map = {
            (0, 255, 255): 0,   # urban_land
            (255, 255, 0): 1,   # agriculture_land
            (255, 0, 255): 2,   # rangeland
            (0, 255, 0): 3,     # forest_land
            (0, 0, 255): 4,     # water
            (255, 255, 255): 5, # barren_land
            (0, 0, 0): 6        # unknown
        }
        
        self.img_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.344, 0.380, 0.407], std=[0.203, 0.136, 0.114])
        ])

        # For masks, NEAREST interpolation is critical
        self.mask_transform = transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST)

    def mask_to_class(self, mask):
        # 轉換為浮點數以計算距離
        mask_np = np.array(mask).astype(np.float32)
        h, w, c = mask_np.shape
        
        # 初始化預設為 6 (unknown)
        class_mask = np.full((h, w), 6, dtype=np.int64)
        min_dist = np.full((h, w), float('inf'))
        
        # 計算每個像素與目標顏色的絕對誤差總和 (L1 距離)
        for rgb, class_idx in self.color_map.items():
            target_color = np.array(rgb, dtype=np.float32)
            dist = np.sum(np.abs(mask_np - target_color), axis=-1)
            
            # 如果這個顏色比之前的更接近，就更新類別
            update_mask = dist < min_dist
            
            # 【關鍵修正】：右邊的 dist 也必須加上 [update_mask]
            min_dist[update_mask] = dist[update_mask] 
            class_mask[update_mask] = class_idx
            
        # 若色差大於閾值 (例如 50)，代表真的不屬於任何已知地貌，設為 unknown (6)
        class_mask[min_dist > 50] = 6 
        
        return torch.from_numpy(class_mask).long()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['sat_image_path']
        mask_name = self.df.iloc[idx]['mask_path']
        
        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, str(mask_name))
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        
        image = self.img_transform(image)
        mask_resized = self.mask_transform(mask)
       
        label_mask = self.mask_to_class(mask_resized)

        return image, label_mask
