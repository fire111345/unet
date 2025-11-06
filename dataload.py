import torch
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import random
import os

class Heart3DDataset(Dataset):
    def __init__(self, root_dir, case_list, data_type="Tr", patch_size=(64,128,128), augment=False):
        self.root_dir = root_dir
        self.case_list = case_list
        self.patch_size = np.array(patch_size)
        self.augment = augment
        self.data_type = data_type  # 新增

        self.images = [os.path.join(root_dir, f"images{data_type}", f"{c}_0000.nii.gz") for c in case_list]
        self.labels = [os.path.join(root_dir, f"labels{data_type}", f"{c}.nii.gz") for c in case_list]


    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, idx):
        # 读取原始数据
        img = nib.load(self.images[idx]).get_fdata().astype(np.float32)
        lbl = nib.load(self.labels[idx]).get_fdata().astype(np.uint8)

        # 每次随机裁剪 patch
        img, lbl = self.random_crop(img, lbl, self.patch_size)
        img, lbl = self.pad_or_crop(img, lbl, self.patch_size)

        # 每次都随机增强
        if self.augment:
            img, lbl = self.random_flip(img, lbl)
            img, lbl = self.random_rotate(img, lbl)

        # 转 (C, D, H, W)
        img = np.expand_dims(img, 0)
        #lbl = np.expand_dims(lbl, 0)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def random_crop(self, img, lbl, patch_size):
        d, h, w = img.shape
        pd, ph, pw = patch_size
        start_d = random.randint(0, max(d-pd,0)) if d>pd else 0
        start_h = random.randint(0, max(h-ph,0)) if h>ph else 0
        start_w = random.randint(0, max(w-pw,0)) if w>pw else 0
        return img[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw], \
               lbl[start_d:start_d+pd, start_h:start_h+ph, start_w:start_w+pw]

    def random_flip(self, img, lbl):
        if random.random()>0.5:
            img = np.flip(img, axis=0)
            lbl = np.flip(lbl, axis=0)
        if random.random()>0.5:
            img = np.flip(img, axis=1)
            lbl = np.flip(lbl, axis=1)
        if random.random()>0.5:
            img = np.flip(img, axis=2)
            lbl = np.flip(lbl, axis=2)
        return img.copy(), lbl.copy()

    def random_rotate(self, img, lbl):
        k = random.randint(0,3)
        img = np.rot90(img, k, axes=(1,2))
        lbl = np.rot90(lbl, k, axes=(1,2))
        return img.copy(), lbl.copy()

    def pad_or_crop(self, img, lbl, patch_size):
        #保证输出 (D,H,W) 一致
        d, h, w = img.shape
        pd, ph, pw = patch_size
        pad_d, pad_h, pad_w = max(0, pd - d), max(0, ph - h), max(0, pw - w)

        # 先 pad，再裁剪
        if any([pad_d, pad_h, pad_w]):
            img = np.pad(img, ((0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')
            lbl = np.pad(lbl, ((0,pad_d),(0,pad_h),(0,pad_w)), mode='constant')

        img = img[:pd, :ph, :pw]
        lbl = lbl[:pd, :ph, :pw]
        return img, lbl