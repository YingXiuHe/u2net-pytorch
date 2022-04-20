# coding: utf-8
# author: hxy
# 2022-04-20
"""
数据读取dataset
"""
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


# dataset for u2net
class U2netSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_size=(320, 320)):
        """
        :param img_dir: 数据集图片文件夹路径
        :param mask_dir: 数据集mask文件夹路径
        :param input_size: 图片输入的尺寸
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.samples = list()
        self.gt_mask = list()
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
        self.load_data()

    def __len__(self):
        return len(self.samples)

    def load_data(self):
        img_dir_full_path = self.img_dir
        mask_dir_full_path = self.mask_dir
        img_files = os.listdir(img_dir_full_path)

        for img_name in tqdm(img_files):
            img_full_path = os.path.join(img_dir_full_path, img_name)
            mask_full_path = os.path.join(mask_dir_full_path, img_name)

            img = cv2.imread(img_full_path)
            img = cv2.resize(img, self.input_size)
            img2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img2norm = (img2rgb - self.mean) / self.std
            # 图像格式改为nchw
            img2nchw = np.transpose(img2norm, [2, 0, 1]).astype(np.float32)

            gt_mask = cv2.imread(mask_full_path)
            gt_mask = cv2.resize(gt_mask, self.input_size)
            gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
            gt_mask = gt_mask / 255.
            gt_mask = np.expand_dims(gt_mask, axis=0)

            self.samples.append(img2nchw)
            self.gt_mask.append(gt_mask)

        return self.samples, self.gt_mask

    def __getitem__(self, index):
        img = self.samples[index]
        mask = self.gt_mask[index]

        return img, mask
