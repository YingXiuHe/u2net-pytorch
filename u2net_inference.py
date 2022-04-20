# coding: utf-8
# author: hxy
# 20220420
"""
u2net/u2netP模型推理程序
"""

import os
import cv2
import torch
import numpy as np
from time import time
from tqdm import tqdm
from src.u2net import U2NET, U2NETP

"""
初始化模型加载
"""
try:
    print('===loading model===')
    current_project_path = os.getcwd()
    net = U2NET(3, 1)
    # net = U2NETP(3, 1)
    checkpoint_path = os.path.join(current_project_path,
                                   'backup/checkpoint_220_0.13608702938807637.pth')
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(checkpoint_path, map_location='cuda:1'))
    else:
        net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    net.eval()
    print('===model lode sucessed===')

except Exception as e:
    print('===model load error:{}==='.format(e))


# 计算dice
def dice_coef(output, target):  # output为预测结果 target为真实结果
    smooth = 1e-5  # 防止0除
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
           (output.sum() + target.sum() + smooth)


# 图像归一化操作
def img2norm(img_array, input_size):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]
    _std = np.array(std).reshape((1, 1, 3))
    _mean = np.array(mean).reshape((1, 1, 3))

    img_array = cv2.resize(img_array, input_size)
    norm_img = (img_array - _mean) / _std

    return norm_img


# 归一化预测结果
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)

    return dn


# 推理
def inference1folder(img_folder, mask_folder, input_size):
    total_times = list()
    total_dices = list()
    img_files = os.listdir(img_folder)
    for img_file in tqdm(img_files):
        img_full_path = os.path.join(img_folder, img_file)
        mask_full_path = os.path.join(mask_folder, img_file)
        img = cv2.imread(img_full_path)
        gt_mask = cv2.imread(mask_full_path)
        gt_mask = cv2.resize(gt_mask, input_size)
        gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_BGR2GRAY)
        gt_mask = gt_mask / 255.

        ori_h, ori_w = img.shape[:2]
        img2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        norm_img = img2norm(img2rgb, input_size)

        x_tensor = torch.from_numpy(norm_img).permute(2, 0, 1).float()
        x_tensor = torch.unsqueeze(x_tensor, 0)

        start_t = time()
        d1, d2, d3, d4, d5, d6, d7 = net(x_tensor)
        end_t = time()

        total_times.append(end_t - start_t)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)
        pred = pred.squeeze().cpu().data.numpy()

        dice_value = dice_coef(pred, gt_mask)
        total_dices.append(dice_value)

        # pred[pred>=0.3]=255
        # pred[pred<0.3]=0
        # pred_res = pred
        pred_res = pred * 255
        pred_res = cv2.resize(pred_res, (ori_w, ori_h))

        cv2.imwrite(os.path.join(current_project_path, 'infer_output/', img_file), pred_res)

    print('==inference 1 pic avg cost:{:.4f}ms=='.format(np.mean(total_times) * 1000))
    print('==inference avg dice:{:.4f}=='.format(np.mean(total_dices)))

    return None


if __name__ == '__main__':
    test_img_folder = os.path.join(os.getcwd(), 'dataset/images/test')
    test_gt_mask_folder = os.path.join(os.getcwd(), 'dataset/masks/test')
    inference1folder(img_folder=test_img_folder, mask_folder=test_gt_mask_folder, input_size=(160, 160))
