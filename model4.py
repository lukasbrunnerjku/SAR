import torch
from torch.utils.data import DataLoader
from datasets import SARdata, Transformation
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.patches as patches
import torchvision.ops as ops
# tensorboard --logdir=runs
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import math
from os.path import join
import os
import argparse
import sys 
import logging
# -> clone yolov5 repo into yolov5 folder 
# to run '$ python *.py' files in subdirectories
sys.path.append('../yolov5')  
logger = logging.getLogger(__name__)



def target_prepare(self, item, h, w):
    """
    Transform data for training.
    :param item: dictionary
        - image: 
        - bboxes: n x 4; [[x, y, width, height], ...]
        - cids: n,
    :return: dictionary
        - image: 
        - cpt_hm: 1 x 128 x 128 # num_classes x 128 x 128
        - cpt_off: n_max x 2 low resolution offset - [0, 1)
        - cpt_ind: n_max, low resolution indices - [0, 128^2)
        - cpt_mask: n_max,
        - wh: n_max x 2, low resolution width, height - [0, 128], [0, 128]
        - cls_id: n_max,
    """
    bboxes = item['bboxes']  # n x 4
    # xmin, ymin, xmax, ymax
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]  # width
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]  # height
    # are now: x, y, width, height
    cids = item["cids"]  # n,
    
    # bboxes can be dropped
    len_valid = len(bboxes)

    # to be batched we have to bring everything to the same shape
    cpt = np.zeros((self.n_max, 2), dtype=np.float32)
    # get center points of bboxes (image coordinates)
    cpt[:len_valid, 0] = bboxes[:, 0] + bboxes[:, 2] / 2  # x
    cpt[:len_valid, 1] = bboxes[:, 1] + bboxes[:, 3] / 2  # y

    cpt_mask = np.zeros((self.n_max,), dtype=np.uint8)
    cpt_mask[:len_valid] = 1

    # LOW RESOLUTION bboxes
    wh = np.zeros((self.n_max, 2), dtype=np.float32)
    wh[:len_valid, :] = bboxes[:, 2:-1] / self.down_ratio

    cls_id = np.zeros((self.n_max,), dtype=np.uint8)
    # the bbox labels help to reassign the correct classes
    cls_id[:len_valid] = cids[bboxes[:, -1].astype(np.int32)]

    # LOW RESOLUTION dimensions
    hl, wl = int(self.h / self.down_ratio), int(self.w / self.down_ratio)
    cpt = cpt / self.down_ratio

    # discrete center point coordinates
    cpt_int = cpt.astype(np.int32)

    cpt_ind = np.zeros((self.n_max,), dtype=np.int64)
    # index = y * wl + x
    cpt_ind[:len_valid] = cpt_int[:len_valid, 1] * wl + cpt_int[:len_valid, 0]

    cpt_off = np.zeros((self.n_max, 2), dtype=np.float32)
    cpt_off[:len_valid] = (cpt - cpt_int)[:len_valid]

    cpt_hms = []
    valid_cpt = cpt[cpt_mask.astype(np.bool)]  # n_valid x 2
    for i in range(self.num_classes):
        mask = (cls_id[:len_valid] == i)  # n_valid,
        xy = valid_cpt[mask]  # n x 2, valid entries for each class
        cpt_hms.append(self.gen_map((hl, wl), xy))  # each 1 x hl x wl
    
    cpt_hm = np.concatenate(cpt_hms, axis=0) 

    item = {
        "cpt_hm": cpt_hm,
        "cpt_off": cpt_off,
        "cpt_ind": cpt_ind,
        "cpt_mask": cpt_mask,
        "wh": wh,
        "cls_id": cls_id,
    }

    return item  # everything needed for training!

def check_dataloader(dl):
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, v.shape)




    # device = "cuda" if torch.cuda.is_available() else 'cpu'
    # #device = 'cpu'
    # print(f'using device: {device}')

    # # ImageNet statistic
    # mean = sum([0.485, 0.456, 0.406]) / 3.0
    # std = sum([0.229, 0.224, 0.225]) / 3.0

    # folders = [f'data/F{i}' for i in range(12)]
    # folders2 = [f'data/T{i}' for i in range(8)]

    # augmentations = [
    #     A.RandomBrightnessContrast(brightness_limit=0.2, 
    #         contrast_limit=0.2, p=0.5),
    #     A.Blur(p=0.5),
    #     A.GaussNoise(p=0.5),
    # ]
    # transform = Transformation(h, w, mean, std, 
    #     bbox_format='pascal_voc', augmentations=augmentations, 
    #     normalize=True, resize_crop=False, bboxes=True)

    # transform2 = Transformation(h, w, mean, std, 
    #     bbox_format='pascal_voc', augmentations=[],  # no augmentation 
    #     normalize=True, resize_crop=False, bboxes=True)  # normalize!

    # data = SARdata(folders, h, w, seq_len=11, use_custom_bboxes=True, 
    #     cache=False, transform=transform, csw=5, isw=13)
    # data2 = SARdata(folders2, h, w, seq_len=11, use_custom_bboxes=False, 
    #     cache=False, transform=transform2, csw=1, isw=13)
    # data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
    #     num_workers=num_workers, drop_last=True)
    # data_loader2 = DataLoader(data2, batch_size=1, shuffle=False,
    #     num_workers=num_workers)

    # check_dataloader(data_loader)
    # check_dataloader(data_loader2)


