from contextlib import ExitStack
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.augmentations.bbox_utils import (
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
)
import numpy as np
import logging
import os
import sys
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data

# conda install -c conda-forge tensorboard
from torch.utils.tensorboard import SummaryWriter

# https://github.com/cheind/pytorch-blender
from blendtorch import btt


def item_transform(item, nhuman_max=6, source_format='pascal_voc', 
    target_format='coco'):
    image = item["image"]  # h x w x 3
    bboxes = item['bboxes']  # nhuman x 4
    cids = item["cids"]  # nhuman,  
    # class label of human = 1, no object = 0

    # to be stacked by default collate_fn:
    bboxes_ = np.zeros((nhuman_max, 4), dtype=bboxes.dtype)
    cids_ = np.zeros((nhuman_max, ), dtype=cids.dtype)

    if len(bboxes) != 0:
        h, w = image.shape[:2]
        
        bboxes = convert_bboxes_to_albumentations(bboxes, source_format,
            rows=h, cols=w, check_validity=True)
        bboxes = convert_bboxes_from_albumentations(bboxes, target_format,
            rows=h, cols=w, check_validity=True)
        
        # do further albumentations transformations here...

        len_valid = len(bboxes)
        # for stacking:
        bboxes_[:len_valid] = bboxes 
        cids_[:len_valid] = 1

    # we will need to create empty heatmap if no bboxes are given
    # in the other implementations dataset was filtered to only 
    # use non empty images but here we want to include that case!

    # swap channels to format that pytorch accepts:
    image = image.transpose((2, 0, 1))  # 3 x h x w
    item = {
        "image": image,  # 3 x h x w
        'bboxes': bboxes_,  # nhuman_max x 4
        'cids': cids_,  # nhuman_max,
    }
    return item

classes = {
    0: 'background',
    1: 'human', 
}


def iterate(dl, folder='./etc', classes: Dict[int, str] = None):
    """ bbox format of 'coco' expected! -> xmin, ymin, width, height """
    os.makedirs(folder, exist_ok=True)
    DPI=96
    for step, item in enumerate(dl):
        img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        print('Received', img.shape, bboxes.shape, cids.shape)

        H, W = img.shape[-2:]  # img: b x 3 x h x w
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), 
            fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
            
        length = min(img.shape[0], 4)  # visualize at most 4 images of a batch
        for i in range(length):
            axs[i].imshow(img[i].permute(1, 2, 0), origin='upper')
            # bboxes: b x nhuman_max x 4, cids: b x nhuman_max
            for cid, bbox in zip(cids[i],bboxes[i]):
                if cid == 0:
                    continue  # no object -> don't draw bbox
                # Rectangle needs xy of bottom left, width, height
                # we use origin upper => xmin,ymin
                rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,
                   edgecolor='g',facecolor='none')
                axs[i].add_patch(rect)

                # draw lowest corner
                axs[i].scatter(*bbox[:2], c='r')

                cls_ = int(cid.item()) if classes is None else classes[int(cid.item())] 
                axs[i].text(bbox[0]+10, bbox[1]+10, f'class: {cls_}', fontsize=14)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)

        fig.savefig(f'{folder}/{step}.png')
        plt.close(fig)


def main():
    launch_args = dict(
        scene=Path(__file__).parent/'blender.blend',
        script=Path(__file__).parent/'blender.py',
        num_instances=1, 
        named_sockets=['DATA'],
    )

    # here we launch Blender instances:
    with btt.BlenderLauncher(**launch_args) as bl:
        addr = bl.launch_info.addresses['DATA']
        ds = btt.RemoteIterableDataset(addr,  item_transform=item_transform, 
            max_items=16)
        dl = data.DataLoader(ds, batch_size=4, num_workers=4)
        
        iterate(dl)  # visualize, sanity check


if __name__ == '__main__':
    main()