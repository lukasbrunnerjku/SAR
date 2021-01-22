from tqdm.notebook import tqdm
import os
from glob import glob
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch import nn 
from torchvision.ops import DeformConv2d 
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval

import sys 
import logging
sys.path.append('../yolov5')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.yolo import Model  


if __name__ == '__main__':                                                                                                                       
    win_size = 9
    model = Model(cfg='../yolov5/models/yolov5s.yaml', 
        ch=win_size, nc=1)
    model.eval()
    model = model.fuse().autoshape()

    img = torch.rand(2, win_size, 512, 640)
    results = model(img)
    results.print()
