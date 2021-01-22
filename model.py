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

import argparse
import sys 
import logging
sys.path.append('../yolov5')  # to run '$ python *.py' files in subdirectories
logger = logging.getLogger(__name__)

from models.yolo import Model  
from utils.general import check_file, set_logging


if __name__ == '__main__':  # test yolo
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()

    # Create model
    model = Model(opt.cfg)
    model.train()
