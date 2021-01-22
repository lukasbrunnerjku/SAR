import albumentations as A
from albumentations.augmentations.bbox_utils import (
    convert_bboxes_from_albumentations,
    convert_bboxes_to_albumentations,
)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import sys
from glob import glob
import cv2
from pathlib import Path
import json
from typing import Dict, List, Tuple, Union
from multiprocessing.pool import ThreadPool
from numpy import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import dill as pickle

import torch
import torch.nn as nn
from torch.utils import data
from torchvision.utils import make_grid


class SARbase(data.Dataset):

    def __init__(self, folders: List[str], h: int, w: int, cache=False,
        transform=None):
        super().__init__()
        self.folders = folders  # e.g. ['data/F0', ...]
        self.h = h  # image height
        self.w = w  # image width
        self.cache = cache  # store images as path string or np.ndarray 
        self.transform = transform
        
        self.data = self.extract_data(folders)
        self.size = len(self.data)
        if cache:
            self.cache_images()

        self.default_agl = 37
        self.getAGL = {
            'F0': 45, 'F1': 30, 'F2': 38, 'F3': 40, 'F4': 40, 'F5': 30, 'F6': 33,
            'T2': 32, 'T3': 33, 'T4': 33, 'T6': 32,
        }

    def cache_images(self): 
        results = ThreadPool(4).imap(self.imread, (lane['images'] for lane in self.data))
        pbar = tqdm(enumerate(results), total=self.size)
        gb = 0
        for lane_nr, lane_images in pbar:
            self.data[lane_nr]['images'] = lane_images  
            gb += lane_images.nbytes
            pbar.desc = f'Caching images ({gb/1E9:.1f}GB)'

    @staticmethod
    def extract_data(folders: List[str]):
        '''
        extract the relevant information from the folder structure and 
        cluster it lane-wise:

        [{'images': [('.../some_name.tiff', ...], 
          'poses': [pose_M3x4, ...],
          'site': e.g. F0,
          'annotated_image': integer_index, 
          'polys': [[...]], }, 
         {...}, ...]
        
        note: for each image there exists a pose, per lane there are bbox 
        annotations for a single image (bboxes make sense after integration)
        ''' 
        data = []  # each lane is a element (dictionary) in the 'data' list
        for folder_path in folders:
            pose_files = sorted(glob(os.path.join(folder_path, 'Poses', '*.json')))
            label_files = sorted(glob(os.path.join(folder_path, 'Labels', '*.json')))
            for pose_file, label_file in zip(pose_files, label_files):
                lane = {}
                poses = json.load(open(pose_file, 'rb'))['images']
                labels = json.load(open(label_file, 'rb'))['Labels']
                lane_nr = os.path.splitext(os.path.basename(pose_file))[0]
                base_path = os.path.join(folder_path, 'Images', lane_nr)

                image_paths = []
                pose_matrices = []
                for pose in poses:
                    image_paths.append(os.path.join(base_path, pose['imagefile']))
                    pose_matrices.append(np.asarray(pose['M3x4']))

                polys = []
                if isinstance(labels, list) and len(labels):  
                    for label in labels:
                        polys.append(label['poly'])
                    # each lane has a single annotated image (the integral image)
                    lane['annotated_image'] = image_paths.index(os.path.join(base_path, 
                        labels[0]['imagefile']))  
                    # polygons: [ [[x1, y1], ...next point], ...next polygon ]
                    lane['polys'] = polys
                else:  # otherwise empty => no annotations (no humans)
                    lane['annotated_image'] = lane['polys'] = None

                # save site for drone height information
                lane['site'] = folder_path.split('/')[-1]  # e.g. 'data/F0' -> F0
                
                # for custom bbox retrieval
                lane['line'] = lane_nr
                
                # we have the camera extrinsics for each image in the lane
                lane['images'] = np.array(image_paths)
                lane['poses'] = np.array(pose_matrices)

                data.append(lane)  # each pose and label file correspondes to a lane
                
        return data

    @staticmethod
    def show_images(images: np.ndarray, mask: np.ndarray = None, figsize=None):
        fig = plt.figure(figsize=figsize)
        if mask is not None:  # k,
            images = images[mask, ...]  # k x h x w
        images = torch.as_tensor(images).unsqueeze(1)  # n(k) x 1 x h x w
        grid = make_grid(images, nrow=int(np.sqrt(images.size(0))))  # 3 x h x w
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.show()

    @staticmethod
    def show_image(image: np.ndarray, cmap: str = 'gray', figsize=None):
        fig = plt.figure(figsize=figsize)
        plt.imshow(image, cmap=cmap)
        plt.axis('off')
        plt.show()

    @staticmethod
    def load_camera_params(param_folder: str = 'calibration/parameters'):
        ''' Camera intrinsics K and distortion coefficients. '''
        K = np.load(os.path.join(param_folder, 'K.npy'))
        dist_coeffs = np.load(os.path.join(param_folder, 'dist_coeffs.npy'))
        return K, dist_coeffs

    @staticmethod
    def imread(paths: List[str], K=None, dist_coeffs=None) -> np.ndarray:
        images = []
        for path in paths:
            image = cv2.imread(path, cv2.IMREAD_ANYDEPTH)  # 16 bit image
            image = cv2.normalize(image, dst=None, alpha=0, beta=2**16 - 1, 
                norm_type=cv2.NORM_MINMAX)
            image = (image >> 8).astype(np.uint8)  # 8 bit image
            
            if K is not None and dist_coeffs is not None:  # undistort images
                h, w = image.shape
                # new camera intrinsics based on free scaling parameter
                refined_K, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, 
                    (w, h), 1, (w, h))
                x, y, w, h = roi
                image = image[y:y+h, x:x+w]
                image = cv2.undistort(image, K, dist_coeffs, None, refined_K)
            
            images.append(image)

        return np.asarray(images)  # n_paths x h x w

    @staticmethod
    def integrate(images: Union[List[str], np.ndarray], pose_matrices: List[np.ndarray], 
        K: np.ndarray, z: float = 30.0, idc: int = None) -> Tuple[np.ndarray, np.ndarray]:
        '''
        integral :=
        a mapping of a image sequence under perspective to another image view

        images contains the image file paths or the np.ndarray images and pose_matrices 
        the M3x4 matrices where M3x4 = (R3x3, t3x1) => thus without intrinsics K

        z is the height of the camera (drone) in meters (units must be the same
        as used when calculating the extrinsics)
        '''
        sequence_len = len(images)

        if idc is None:  # otherwise given and must not be at the center
            idc = sequence_len // 2  # index of center image = center of images (if odd length)
        
        if images.ndim == 1:  # file paths
            images = SARbase.imread(images)  # 8 bit images from path
        
        h, w = images.shape[1:]  # gray scale images
        integral = np.zeros((h, w), np.float64)
        
        # inverse of the intrinsic mapping
        K_inv = np.linalg.inv(K)

        Mc = pose_matrices[idc]
        Rc = Mc[:, :3]  # 3 x 3
        tc = Mc[:, 3:]  # 3 x 1

        for idx in range(sequence_len):
            if idx != idc:
                Mr = pose_matrices[idx]  # 3 x 4
                Rr = Mr[:, :3]  # 3 x 3
                tr = Mr[:, 3:]  # 3 x 1

                # relative translation and rotation
                R_rel = Rc @ Rr.T  # 3 x 3
                t_rel = tc - R_rel @ tr  # 3 x 1

                B = K @ R_rel @ K_inv
                B[:, 2:] += K @ t_rel / z
                warped = cv2.warpPerspective(images[idx], B, (w, h))

                integral += warped
            else:
                integral += images[idx]

        integral /= sequence_len
        # 8 bit gray scale integral image
        integral = cv2.normalize(integral, None, 
            0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return images, integral  # original images, integral image

    def __len__(self):
        return self.size

    def _float(self, x: float):
        return float(f'{x:0.2f}')


class SARintegral(SARbase):

    def __init__(self, folders: List[str], h: int, w: int, seq_len,
        cache=False, transform=None):
        super().__init__(folders, h, w, cache, transform)
        self.seq_len = seq_len
        self.K, _ = self.load_camera_params()

    def __getitem__(self, idx):
        lane = self.data[idx]  # select lane
        start = lane['annotated_image'] - self.seq_len // 2
        images = lane['images'][start:start + self.seq_len]
        pose_matrices = lane['poses'][start:start + self.seq_len]
        images, integral = self.integrate(images, pose_matrices, self.K)

        if self.transform is not None:  # data augmentation
            images, integral = self.transform(images, integral)
        
        return images, integral


class SARdata(SARbase):

    def __init__(self, folders: List[str], h: int, w: int, seq_len: int, csw: int,
        isw: int, n_max=15, use_custom_bboxes=False, cache=False, transform=None):
        super().__init__(folders, h, w, cache, transform)
        self.csw = csw  # center sampling window
        self.isw = isw  # image sampling window
        self.seq_len = seq_len  # num. of images to return by getitem
        self.K, _ = self.load_camera_params()
        self.n_max = n_max  # max. number of bbox annotations
        self.use_custom_bboxes = use_custom_bboxes
        if use_custom_bboxes:
            self.bboxes = self.load_custom_bboxes('./custom_bboxes/result_file1.pkl',
                './custom_bboxes/result_file2.pkl')

    def load_custom_bboxes(self, *paths):
        bboxes = {}
        for path in paths:
            assert os.path.exists(path), f'{path} does not exist'
            with open(path, 'rb') as fp:
                bboxes.update(pickle.load(file=fp))
        return bboxes

    @staticmethod
    def get_wrapped_images(images: Union[List[str], np.ndarray], 
        pose_matrices: List[np.ndarray], K: np.ndarray, z: int, 
        idc: int = None) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = len(images)

        if idc is None:  # otherwise given and must not be at the center
            idc = seq_len // 2  # index of center image = center of images (if odd length)
        
        if images.ndim == 1:  # file paths
            images = SARbase.imread(images)  # 8 bit images from path
        
        h, w = images.shape[1:]  # gray scale images
        wrapped = np.empty_like(images)  # seq_len x h x w
        
        # inverse of the intrinsic mapping
        K_inv = np.linalg.inv(K)

        Mc = pose_matrices[idc]
        Rc = Mc[:, :3]  # 3 x 3
        tc = Mc[:, 3:]  # 3 x 1

        for idx in range(seq_len):
            if idx != idc:
                Mr = pose_matrices[idx]  # 3 x 4
                Rr = Mr[:, :3]  # 3 x 3
                tr = Mr[:, 3:]  # 3 x 1

                # relative translation and rotation
                R_rel = Rc @ Rr.T  # 3 x 3
                t_rel = tc - R_rel @ tr  # 3 x 1

                B = K @ R_rel @ K_inv
                B[:, 2:] += K @ t_rel / z
                warped = cv2.warpPerspective(images[idx], B, (w, h))
                wrapped[idx] = warped
            else:
                wrapped[idx] = images[idx]

        return wrapped

    def __getitem__(self, idx):
        lane = self.data[idx]  # select lane
        images = lane['images']
        poses = lane['poses']
        cidx = lane['annotated_image']
        site = lane['site']  # e.g. F0
        line = lane['line']  # e.g. 3

        w = self.csw // 2
        cidx = np.random.randint(cidx - w, cidx + w + 1)  # [a, b)

        w = self.isw // 2
        # shift center if not enough samples available
        while cidx - self.seq_len // 2 < 0:
            cidx += 1
        while cidx + self.seq_len // 2 >= len(images):
            cidx -= 1
        
        # max(cidx - w, 0); ensure valid sampling
        low_idx = np.random.choice(np.arange(max(cidx - w, 0), cidx), 
            replace=False, size=(self.seq_len // 2))
        # min(cidx + w + 1, len(images)); ensure valid sampling
        high_idx = np.random.choice(np.arange(cidx + 1, min(cidx + w + 1, len(images))), 
            replace=False, size=(self.seq_len // 2))

        indices = np.concatenate((low_idx, np.array([cidx]), high_idx))
        indices = indices.astype(np.int32)

        images = images[indices]
        poses = poses[indices]
        z = self.getAGL.get(site, self.default_agl)
        # idc=None because center image is center of those images
        wrapped = self.get_wrapped_images(images, poses, self.K, z)  # seq_len x h x w

        bboxes = np.zeros((self.n_max, 4))
        if self.use_custom_bboxes:
            # List[List[float]], format: x_min, x_max, y_min, y_max
            bboxes_ = self.bboxes[f'img_{site}_{line}_{cidx + 1}']['bbox']
            bboxes_ = np.array(bboxes_)  # n x 4
            # format: pascal_voc - x_min, ymin, x_max, y_max
            bboxes_[:, [1, 2]] = bboxes_[:, [2, 1]]  # center image annotation
        else: 
            # lane['polys'] list of list
            raise NotImplementedError

        # can apply augmentation, scaling, normalization
        if self.transform is not None:
            # add labels to the end of bboxes for albumentations:
            # n x 4 with n x 1
            bboxes_ = np.concatenate([bboxes_, 
                np.arange(len(bboxes_))[:, None]], axis=-1)
            out = self.transform(image=wrapped, bboxes=bboxes_)
            wrapped = out['image']
            bboxes_ = out['bboxes'][:, :4]  # drop added labels

        bboxes[:len(bboxes_)] = bboxes_  # n_max x 4
        cids = np.zeros((len(bboxes), ))  # n_max,
        mask = np.zeros((len(bboxes), ))
        mask[:len(bboxes_)] = 1  # n_max,

        item = {'images': wrapped, 'bboxes': bboxes, 'cids': cids, 'mask': mask}
        return item


def iterate(dl):
    DPI=96
    for step, item in enumerate(dl):
        img, bboxes, cids = item['image'], item['bboxes'], item['cids']
        H, W = img.shape[2:]  # img: b x 3 x h x w
        fig = plt.figure(frameon=False, figsize=(W*2/DPI,H*2/DPI), dpi=DPI)
        axs = [fig.add_axes([0,0,0.5,0.5]), fig.add_axes([0.5,0.0,0.5,0.5]), fig.add_axes([0.0,0.5,0.5,0.5]), fig.add_axes([0.5,0.5,0.5,0.5])]
        for i in range(img.shape[0]):
            axs[i].imshow(img[i].permute(1, 2, 0), origin='upper')
            for cid, bbox in zip(cids[i],bboxes[i]):
                rect = patches.Rectangle(bbox[:2],bbox[2],bbox[3],linewidth=2,edgecolor='r',facecolor='none')
                axs[i].add_patch(rect)
                axs[i].text(bbox[0]+10, bbox[1]+10, f'Class {cid.item()}', fontsize=18)
            axs[i].set_axis_off()
            axs[i].set_xlim(0,W-1)
            axs[i].set_ylim(H-1,0)
        fig.savefig(f'./data/output_{step}.png')
        plt.close(fig)


class Transformation:

        def __init__(self, h, w, mean, std, bbox_format='coco',
            augmentations: List = None, normalize=True,
            resize_crop=False, bboxes=True):
            super().__init__()
            self.resize_crop = resize_crop
            self.normalize = normalize
            self.mean = mean 
            self.std = std 
            self.h = h 
            self.w = w
            self.augmentations = augmentations
            self.bbox_format = bbox_format 
            self.bboxes = bboxes

            # build transformation
            if augmentations is None:
                # standard augmentation if not given explicitly
                transformations = [
                    A.HueSaturationValue(p=0.5),
                    A.ChannelShuffle(p=0.4),
                    A.HorizontalFlip(p=0.2),
                    A.GaussNoise(p=0.3),
                ]
            else:
                transformations = augmentations

            if resize_crop:
                transformations.extend([
                    # rescale an image so that minimum side is equal to max_size,
                    # keeping the aspect ratio of the initial image.
                    A.SmallestMaxSize(max_size=min(h, w)),
                    A.RandomCrop(height=h, width=w),
                ])

            if normalize:  # zero mean, unit variance, range: [0, 1]
                transformations.append(A.Normalize(mean=mean, std=std))  
            
            if bboxes:
                bbox_params = A.BboxParams(
                    # pascal_voc, albumentatons, coco, yolo
                    format=bbox_format,  
                    # < min_area (in pixels) will drop bbox
                    min_area=50,  
                    # < min_visibility (relative to input bbox) will drop bbox
                    min_visibility=0.5,  
                )
            else:
                bbox_params = None

            self.transform_fn = A.Compose(transformations, 
                bbox_params=bbox_params)

        def __call__(self, **items):
            return self.transform_fn(**items)


def convert_bbox_format(bboxes, source_fmt, target_fmt,
    h, w):
    # pascal_voc: [x_min, y_min, x_max, y_max]
    # albumentatons: normalized [x_min, y_min, x_max, y_max]
    # coco: [x_min, y_min, width, height]
    # yolo: normalized [x_center, y_center, width, height]
    if source_fmt != 'albumentations':
        bboxes = convert_bboxes_to_albumentations(bboxes, 
            source_fmt, rows=h, cols=w)
    bboxes = convert_bboxes_from_albumentations(bboxes,
        target_fmt, rows=h, cols=w)
    return bboxes


if __name__ == '__main__':
    folders = [f'data/F{i}' for i in range(12)]
    h, w = 512, 640

    """
    ds = SARintegral(folders, h, w, seq_len=9, cache=True, transform=None)
    images, integral = ds[0]  # original images, integral image
    ds.show_images(images, figsize=(8, 10))
    ds.show_image(integral)
    """

    # ImageNet statistic
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # for grayscale use mean over RGB channels:
    mean = sum(mean) / 3.0
    std = sum(std) / 3.0

    h, w = 512, 640

    augmentations = [
        A.RandomBrightnessContrast(brightness_limit=0.4, 
            contrast_limit=0.4, p=1.0),
        A.GaussNoise(p=1.0),
    ]

    """
    A.OneOf([
        A.RandomBrightnessContrast(),
        # FDA ... fourier domain adaptation
        A.FDA(some_images, read_fn=lambda x: x),
    ], p=1),
    """
    
    transform = None
    transform = Transformation(h, w, mean, std, 
        bbox_format='coco', augmentations=augmentations, 
        normalize=False, resize_crop=False, bboxes=True)

    ds = SARdata(folders, h, w, seq_len=5, use_custom_bboxes=True, cache=False, 
        transform=transform, csw=5, isw=13)
    item = ds[7] 
    mask = item['mask']
    bboxes = item['bboxes']
    images = item['images']

    def show_bboxes(images, bboxes, mask):
        DPI=96
        H, W = images[0].shape
        
        for img in images:
            fig = plt.figure(frameon=False, figsize=(W/DPI,H/DPI), dpi=DPI)
            plt.imshow(img, origin='upper', cmap='gray')
            # coco: x_min, y_min, width, height
            for bbox in bboxes[:int(mask.sum())]:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                rect = patches.Rectangle(bbox[:2], width , height, linewidth=2,
                    edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)
            plt.show()

    show_bboxes(images, bboxes, mask)