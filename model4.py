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


## custom centernet implementation ##

def gen_map(self, shape, xy: np.ndarray, mask=None, sigma=2, cutoff=1e-3, 
            normalize=False, bleed=True):
        """
        Generates a single belief map of 'shape' for each point in 'xy'.
        Parameters
        ----------
        shape: tuple
            h x w of image
        xy: n x 2
            n points with x, y coordinates (image coordinate system)
        mask: n,
            zero-one mask to select points from xy
        sigma: scalar
            gaussian sigma
        cutoff: scalar
            set belief to zero if it is less then cutoff
        normalize: bool
            whether to multiply with the gaussian normalization factor or not
        Returns
        -------
        belief map: 1 x h x w
        """
        n = xy.shape[0]
        h, w = shape[:2] 

        if n == 0:
            return np.zeros((1, h, w), dtype=np.float32)

        if not bleed:
            wh = np.asarray([w - 1, h - 1])[None, :]
            mask_ = np.logical_or(xy[..., :2] < 0, xy[..., :2] > wh).any(-1)
            xy = xy.copy()
            xy[mask_] = np.nan

        # grid is 2 x h x h
        grid = np.array(np.meshgrid(np.arange(w), np.arange(h)), dtype=np.float32)
        # reshape grid to 1 x 2 x h x w
        grid = grid.reshape((1, 2, h, w))
        # reshape xy to n x 2 x 1 x 1
        xy = xy.reshape((n, 2, 1, 1))
        # compute squared distances to joints
        d = ((grid - xy) ** 2).sum(1)
        # compute gaussian
        b = np.nan_to_num(np.exp(-(d / (2.0 * sigma ** 2))))

        if normalize:
            b = b / np.sqrt(2 * np.pi) / sigma  # n x h x w

        # b is n x h x w
        b[(b < cutoff)] = 0

        if mask is not None:
            # set the invalid center point maps to all zero
            b *= mask[:, None, None]  # n x h x w

        b = b.max(0, keepdims=True)  # 1 x h x w

        # focal loss is different if targets aren't exactly 1
        # thus make sure that 1s are at discrete pixel positions 
        b[b >= 0.95] = 1
        return b  # 1 x h x w

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

def _sigmoid(x):
    """
    Clamping values, therefore this activation function can
    be used with Focal Loss.
    """
    y = torch.clamp(torch.sigmoid(x), min=1e-4, max=1-1e-4)
    return y


def gather_from_maps(maps: torch.Tensor, ind: torch.Tensor,
                     mask: torch.Tensor = None):
    """
    Implementation of 'looking through' maps at specified points.
    The points are given as indices from 0 to h * w.
    Where h and w are height and width of the maps.
    Parameters
    ----------
    maps: b x s x h x w
    ind: b x n
    mask: b x n
    Returns
    -------
    b x n x s or
    b x valid n x s (using a mask)
    n ... max num. of objects
    h, w ... low resolution height and width
    Possible calls
    --------------
    center point offset
        map: b x 2 x h x w, ind: b x n
    center point bounding box width and height
        map: b x 2 x h x w, ind: b x n
    """
    maps = maps.permute(0, 2, 3, 1).contiguous()  # b x h x w x s
    maps = maps.view(maps.size(0), -1, maps.size(3))  # b x hw x s
    s = maps.size(2)
    ind = ind.unsqueeze(2).expand(-1, -1, s)  # b x n x s
    maps = maps.gather(1, ind)  # from: b x hw x s to: b x n x s
    if mask is not None:  # b x n
        mask = mask.unsqueeze(2).expand_as(maps)  # b x n x s
        maps = maps[mask]  # b x valid n x s
    return maps  # b x valid n x s or b x n x s


class FocalLoss(nn.Module):

    def __init__(self, alpha=2, beta=4):
        """
        Keep in mind that the given out map, which is the
        network output must be clamped: 0 < p < 1 for each
        pixel values p in the out map!! Otherwise loss is Nan.
        E.g. use clamped sigmoid activation function!
        No mask needed for this loss, the ground truth heat maps
        are produced s.t. all map peaks are valid.
        Parameters
        ----------
        alpha: focal loss parameter from Center Net Paper
        beta: focal loss parameter from Center Net Paper
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, out, target):
        """
        Modified focal loss. Runs faster and costs a little bit more memory.
        The out parameter the network output which contains
        the center point heat map predictions.
        Parameters
        ----------
        out : (b x num_classes x h x w)
          network output, prediction
        target : (b x num_classes x h x w)
          ground truth heat map
        Returns
        -------
        focal loss : scalar tensor
        """
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_weights = torch.pow(1 - out, self.alpha)
        neg_weights = (torch.pow(1 - target, self.beta)
                       * torch.pow(out, self.alpha))

        # masks are zero one masks
        pos_loss = pos_weights * torch.log(out) * pos_mask
        neg_loss = neg_weights * torch.log(1 - out) * neg_mask

        # num. of peaks in the ground truth heat map
        num_p = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        loss = 0
        if num_p == 0:  # no peaks in ground truth heat map
            loss = loss - neg_loss
        else:  # normalize focal loss
            loss = loss - (pos_loss + neg_loss) / num_p
        return loss


class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, out, target, ind, mask):
        """
        Calculate the offset loss (caused by discrete pixels)
        or the size loss (bounding box regression to width and height).
        Parameters
        ----------
        out : b x 2 x h x w
        target : b x n x 2
            ground truth offset of center or key points or
            the bounding box dimensions
        ind : b x n
            indices at which we extract information from out,
            therefore low resolution indices, each index is a scalar
            in range 0 to h * w (index in heat map space)
        mask : b x n
            mask out not annotated or visible key or center points
        n ... max num. of objects
        h, w ... low resolution height and width of feature map
        Returns
        -------
        L1 loss : scalar tensor
        """
        pred = gather_from_maps(out, ind)  # b x n x 2
        mask = mask.unsqueeze(-1)  # b x n x 1
        loss = F.l1_loss(pred * mask, target * mask, reduction="sum")
        loss = loss / (mask.sum() + 1e-4)
        return loss


class CenterLoss(nn.Module):

    def __init__(self):
        super().__init__()
        # set criterion to calculate...
        self.crit_hm = FocalLoss()  # ... loss on heat map
        self.crit_off = L1Loss()  # ... loss on center point offset
        self.crit_wh = L1Loss()  # ... loss on bounding box regression
        self.names = ("total_loss", "cpt_hm_loss", "cpt_off_loss", "wh_loss")

    def forward(self, output, batch):
        """
        Combine the different loss terms to a total loss for a
        center point detection task. Loss term weights from Center Net Paper.
        Parameters
        ----------
        output: dictionary with output tensors
            network output of shape b x s x h x w,
            where s can be the number of classes or 2 for offset or
            bounding box regression
        batch: dictionary with target tensors
            shapes are either b x s x h x w (heat map) or
            b x n x 2 (offset or bounding box regression)
        Returns
        -------
        tuple of total loss (weighted sum of individual terms) and
        a dict of "loss_term_name": "loss_term_value" pairs to track
        the individual contribution of each term
        """
        total_loss = 0
        # apply activation function
        cpt_hm = _sigmoid(output["cpt_hm"])

        # calculate the loss on the center point heat map
        cpt_hm_loss = self.crit_hm(cpt_hm, batch["cpt_hm"])
        total_loss += cpt_hm_loss

        # calculate the loss of the center point offsets
        cpt_off_loss = self.crit_off(output["cpt_off"], batch["cpt_off"],
                                     batch["cpt_ind"], batch["cpt_mask"])
        total_loss += cpt_off_loss

        # calculate the loss on the bounding box dimensions
        wh_loss = self.crit_wh(output["wh"], batch["wh"],
                               batch["cpt_ind"], batch["cpt_mask"])
        total_loss += 0.1 * wh_loss

        # keep track of individual loss terms
        loss_stats = {
            "cpt_hm_loss": cpt_hm_loss, "cpt_off_loss": cpt_off_loss,
            "wh_loss": wh_loss, "total_loss": total_loss
        }

        # only store each loss tensor's value!
        loss_stats = {k: v.item() for k, v in loss_stats.items()}
        # total_loss is still a tensor, needed for backward computation!
        return total_loss, loss_stats

BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, residual_root=False, return_levels=False,
                 pool_size=7, linear_root=False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self):
        model_path = join(os.path.dirname(__file__), "./models/dla34-ba72cf86.pth")
        model_weights = torch.load(model_path)
        self.load_state_dict(model_weights)


def dla34(pretrained, **kwargs):  # DLA-34
    model = DLA([1, 1, 1, 2, 2, 1],
                [16, 32, 64, 128, 256, 512],
                block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model()
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x


def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, std=0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DLASeg(nn.Module):
    def __init__(self, base_name, heads,
                 pretrained=True, down_ratio=4, head_conv=256):
        super(DLASeg, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.heads = heads
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](
            pretrained=pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
                if 'hm' in head:
                    fc[-1].bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            else:
                fc = nn.Conv2d(channels[self.first_level], classes,
                               kernel_size=1, stride=1,
                               padding=0, bias=True)
                if 'hm' in head:
                    fc.bias.data.fill_(-2.19)
                else:
                    fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        print([x_.shape for x_ in x])
        x = self.dla_up(x[self.first_level:])
        print(x.shape)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return ret


# get centernet model
def get_model(heads, head_conv=256, down_ratio=4):
    model = DLASeg('dla34', heads,
                   pretrained=True,
                   down_ratio=down_ratio,
                   head_conv=head_conv)
    return model

def _nms(heat, kernel=3):
    # select padding to keep map dimensions
    pad = (kernel - 1) // 2
    # due to stride 1 and kernel size 3 a peak can be present
    # at most 9 times in hmax and can effect a 5x5 area of heat
    # => in this way we create a more sparse heat map as we would
    # get if a 3x3 kernel with stride 3 was used!
    # this nms method can detect with a minimum distance of 2
    # pixels in between center point detection peaks
    hmax = F.max_pool2d(heat, kernel, 1, pad)
    keep = (hmax == heat).float()  # zero-one mask
    # largest elements remain, others are zero
    return heat * keep  # keeps heat dimensions!


def _topk(heat: torch.Tensor, k):
    """
    Enhances the torch.topk version
    torch.topk(input, k, dim=-1) -> (values, indices)
    indices of the input tensor, values and indices
    are sorted in descending order!
    Parameters
    ----------
    heat: b x c x h x w
        model output heat map
    k: int
        find the k best model output positions
    Returns
    -------
    (topk_score, topk_inds, topk_cids, topk_ys, topk_xs)
    topk_score: b x k
        scores are values form peaks in h x w plane, over multiple
        classes c(or channels)
    topk_inds: b x k
        indices values [0, h * w), over multiple classes
    topk_cids: b x k
        [0, num. of classes), class index to which the score and inds
        belong
    -> each entry in inds can be given as (x, y) coordinate tuple
    topk_ys: b x k
    topk_xs: b x k
    """

    batch, cat, height, width = heat.size()
    topk_scores, topk_inds = torch.topk(heat.view(batch, -1), k)
    topk_cids = torch.true_divide(topk_inds, (height * width)).int()
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_cids, topk_ys, topk_xs


def _gather_feat(feat, ind, mask=None):
    """
    Parameters
    ----------
    feat: b x h * w x c
    ind: b x k
    mask: b x k
    Returns
    -------
    without mask: a x d x c
    with mask: ??
    """
    dim = feat.size(2)  # c
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)  # b x k x c
    feat = feat.gather(1, ind)  # b x k x c
    if mask is not None:  # TODO
        mask = mask.unsqueeze(2).expand_as(feat)  # b x k x c
        # if mask is bool -> flat with length of mask.sum()
        feat = feat[mask]  # ??
        feat = feat.view(-1, dim)  # ??
        raise NotImplementedError

    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    If the network output is given as b x c x h x w and the
    indices at which we want to look at [0, h * w) with shape
    of b x k -> return b x k x c which are the entries for each
    channel and batch of the feature map h x w at the given indices!
    Parameters
    ----------
    feat: b x c x h x w
    ind: b x k
    Returns
    -------
    b x k x c, where k indices are provided for h * w values
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # b x h x w x c
    feat = feat.view(feat.size(0), -1, feat.size(3))  # b x h * w x c
    feat = _gather_feat(feat, ind)  # b x k x c
    return feat  # b x k x c


def decode(out, k):
    """
    From network output to center point detections.
    Parameters
    ----------
    out: model output
        cpt_hm: b x c x h x w
            where c is the num. of object classes, center point map
        wh: b x 2 x h x w
            width, height prediction map of bounding box
        cpt_off: b x 2 x h x w
            offset prediction map cause by going discrete
    k: scalar
        top k of center point peaks are considered
    Returns
    -------
    tensor b x k x 6 is a concatenation of:
        topk_bboxes b x k x 4,
        topk_scores b x k x 1,
        topk_class number b x k x 1
    ...in exact that order!
    """
    cpt_hm = torch.sigmoid(out["cpt_hm"])
    cpt_off = out["cpt_off"]
    wh = out["wh"]

    """ import matplotlib.pyplot as plt
    plt.title("cpt_hm in decode")
    plt.imshow(cpt_hm.detach().clone().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        cmap="Greys", vmin=0, vmax=1.0, origin='upper')
    plt.savefig("debug/debug01.png") """

    b = cpt_hm.size(0)
    cpt_hm = _nms(cpt_hm)  # b x c x h x w

    """ import matplotlib.pyplot as plt
    plt.title("cpt_hm in decode post nms")
    plt.imshow(cpt_hm.detach().clone().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        cmap="Greys", vmin=0, vmax=1.0, origin='upper')
    plt.savefig("debug/debug02.png") """

    # each of shape: b x k
    topk_scores, topk_inds, topk_cids, topk_ys, topk_xs = _topk(cpt_hm, k)

    """ import matplotlib.pyplot as plt
    plt.title("cpt_hm in decode post nms with topk")
    plt.imshow(cpt_hm.detach().clone().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        cmap="Greys", vmin=0, vmax=1.0, origin='upper')
    plt.scatter(topk_xs.detach().clone().cpu(), 
        topk_ys.detach().clone().cpu(), marker="x")
    plt.savefig("debug/debug03.png") """

    topk_cpt_off = _transpose_and_gather_feat(cpt_off, topk_inds)  # b x k x 2

    # each of shape: b x k
    topk_xs = topk_xs.view(b, k, 1) + topk_cpt_off[..., 0:1]
    topk_ys = topk_ys.view(b, k, 1) + topk_cpt_off[..., 1:2]

    topk_wh = _transpose_and_gather_feat(wh, topk_inds)  # b x k x 2
    topk_cids = topk_cids.view(b, k, 1).float()  # b x k x 1
    topk_scores = topk_scores.view(b, k, 1)  # b x k x 1

    # bboxes, coco format: x, y, width, height; b x k x 4
    topk_bboxes = torch.cat([topk_xs - topk_wh[..., 0:1] / 2,
                             topk_ys - topk_wh[..., 1:2] / 2,
                             topk_wh[..., 0:1],
                             topk_wh[..., 1:2]], dim=-1)
    detections = torch.cat([
        topk_bboxes,
        topk_scores,
        topk_cids,
    ], dim=2)  # b x k x 6

    # for each item in the batch return the top k bboxes together
    # with the corresponding scores and class ids
    return detections  # b x k x 6


def filter_dets(dets, thres):
    """
    Parameters
    ----------
    dets: b x k x 6
    thres: scalar
    """
    b = dets.size(0)
    scores = dets[..., 4]  # b x k
    
    mask = scores >= thres  # b x k
    filtered_dets = dets[mask]  # b * k_filtered x 6
    return filtered_dets.view(b, -1, 6)

class AverageMeter:
    """Compute and store the average and current value.
    Examples::
        >>> # 1. Initialize a meter to record loss
        >>> losses = AverageMeter()
        >>> # 2. Update meter after every mini-batch update
        >>> losses.update(loss_value, batch_size)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter:
    """Store the average and current value for a set of metrics.
    Examples::
        >>> # 1. Create an instance of MetricMeter
        >>> metric = MetricMeter()
        >>> # 2. Update using a dictionary as input
        >>> input_dict = {'loss_1': value_1, 'loss_2': value_2}
        >>> metric.update(input_dict)
        >>> # 3. Convert to string and print
        >>> print(str(metric))
    """

    def __init__(self, delimiter='\t'):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter

    def update(self, input_dict):
        if input_dict is None:
            return

        if not isinstance(input_dict, dict):
            raise TypeError(
                'Input to MetricMeter.update() must be a dictionary'
            )

        for k, v in input_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def to_writer(self, writer, tag, n_iter):
        for name, meter in self.meters.items():
            writer.add_scalar(f"{tag}/{name}", meter.val, n_iter)

    def get_avg(self, tag):
        return self.meters[tag].avg

    def get_val(self, tag):
        return self.meters[tag].val

    def __str__(self):
        output_str = []
        for name, meter in self.meters.items():
            output_str.append(
                '{} {:.4f} ({:.4f})'.format(name, meter.val, meter.avg)
            )
        return self.delimiter.join(output_str)

def train(epoch, model, optimizer, dataloader, loss_fn, writer, opt):
    device = next(model.parameters()).device
    model.train()

    meter = MetricMeter()

    with tqdm(total=len(dataloader)) as pbar:
        for i, batch in enumerate(dataloader):
            batch = {k: v.to(device=device) for k, v in batch.items()}

            output = model(batch["image"])
            loss, loss_dict = loss_fn(output, batch)

            meter.update(loss_dict)
            meter.to_writer(writer, "Train", 
                n_iter=(epoch - 1) * len(dataloader) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())
            pbar.update()
        pbar.close()

    return meter

@torch.no_grad()
def eval(epoch, model, dataloader, loss_fn, writer):
    device = next(model.parameters()).device
    model.eval()

    meter = MetricMeter()

    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device=device) for k, v in batch.items()}

        output = model(batch["image"])
        _, loss_dict = loss_fn(output, batch)

        meter.update(loss_dict)
        meter.to_writer(writer, "Val", n_iter=(epoch - 1) * len(dataloader) + i)

    return meter

@torch.no_grad()
def mAP(model, data_loader, writer):
    device = next(model.parameters()).device
    model.eval()

    # have to create ground truth and result json files
    # while looping over for official mAP calculation:
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        images = batch['images']
        pred = model(images)
        
    mAP = 0
    writer.add_scalar('mAP', mAP.item(), None)
    return mAP

## end of centernet implementation ##  


def autopad(k, p=None):  # kernel, padding
    if p is None:  # Pad to 'same'
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, s=1, p=None, g=1, act=True, deform=False):
        # chi ... input channels
        # cho ... output channels
        super().__init__()
        if deform:
            self.conv = ops.DeformConv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)
        else:
            self.conv = nn.Conv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)

        self.bn = nn.BatchNorm2d(cho)

        if act is True:
            self.act = nn.ReLU()  # nn.SiLU()
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, chi, cho, k=1, s=1, p=None, g=1, act=True):  
        super().__init__()
        self.conv = Conv(chi * 4, cho, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], 
            x[..., ::2, 1::2], x[..., 1::2, 1::2]], dim=1))
        

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, chi, cho, shortcut=True, g=1, e=0.5):
        super().__init__()
        chh = int(cho * e)  # hidden channels
        self.cv1 = Conv(chi, chh, 1, 1)
        self.cv2 = Conv(chh, cho, 3, 1, g=g)
        self.add = shortcut and chi == cho

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ConvLSTM(nn.Module):  
    # focused LSTM:
    # no forget gate, focus on either external x(t) or
    # internal y(t-1) information! => less parameters

    def __init__(self, chh, cho, batch_first=True):  
        # chh ... hidden channels
        super().__init__()
        self.batch_first = batch_first
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # increase receptive field:
        # (decrease spatial dimensions)
        #self.focus = Focus(chi=1, cho=16, k=3) 

        # keep spatial dimensions
        self.focus = Conv(chi=1, cho=16, k=3, deform=False)
        # focus on regions of interest:
        self.dconv = Conv(chi=16, cho=32, k=3, deform=False)

        # sigmoid as input activation when rare events are interesting:
        self.Z  = Conv(chi=32, cho=chh, k=3, act=self.sigmoid)  # self.tanh
        self.I = Conv(chi=chh, cho=chh, k=3, act=self.sigmoid)
        self.O = Conv(chi=chh, cho=chh, k=3, act=self.sigmoid)

        # 1x1 convolution to get desired output channels:
        self.final = Conv(chi=chh, cho=cho, k=1)  

    def reset_states(self, shape):
        device = next(self.parameters()).device
        self.y = torch.zeros(shape, device=device)  # at t-1
        self.c = torch.zeros(shape, device=device)  # at t-1

    def init_weights(self):
        def init(m):
            if type(m) == nn.Conv2d:
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.apply(init)  # recursively to every submodule

    def forward(self, x):
        if self.batch_first:  # bsz x seq_len x chi x h x w
            x = x.permute(1, 0, 2, 3, 4)
            # -> seq_len x bsz x chi x h x w

        # go through elements of input sequence:
        for i, x_ in enumerate(x):
            # pre-process input:
            x_ = self.focus(x_)  # bsz x 1 x h x w
            x_ = self.dconv(x_)

            # input activations:
            z = self.Z(x_)  # has hidden cell dimension
            if i == 0:  # initialize self.c, self.y
                self.reset_states(z.shape)  # bsz x chh x h/2 x w/2
            
            # input gate:
            i = self.I(self.y)  # feed y(t-1)
            # output gate:
            o = self.O(self.y)
            # memory cell state:
            self.c += i * z
            # lstm output:
            self.y = o * self.tanh(self.c)

        return self.final(self.y)  # use last lstm output


class Model(nn.Module):
    def __init__(self, lstm_kwargs, centernet_kwargs):
        super().__init__()
        # get_model -> centernet
        self.centernet = get_model(**centernet_kwargs)
        self.lstm = ConvLSTM(**lstm_kwargs)

    def forward(self, x):
        # x: bsz x seq_len x h x w <- from dataloader
        x = x.unsqueeze(2)  # each element is a grayscale image
        # x: bsz x seq_len x 1 x h x w <- lstm
        x = self.lstm(x)
        x = self.centernet(x)
        return x


def check_dataloader(dl):
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, v.shape)


if __name__ == '__main__':
    model = get_model({"cpt_hm": 1, "cpt_off": 2, "wh": 2})
    model(torch.randn(1, 3, 512, 512))
    exit()
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(f'using device: {device}')

    # hyper parameters
    h, w = 512, 640
    batch_size = 1
    nc = 1  # number of classes
    num_workers = 0
    num_epochs = 10
    lr = 0.01
    save_interval = 15
    best_model_tag = 'best_model'
    epoch_model_tag = 'model_epoch'
    resume = False
    model_path_to_load = r"models/best_model"
    chh = 64  # lstm hidden channels
    lr_step_size = 10  # number of epochs till lr decrease

    # ImageNet statistic
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # for grayscale use mean over RGB channels:
    mean = sum(mean) / 3.0
    std = sum(std) / 3.0

    os.makedirs("./models", exist_ok=True)  # create 'models' folder

    lstm_kwargs={'chh': chh, 'cho': 3, 'batch_first': True}

    # heads - head_name: num. channels of model
    loss_fn = CenterLoss()
    centernet_kwargs={'heads': {"cpt_hm": 1, "cpt_off": 2, "wh": 2}}
    model = Model(lstm_kwargs, centernet_kwargs)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=lr_step_size, gamma=0.1)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model with: {num_params/10**6:.2f}M number of parameters')

    if resume:
        print('load checkpoint...')
        checkpoint = torch.load(model_path_to_load)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
    else:
        print('train from scratch...')
        start_epoch = 0
        best_metric = 0

    folders = [f'data/F{i}' for i in range(12)]
    folders2 = [f'data/T{i}' for i in range(8)]

    augmentations = [
        A.RandomBrightnessContrast(brightness_limit=0.2, 
            contrast_limit=0.2, p=0.5),
        A.Blur(p=0.5),
        A.GaussNoise(p=0.5),
    ]
    transform = Transformation(h, w, mean, std, 
        bbox_format='pascal_voc', augmentations=augmentations, 
        normalize=True, resize_crop=False, bboxes=True)

    transform2 = Transformation(h, w, mean, std, 
        bbox_format='pascal_voc', augmentations=[],  # no augmentation 
        normalize=True, resize_crop=False, bboxes=True)  # normalize!

    data = SARdata(folders, h, w, seq_len=11, use_custom_bboxes=True, 
        cache=False, transform=transform, csw=5, isw=13)
    data2 = SARdata(folders2, h, w, seq_len=11, use_custom_bboxes=False, 
        cache=False, transform=transform2, csw=1, isw=13)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True)
    data_loader2 = DataLoader(data2, batch_size=1, shuffle=False,
        num_workers=num_workers)

    check_dataloader(data_loader)
    check_dataloader(data_loader2)

    writer = SummaryWriter()

    eval(model, data_loader2, loss_fn, writer)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        meter = train(model, data_loader, loss_fn, optimizer, writer)
        total_loss = meter.get_avg("total_loss")
        logging.info(f"Loss: {total_loss} at epoch: {epoch} / {start_epoch + num_epochs}")

        meter = eval(model, data_loader2, loss_fn, writer)
        metric = meter.get_avg("total_loss")

        if metric > best_metric:
            torch.save({"model": model.state_dict()}, r"models/" + best_model_tag)
            best_metric = metric

        if (epoch+1) % save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
            }, r"models/" + f"{epoch_model_tag}_{epoch}")

    torch.save({
        "model": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': num_epochs-1,
        'best_metric': best_metric,
    }, r"models/" + "last_model")

