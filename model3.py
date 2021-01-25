import torch
from torch.utils.data import DataLoader
from datasets import SARdata, Transformation
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import matplotlib.patches as patches
import torchvision.ops as ops
# tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import os
import argparse
import sys 
import logging
# -> clone yolov5 repo into yolov5 folder 
# to run '$ python *.py' files in subdirectories
sys.path.append('../yolov5')  
logger = logging.getLogger(__name__)

from models.yolo import Model as Yolo
from utils.general import check_file, set_logging
from utils.loss import ComputeLoss
    

def train(model, data_loader, compute_loss, optimizer, writer):
    model.train()
    device = next(model.parameters()).device
    for n_iter, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        images = batch['images']
        pred = model(images)
        
        targets ??
        loss, loss_items = compute_loss(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        writer.add_scalar('Loss/train', loss.item(), n_iter)
        optimizer.step()


@torch.no_grad()
def test(model, data_loader, optimizer, writer):
    pass


@torch.no_grad()
def evaluate(model, data_loader, writer):
    model.eval()
    device = next(model.parameters()).device
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        images = batch['images']
        pred = model(images)
        
    mAP = 0
    writer.add_scalar('mAP', mAP.item(), None)
    return mAP


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
        self.focus = Focus(chi=1, cho=16, k=3)  
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
    def __init__(self, lstm_kwargs, yolo_kwargs):
        super().__init__()
        self.yolo = Yolo(**yolo_kwargs)
        self.lstm = ConvLSTM(**lstm_kwargs)

    def forward(self, x):
        # x: bsz x seq_len x h x w <- from dataloader
        x = x.unsqueeze(2)  # each element is a grayscale image
        # x: bsz x seq_len x 1 x h x w <- lstm
        x = self.lstm(x)
        x = self.yolo(x)
        return x


def check_dataloader(dl):
    batch = next(iter(dl))
    for k, v in batch.items():
        print(k, v.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
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
    yolo_kwargs={'cfg': opt.cfg, 'nc': nc, 'ch': 3}
    model = Model(lstm_kwargs, yolo_kwargs)
    model = model.to(device)

    # hyper-parameters for loss function of yolo
    hyp = dict()
    hyp['cls_pw'] = (1, 0.5, 2.0)  # cls BCELoss positive_weight
    hyp['obj_pw'] = (1, 0.5, 2.0)  # obj BCELoss positive_weight
    hyp['fl_gamma'] = (0, 0.0, 2.0)  # focal loss gamma
    hyp['box'] = (1, 0.02, 0.2)  # box loss gain
    hyp['obj'] = (1, 0.2, 4.0)  # obj loss gain (scale with pixels)
    hyp['cls'] = (1, 0.2, 4.0)  # cls loss gain

    nl = model.model[-1].nl  # number of detection layers
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    imgsz = [h, w]
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers

    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.nc = nc  # attach number of classes to model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
        step_size=lr_step_size, gamma=0.1)

    compute_loss = ComputeLoss(model)  # init loss class

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
        A.RandomBrightnessContrast(brightness_limit=0.4, 
            contrast_limit=0.4, p=0.5),
        A.GaussNoise(p=0.5),
    ]
    
    transform = Transformation(h, w, mean, std, 
        bbox_format='coco', augmentations=augmentations, 
        normalize=True, resize_crop=False, bboxes=True)

    data = SARdata(folders, h, w, seq_len=11, use_custom_bboxes=True, 
        cache=False, transform=transform, csw=5, isw=13)
    data2 = SARdata(folders2, h, w, seq_len=11, use_custom_bboxes=True, 
        cache=False, transform=transform, csw=1, isw=13)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    data_loader2 = DataLoader(data2, batch_size=1, shuffle=False,
        num_workers=num_workers)

    check_dataloader(data_loader)
    check_dataloader(data_loader2)

    writer = SummaryWriter()

    evaluate(model, data_loader2, writer)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train(model, data_loader, compute_loss, optimizer, writer)
        metric = evaluate(model, data_loader2, writer)

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

