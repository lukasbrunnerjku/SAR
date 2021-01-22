import torch
import torchvision.models.detection as m
import torchvision.models as back
from torch.utils.data import DataLoader
from dataset_computer_vision import SARdata
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import matplotlib.patches as patches
import torchvision.ops as ops
# tensorboard --logdir=runs
from torch.utils.tensorboard import SummaryWriter


def train(model, data_loader, optimizer, writer):
    model.train()
    device = next(model.parameters()).device
    for n_iter, (x, y, b) in enumerate(data_loader):
        x = x.to(device)  # images
        y = y.to(device)  # labels
        b = b.to(device)  # number of boxes
        targets = []
        for i in range(len(x)):
            d = {}
            d['boxes'] = y[i, :b[i], :]
            d['labels'] = torch.ones(b[i], dtype=torch.int64, device=device)
            targets.append(d)
        losses = model(x, targets)
        loss = losses["loss_classifier"]
        loss += losses["loss_box_reg"]
        loss += losses["loss_objectness"]
        loss += losses["loss_rpn_box_reg"]
        optimizer.zero_grad()
        loss.backward()
        writer.add_scalar('Loss/train', loss.item(), n_iter)
        optimizer.step()


@torch.no_grad()
def evaluate(model, data_loader, writer):
    y_m = []
    y_t = []
    model.eval()
    device = next(model.parameters()).device
    for x, y, b in data_loader:
        x = x.to(device)
        model.eval()
        out = model(x)
        x = x.cpu().detach()
        for index, res in enumerate(out):
            bboxen = res["boxes"].cpu().detach()
            scores = res["scores"].cpu().detach()
            indices = ops.nms(bboxen, scores, 0.1)
            bboxen = bboxen[indices]
            scores = scores[indices]
            y_model, y_target = compute_mapping(bboxen, scores, y[index, :b[index]])
            y_m = list((*y_m, *y_model))
            y_t = list((*y_t, *y_target))
    mAP = compute_mAP(y_t, y_m)
    writer.add_scalar('mAP', mAP.item(), None)
    return mAP


def show_bbox(images, bboxes, bboxes_target):
    DPI = 96
    H, W = images[0].shape
    for img in images:
        fig = plt.figure(frameon=False, figsize=(W * 2 / DPI, H * 2 / DPI), dpi=DPI)
        plt.imshow(images[4], "gray", origin='upper')
        # coco: x_min, y_min, widht, height
        for bbox in bboxes:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            rect = patches.Rectangle(bbox[:2], width, height, linewidth=2, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
        for bbox in bboxes_target:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            rect = patches.Rectangle(bbox[:2], width, height, linewidth=2, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
        plt.show()
        break


def compute_mapping(bboxes, scores, y):
    matrix_iou = ops.box_iou(bboxes, y)
    matrix_iou[matrix_iou >= 0.05]=1
    matrix_iou[matrix_iou < 0.05]=0
    y_model=[]
    y_target=[]
    bboxes_model = np.ones(len(bboxes))
    bboxes_target = np.ones(len(y))
    for col in range(matrix_iou.shape[1]):
        first_entry = True
        for row in range(matrix_iou.shape[0]):
            if matrix_iou[row,col]==1 and first_entry:
                y_model.append(float(scores[row]))
                y_target.append(1)
                bboxes_model[row]=0
                bboxes_target[col]=0
                first_entry=False
            elif matrix_iou[row,col]==1 and not first_entry:
                bboxes_model[row] = 0
    for bbox in range(bboxes_model.shape[0]):
        if bboxes_model[bbox]==1:
            y_model.append(float(scores[bbox]))
            y_target.append(0)
    for bbox in range(bboxes_target.shape[0]):
        if bboxes_target[bbox]==1:
            y_model.append(0)
            y_target.append(1)
    return y_model, y_target


def compute_mAP(y_true, y_model):
    y_true = torch.tensor(y_true)
    y_model = torch.tensor(y_model)
    topk = torch.topk(y_model,len(y_model))
    y_true = y_true[topk.indices]
    y_model = y_model[topk.indices]
    mAP = []
    recall_old = 0
    true_positiv = false_positiv = false_negativ = 0
    for i in range(len(y_model)):
        y_true_current = torch.tensor(y_true[:i+1])
        y_model_current = torch.tensor(y_model[:i+1])
        y_model_current[y_model_current>0]=1
        true_positiv = y_true_current[y_model_current.to(dtype=torch.bool)].sum()
        false_negativ = torch.tensor(y_true.sum()-true_positiv, dtype=torch.float32)
        false_positiv = y_model_current[torch.tensor(1-y_true_current, dtype=torch.bool)].sum()
        if not true_positiv == 0:
            precision = true_positiv / (true_positiv + false_positiv)
        else:
            precision = 0
        recall = true_positiv / (true_positiv + false_negativ)
        AP = (recall - recall_old)*precision
        mAP.append(AP)
        recall_old = recall
    print(f"TP: {true_positiv}")
    print(f"FP: {false_positiv}")
    print(f"FN: {false_negativ}")
    return torch.tensor(mAP).sum()


def autopad(k, p=None):  # kernel, padding
    if p is None:  # Pad to 'same'
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, chi, cho, k=1, p=None, s=1, deform=False, g=1, act=True):
        # chi ... input channels
        # cho ... output channels
        super().__init__()
        if deform:
            self.conv = ops.DeformConv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)
        else:
            self.conv = nn.Conv2d(chi, cho, k, s, autopad(k, p), groups=g, bias=False)

        self.bn = nn.BatchNorm2d(cho)

        if act is True:
            self.act = nn.SiLU()
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
            x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


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

    def __init__(self, chh, cho):  
        # chh ... hidden channels
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # increase receptive field:
        self.focus = Focus(chi=1, cho=16, k=3)  
        # focus on regions of interest:
        self.dconv = Conv(chi=16, cho=32, k=3, deform=True)

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
        # bsz x seq_len x h x w -> seq_len x bsz x h x w
        x = x.permute(1, 0, 2, 3)

        # go through elements of input sequence:
        for i, x_ in enumerate(x):
            # pre-process input:
            x_ = self.focus(x_)
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
    def __init__(self):
        super().__init__()
        backbone = resnet_fpn_backbone("resnet50", True)
        # 2 classes, one for background!
        self.rcnn = FasterRCNN(backbone, num_classes=2, 
            image_mean=None, image_std=None)
        self.lstm = ConvLSTM()

    def forward(self, x, targets=None):
        x = self.lstm(x)
        return self.rcnn(x, targets) if self.training else self.rcnn(x)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f'using device: {device}')

    # hyper parameters
    batch_size = 1
    num_workers = 0
    num_epochs = 10
    lr = 0.01
    save_interval = 15
    best_model_tag = 'best_model'
    epoch_model_tag = 'model_epoch'

    resume = False
    model_path_to_load = r"models/best_model"

    model = Model()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'model with: {num_params/10**6}M number of parameters')

    if resume:
        print('load checkpoint...')
        checkpoint = torch.load(model_path_to_load)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_metric = checkpoint['best_metric']
    else:
        print('train from scratch...')
        start_epoch = 0
        best_metric = 0

    folders = [f'data/F{i}' for i in range(12)]
    folders2 = [f'data/T{i}' for i in range(8)]
    h, w = 512, 640
    data = SARdata(folders, h, w, seq_len=13, use_custom_bboxes=True, 
        cache=False, transform=None, csw=5, isw=19)
    data2 = SARdata(folders2, h, w, seq_len=13, use_custom_bboxes=False, 
        cache=False, transform=None, csw=1, isw=13, evaluate=True)
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers)
    data_loader2 = DataLoader(data2, batch_size=1, shuffle=False,
        num_workers=num_workers)

    # ImageNet statistic
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    writer = SummaryWriter()

    evaluate(model, data_loader2, writer)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train(model, data_loader, optimizer, writer)
        metric = evaluate(model, data_loader2, writer)

        if metric > best_metric:
            torch.save({"model": model.state_dict()}, r"models/" + best_model_tag)
            best_metric = metric

        if (epoch+1) % save_interval == 0:
            torch.save({
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'best_metric': best_metric,
            }, r"models/" + f"{epoch_model_tag}_{epoch}")

    torch.save({
        "model": model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': num_epochs-1,
        'best_metric': best_metric,
    }, r"models/" + "last_model")

