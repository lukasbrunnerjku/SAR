
# 2020.06.09-Changed for building GhostNet
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""
Creates a GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.
https://arxiv.org/abs/1911.11907
Modified from https://github.com/d-li14/mobilenetv3.pytorch and https://github.com/rwightman/pytorch-image-models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x    

    
class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, act_layer=nn.ReLU):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chs)
        self.act1 = act_layer(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                             padding=(dw_kernel_size-1)//2,
                             groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
        
        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                       padding=(dw_kernel_size-1)//2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )


    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)
        
        x += self.shortcut(residual)
        return x


class GhostNet(nn.Module):
    def __init__(self, cfgs, width=1.0):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        output_channel = _make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.act1 = nn.ReLU(inplace=True)
        input_channel = output_channel

        # building inverted residual blocks
        stages = []
        block = GhostBottleneck
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                output_channel = _make_divisible(c * width, 4)
                hidden_channel = _make_divisible(exp_size * width, 4)
                layers.append(block(input_channel, hidden_channel, output_channel, k, s,
                              se_ratio=se_ratio))
                input_channel = output_channel
            stages.append(nn.Sequential(*layers))

        output_channel = _make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))     
        self.stages = nn.ModuleList(stages) 

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        ret = []
        for stage in self.stages:
            x = stage(x)
            ret.append(x)
        return ret

from typing import List
from torch.nn.functional import interpolate
from typing import Dict

class CenternetHeads(nn.Module):

    def __init__(self, heads: Dict[str, int], in_channels: int, 
        head_hidden_channels=256):
        super().__init__()
        
        self.in_channels = in_channels  # num. of input channels to heads 
        self.heads = heads
        self.head_hidden_channels = head_hidden_channels

        for head_name, head_out_channels in heads.items():
            self.create_head(head_name, head_out_channels)

    def forward(self, x):
        return {head: self.__getattr__(head)(x) for head in self.heads}
    
    @staticmethod
    def fill_head_weights(layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def create_head(self, name: str, out_channels: int):
        head = nn.Sequential(
            nn.Conv2d(self.in_channels, self.head_hidden_channels,
                kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_hidden_channels, out_channels,
                kernel_size=1, stride=1, padding=0, bias=True))
        if 'hm' in name:
            head[-1].bias.data.fill_(-2.19)
        else:
            self.fill_head_weights(head)
        self.__setattr__(name, head)

def rescale(x, scale):
    N, C, H, W = x.size()

    # N, W, H, C
    x = x.permute(0, 3, 2, 1)  

    # N, W, H*scale, C/scale
    x.contiguous().view((N, W, H * scale, int(C / scale)))

    # N, H*scale, W, C/scale
    x = x.permute(0, 2, 1, 3)

    # N, H*scale, W*scale, C/(scale**2)
    x = x.contiguous().view((N, W * scale, H * scale, int(C / scale**2)))

    # N, C/(scale**2), H*scale, W*scale
    x = x.permute(0, 3, 1, 2)
    
    return x

class Decoder(nn.Module):

    def __init__(self, backbone: nn.Module,
        stage_nums: List[int], channels: List[int], 
        down_ratio: int = 4, hidden_channels: int = 256,
        out_channels=64):
        super().__init__()
        assert len(stage_nums) == len(channels)
        assert down_ratio in [1, 2, 4, 8]

        self.backbone = backbone
        self.stage_nums = stage_nums 
        self.channels = channels 
        self.down_ratio = down_ratio
        self.hidden_channels = hidden_channels

        self.scale = self.calculate_scale()

        self.low_level_layer = nn.Sequential(
            nn.Conv2d(sum(channels[:-1]), 
                hidden_channels, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.high_level_layer = nn.Sequential(
            nn.Conv2d(channels[-1] + hidden_channels, 
                hidden_channels, 
                kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )

        self.pre_resize_layer = nn.Sequential(
            nn.Conv2d(hidden_channels, 
                out_channels * self.scale**2, 
                kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels * self.scale**2, affine=False),
            nn.ReLU(inplace=True)
        )
        
    def calculate_scale(self):
        # scale for upsampling lowest stage resolution
        # to obtain feature maps expected by 'down_ratio'
        with torch.no_grad():
            x = torch.randn(1, 3, 512, 512)
            out = self.backbone(x)
        return int(512 / self.down_ratio / out[-1].size(-1))

    def forward(self, x):
        x = self.backbone(x)
        high_level_features = x[self.stage_nums[-1]]
        low_stages = [x[stage_num] for stage_num in self.stage_nums[:-1]]

        size = high_level_features.shape[-2:]  # h x w
        low_stages = [interpolate(stage, size, mode='bilinear', 
            align_corners=True) for stage in low_stages]

        low_level_features = torch.cat(low_stages, dim=1)
        low_level_features = self.low_level_layer(low_level_features)

        x = torch.cat((high_level_features, low_level_features), dim=1)
        x = self.high_level_layer(x)

        x = self.pre_resize_layer(x)
        x = rescale(x, self.scale)
        return x  # bsz x out_channels x h/down_ratio x w/down_ratio

def ghostnet(**kwargs):
    """
    Constructs a GhostNet model
    """
    cfgs = [  # input e.g. 512 x 512
        # k, t, c, SE, s 
        # stage1
        [[3,  16,  16, 0, 1]],  # 256 x 256
        # stage2
        [[3,  48,  24, 0, 2]],
        [[3,  72,  24, 0, 1]],  # 128 x 128
        # stage3
        [[5,  72,  40, 0.25, 2]],
        [[5, 120,  40, 0.25, 1]],  # 64 x 64
        # stage4
        [[3, 240,  80, 0, 2]],
        [[3, 200,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 184,  80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]
        ],  # 32 x 32
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [[5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1],
         [5, 960, 160, 0, 1],
         [5, 960, 160, 0.25, 1]
        ]  # 16 x 16
    ]
    return GhostNet(cfgs, **kwargs)

def print_stage_outputs(backbone, h=512, w=512):
    x = torch.randn(1, 3, h, w)
    out = backbone(x)
    assert isinstance(out, list), 'backbone must provide stages'
    print('\n'.join([str(o.shape) for o in out]))

import thop 
from copy import deepcopy
def model_info(model, verbose=False, img_size=512):
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:
        # assume model expects RGB image if not otherwise specified
        input_channels = getattr(model, 'input_channels', 3)
        img = torch.randn(1, input_channels, img_size, img_size, 
            device=next(model.parameters()).device)
        # macs ... multiply-add computations
        # flops ... floating point operations
        macs, _ = thop.profile(deepcopy(model), inputs=(img,), verbose=False)
        flops = macs / 1E9 * 2  # each mac = 2 flops (addition + multiplication)
        fs = ', %.1f GFLOPs' % (flops)
    except ImportError:
        fs = ''

    print(f"Model Summary: {len(list(model.modules()))} layers, {n_p / 10**6:0.3}M parameters, {n_g / 10**6:0.3}M gradients{fs}")

if __name__ == '__main__':
    backbone = ghostnet()
    model_info(backbone)
    #print_stage_outputs(backbone); exit()

    stage_nums, channels = [2, 4, 6, 9], [24, 40, 112, 960]
    decoder = Decoder(backbone, stage_nums, channels)

    model_info(decoder)
    image = torch.randn(1, 3, 512, 512)
    out = decoder(image)
    print(out.shape)

    #CenternetHeads(heads, in_channels=1024)
    
    