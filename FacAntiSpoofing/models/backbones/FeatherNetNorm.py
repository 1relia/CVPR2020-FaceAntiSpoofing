
# the code base on https://github.com/tonylins/pytorch-mobilenet-v2
import torch.nn as nn
import math
import torch
from ..utils import build_norm_layer
from ..norms import *


class Norm_Activation(nn.Module):
    def __init__(self, num_features, num_pergroup = 16, norm_type='frn'):
        super(Norm_Activation,self).__init__()
        self.type = norm_type
        if self.type == 'frn':
            self.norm = FRN(num_features)
        elif self.type == 'sw2':
            self.norm = SwitchWhiten2d(num_features, num_pergroup, sw_type=2)
        elif self.type == 'sw3':
            self.norm = SwitchWhiten2d(num_features, num_pergroup, sw_type=3)
        elif self.type == 'sw5':
            self.norm = SwitchWhiten2d(num_features, num_pergroup, sw_type=5)
        elif self.type =='bn':
            self.norm = nn.BatchNorm2d(num_features)
        elif self.type =='in':
            self.norm = nn.InstanceNorm2d(num_features)
        self.act = nn.ReLU6(inplace=True)
    def forward(self, x):
        x = self.norm(x)
        if self.type == 'frn':
            return x
        x = self.act(x)
        return x

def conv_bn(inp, oup, stride,norm_type='frn'):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        Norm_Activation(oup,norm_type)
    )
# Norm_Activation(oup,norm_type=norm_type)

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

# reference form : https://github.com/moskomule/senet.pytorch  
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
     
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, downsample=None,norm_type='frn'):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.downsample = downsample

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                Norm_Activation(hidden_dim, norm_type=norm_type),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                Norm_Activation(hidden_dim, norm_type=norm_type),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
#                 Norm_Activation(hidden_dim, norm_type=norm_type),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            if self.downsample is not None:
                return self.downsample(x) + self.conv(x)
            else:
                return self.conv(x)


class FeatherNet(nn.Module):
    def __init__(self, n_class=2, input_size=224, se = False, avgdown=False, width_mult=1.,depth_mult=1.,norm_type='frn'):
        super(FeatherNet, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1024
        self.se = se
        self.avgdown = avgdown
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 32, 2, 2], # 56x56
            [6, 48, 6, 2], # 14x14
            [6, 64, 3, 2], # 7x7
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        
        self.features = [conv_bn(3, input_channel, 2, norm_type=norm_type)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            n = int(n * depth_mult)
            for i in range(n):
                downsample = None
                if i == 0:
                    if self.avgdown:
                        downsample = nn.Sequential(nn.AvgPool2d(2, stride=2),
                        nn.BatchNorm2d(input_channel),
                        nn.Conv2d(input_channel, output_channel , kernel_size=1, bias=False)
                        )
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t, downsample = downsample, norm_type=norm_type))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t, downsample = downsample, norm_type=norm_type))
                input_channel = output_channel
            if self.se:
                self.features.append(SELayer(input_channel))

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
#         building last several layers        
        self.final_DW = nn.Sequential(nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=2, padding=1,
                                  groups=input_channel, bias=False),
                                     )


        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.final_DW(x)
        
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def FeatherNetBNorm(norm_type):
    model = FeatherNet(se = True,avgdown=True,norm_type=norm_type)
    return model

def FeatherNetBNorm_w15(norm_type):
    model = FeatherNet(se = True,avgdown=True,width_mult=1.5,depth_mult=1.,norm_type=norm_type)
    return model

def FeatherNetBNorm_w15_d15(norm_type):
    model = FeatherNet(se = True,avgdown=True,width_mult=1.5,depth_mult=1.5,norm_type=norm_type)
    return model

def FeatherNetBNorm_w20_d15(norm_type):
    model = FeatherNet(se = True,avgdown=True,width_mult=2.0,depth_mult=1.5,norm_type=norm_type)
    return model

def FeatherNetBNorm_w20_d20(norm_type):
    model = FeatherNet(se = True,avgdown=True,width_mult=2.,depth_mult=2.,norm_type=norm_type)
    return model

