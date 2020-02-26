import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# from ..utils import build_norm_layer
from ..norms import *



__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class Norm_Activation(nn.Module):
    def __init__(self, num_features, num_pergroup = 16, norm_type='frn',act=True):
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
        self.act = act
        if self.act and self.type != 'frn':
            self.activate = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.norm(x)
        if self.type == 'frn':
            return x
        if self.act:
            x = self.activate(x)
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
        nn.ReLU(inplace=True)
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
     

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_type= 'frn'):
        super(BasicBlock, self).__init__()
        
        self.norm_act1 = Norm_Activation(inplanes, norm_type=norm_type)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm_act2 = Norm_Activation(planes, norm_type=norm_type)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

# 预激活，ResNetv2    
    def forward(self, x):
        residual = x

        out = self.norm_act1(x)
        out = self.conv1(out)

        out = self.norm_act2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 norm_type= 'frn'):
        super(Bottleneck, self).__init__()      
        self.norm_act1 = Norm_Activation(inplanes, norm_type=norm_type)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.norm_act2 = Norm_Activation(planes, norm_type=norm_type)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm_act3 = Norm_Activation(planes, norm_type=norm_type)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride
    
# 预激活，ResNetv2  
    def forward(self, x):
        residual = x
        out = self.norm_act1(x)
        out = self.conv1(out)

        out = self.norm_act2(out)
        out = self.conv2(out)

        out = self.norm_act3(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=2,
                 norm_type='frn'):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.norm_type = norm_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.norm_act1 = Norm_Activation(64, norm_type=norm_type)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,
                                       norm_type=norm_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       norm_type=norm_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       norm_type=norm_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       norm_type='frn')
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride, norm_type='frn'):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm_Activation(planes * block.expansion, norm_type=norm_type),
            )

        layers = []
        layers.append(
            block(self.inplanes,
                  planes,
                  stride,
                  downsample,
                  norm_type=norm_type))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      norm_type=norm_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm_act1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(norm_type, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], norm_type=norm_type)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']),
                              strict=False)
    return model


def resnet34(norm_type, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], norm_type=norm_type)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']),
                              strict=False)
    return model


def resnet50(norm_type, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], norm_type=norm_type)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']),
                              strict=False)
    return model


def resnet101(norm_type, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], norm_type=norm_type)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']),
                              strict=False)
    return model


def resnet152(norm_type, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], norm_type=norm_type)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']),
                              strict=False)
    return model

