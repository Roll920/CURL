"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import torch

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v: output channel
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        self.relu = nn.ReLU6(inplace=True)
        self.expand_ratio = expand_ratio
        self.index_mask = torch.ones([1, hidden_dim, 1, 1])
        if expand_ratio == 1:
            # dw
            self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            # pw-linear
            self.conv2 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn2 = nn.BatchNorm2d(oup)
        else:
            # pw
            self.conv1 = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            # dw
            self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
            self.bn2 = nn.BatchNorm2d(hidden_dim)
            # pw-linear
            self.conv3 = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
            self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):
        res = self.relu(self.bn1(self.conv1(x)))

        if self.expand_ratio == 1:
            res = self.bn2(self.conv2(res))
        else:
            mask = self.index_mask.expand(res.size()).cuda()
            res = res.mul(mask)

            res = self.relu(self.bn2(self.conv2(res)))
            mask = self.index_mask.expand(res.size()).cuda()
            res = res.mul(mask)

            res = self.bn3(self.conv3(res))

        if self.identity:
            return x + res
        else:
            return res


class ResidualStage(nn.Module):
    def __init__(self, inp, oup, s, n, t):
        super(ResidualStage, self).__init__()
        block = InvertedResidual
        self.n = n
        self.index_mask = torch.ones([1, oup, 1, 1])
        self.block_1 = block(inp, oup, s, t)
        if n >= 2:
            self.block_2 = block(oup, oup, 1, t)
        if n >= 3:
            self.block_3 = block(oup, oup, 1, t)
        if n >= 4:
            self.block_4 = block(oup, oup, 1, t)

    def forward(self, x):
        x = self.block_1(x)
        mask = self.index_mask.expand(x.size()).cuda()
        x = x.mul(mask)
        if self.n >= 2:
            x = self.block_2(x)
            x = x.mul(mask)
        if self.n >= 3:
            x = self.block_3(x)
            x = x.mul(mask)
        if self.n >= 4:
            x = self.block_4(x)
            x = x.mul(mask)
        return x


class MobileNetV2(nn.Module):
    def __init__(self, model_weight=None, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            # expansion factor, channel number, repeated times, stride
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            layers.append(ResidualStage(input_channel, output_channel, s, n, t))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights(model_weight)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, model_weight):
        my_weight = self.state_dict()
        my_keys = list(my_weight)
        for i, (k, v) in enumerate(model_weight.items()):
            my_weight[my_keys[i]] = v
        self.load_state_dict(my_weight)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    model_weight = torch.load('/opt/luojh/pretrained_models/cub200/mobilenet_v2.pth')
    model = MobileNetV2(model_weight, num_classes=200).cuda()
    input = torch.rand([16, 3, 224, 224]).cuda()
    model(input)
