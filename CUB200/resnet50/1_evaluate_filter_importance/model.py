import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dim = width

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return out


class ResidualStage(nn.Module):
    def __init__(self, inplanes, block, planes, blocks, stride=1, dilate=False):
        super(ResidualStage, self).__init__()
        self.dim = planes * block.expansion
        self.n = blocks  # the number of blocks in each stage: [3, 4, 6, 3]

        norm_layer = nn.BatchNorm2d
        self.downsample = None
        self.inplanes = inplanes
        self.groups = 1
        self.base_width = 64
        previous_dilation = 1
        self.dilation = 1
        self.relu = nn.ReLU(inplace=True)
        if dilate:
            self.dilation *= stride
            stride = 1

        self.block_1 = block(self.inplanes, planes, stride, self.groups,
                             self.base_width, previous_dilation, norm_layer)
        if stride != 1 or self.inplanes != planes * block.expansion:
            self.downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        self.inplanes = planes * block.expansion
        self.block_2 = block(self.inplanes, planes, groups=self.groups,
                             base_width=self.base_width, dilation=self.dilation,
                             norm_layer=norm_layer)
        self.block_3 = block(self.inplanes, planes, groups=self.groups,
                             base_width=self.base_width, dilation=self.dilation,
                             norm_layer=norm_layer)
        if self.n > 3:
            # n = 4
            self.block_4 = block(self.inplanes, planes, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer)
        if self.n > 4:
            # n = 6
            self.block_5 = block(self.inplanes, planes, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer)
            self.block_6 = block(self.inplanes, planes, groups=self.groups,
                                 base_width=self.base_width, dilation=self.dilation,
                                 norm_layer=norm_layer)

    def forward(self, x):
        x = self.res_block_forward(x, self.block_1, check_downsample=True)
        x = self.res_block_forward(x, self.block_2)
        x = self.res_block_forward(x, self.block_3)
        if self.n > 3:
            x = self.res_block_forward(x, self.block_4)
        if self.n > 4:
            x = self.res_block_forward(x, self.block_5)
            x = self.res_block_forward(x, self.block_6)
        return x

    def res_block_forward(self, x, block, check_downsample=False):
        identity = x
        if check_downsample and self.downsample is not None:
            identity = self.downsample(x)
        out = block(x)
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, model_weight, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = ResidualStage(self.inplanes, block, 64, layers[0])
        self.inplanes = 64 * block.expansion
        self.layer2 = ResidualStage(self.inplanes, block, 128, layers[1], stride=2,
                                    dilate=replace_stride_with_dilation[0])
        self.inplanes = 128 * block.expansion
        self.layer3 = ResidualStage(self.inplanes, block, 256, layers[2], stride=2,
                                    dilate=replace_stride_with_dilation[1])
        self.inplanes = 256 * block.expansion
        self.layer4 = ResidualStage(self.inplanes, block, 512, layers[3], stride=2,
                                    dilate=replace_stride_with_dilation[2])
        self.inplanes = 512 * block.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_old_weights(model_weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_old_weights(self, model_weight):
        my_weight = self.state_dict()
        my_keys = list(my_weight)
        model_keys = list(model_weight)

        for i in range(len(my_keys)):
            my_weight[my_keys[i]] = model_weight[model_keys[i]]
        self.load_state_dict(my_weight)


def resnet50(model_weight, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    layers = [3, 4, 6, 3]
    model = ResNet(model_weight, Bottleneck, layers, **kwargs)
    return model


if __name__ == '__main__':
    model_weight = torch.load('/home/luojh2/.torch/models/resnet50-19c8e357.pth')
    model = resnet50(model_weight, num_classes=1000).cuda()
    input = torch.rand([16, 3, 224, 224]).cuda()
    model(input)
