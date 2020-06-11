import torch
import torch.nn as nn
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, output_channel, channel_index, inplanes, stride=1, downsample=None, groups=1,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width1 = channel_index.pop(0)
        width2 = channel_index.pop(0)
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width1)
        self.bn1 = norm_layer(width1)
        self.conv2 = conv3x3(width1, width2, stride, groups, dilation)
        self.bn2 = norm_layer(width2)
        self.conv3 = conv1x1(width2, output_channel)
        self.bn3 = norm_layer(output_channel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

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

        # channel number
        index_list = list(np.load('../1_evaluate_filter_importance/index.npy'))
        channel_number = []
        for i in range(len(index_list)):
            tmp = index_list[i]
            channel_number.append(int(tmp.sum()))
        print(channel_number)
        self.channel_index = channel_number

        self.input_channel = self.inplanes
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
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.input_channel, num_classes)

        self._initialize_weights(model_weight, index_list)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        output_channel = self.channel_index.pop(0)
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.input_channel, output_channel, stride),
                norm_layer(output_channel),
            )

        layers = []
        layers.append(block(output_channel, self.channel_index, self.input_channel, stride, downsample, self.groups,
                            previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        self.input_channel = output_channel
        for _ in range(1, blocks):
            layers.append(block(output_channel, self.channel_index, self.input_channel, groups=self.groups,
                                dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

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

    def _initialize_weights(self, model_weight, index_list):
        my_weight = self.state_dict()
        my_keys = list(my_weight)
        model_keys = list(model_weight)
        # 1. copy the weight of first few filters
        for i in range(6):  # the first layer
            my_weight[my_keys[i]] = model_weight[model_keys[i]]

        # 2. prune stage output
        stage_index_location = [0, 7, 16, 29]
        layers = [3, 4, 6, 3]
        downsample_index = [0, 3, 7, 13]
        stage_index = []
        block_index = []
        count = 0
        for i in range(len(index_list)):
            if i in stage_index_location:
                tmp = index_list[i]
                ind = np.argsort(-tmp)
                ind = ind[0:int(sum(tmp))]
                ind.sort()
                for j in range(layers[count]):
                    stage_index.append(ind)
                count += 1
            else:
                tmp = index_list[i]
                ind = np.argsort(-tmp)
                ind = ind[0:int(sum(tmp))]
                ind.sort()
                block_index.append(ind)

        # 3.1 process rest stages, 16 blocks*3 layers = 48 layers, each layer: conv+BN 6
        k_ind = 5
        for i in range(16):
            if i != 0:
                old_ind = ind.copy()
            ind = stage_index.pop(0)
            hidden_ind_1 = block_index.pop(0)
            hidden_ind_2 = block_index.pop(0)

            # layer 1
            # conv
            k_ind += 1
            v = model_weight[model_keys[k_ind]]
            v = v[hidden_ind_1, :, :, :]
            if i == 0:
                my_weight[my_keys[k_ind]] = v
            else:
                my_weight[my_keys[k_ind]] = v[:, old_ind, :, :]

            # BN 4
            for j in range(4):
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[hidden_ind_1]

            # num_batches_tracked
            k_ind += 1
            my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

            # layer 2
            # dw conv
            k_ind += 1
            v = model_weight[model_keys[k_ind]]  # 64, 64, 3, 3
            v = v[hidden_ind_2, :, :, :]
            my_weight[my_keys[k_ind]] = v[:, hidden_ind_1, :, :]  # 42, 43, 3, 3

            # BN 4
            for j in range(4):
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[hidden_ind_2]

            # num_batches_tracked
            k_ind += 1
            my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

            # layer 3
            # conv
            k_ind += 1
            v = model_weight[model_keys[k_ind]]  # 256, 64, 1, 1
            v = v[ind, :, :, :]
            my_weight[my_keys[k_ind]] = v[:, hidden_ind_2, :, :]  # 217, 42, 1, 1

            # BN 4
            for j in range(4):
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[ind]

            # num_batches_tracked
            k_ind += 1
            my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

            # downsample
            if i in downsample_index:
                # conv
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                v = v[ind, :, :, :]
                if i == 0:
                    my_weight[my_keys[k_ind]] = v
                else:
                    my_weight[my_keys[k_ind]] = v[:, old_ind, :, :]

                # BN 4
                for j in range(4):
                    k_ind += 1
                    v = model_weight[model_keys[k_ind]]
                    my_weight[my_keys[k_ind]] = v[ind]

                # num_batches_tracked
                k_ind += 1
                my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

        # 1. copy the rest filters
        for i in range(2):
            k_ind += 1
            if i == 0:
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[:, ind]
            else:
                my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]
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
    model_weight = torch.load('../2_0_gradually_pruning/checkpoint/fine_tune/model.pth')
    model = resnet50(model_weight, num_classes=1000).cuda()
    input = torch.rand([16, 3, 224, 224]).cuda()
    model(input)
