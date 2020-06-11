"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import torch
import numpy as np

__all__ = ['mobilenetv2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
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
    def __init__(self, first_block, channel_number, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.identity = stride == 1 and inp == oup

        if first_block:
            self.identity = False

        if expand_ratio == 1:
            hidden_dim = inp
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            hidden_dim = channel_number.pop(0)
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, model_weight, num_classes=1000, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # channel number
        index_list = list(np.load('../1_evaluate_filter_importance/index.npy'))
        channel_number = []
        for i in range(len(index_list)):
            tmp = index_list[i]
            channel_number.append(int(tmp.sum()))
        print(channel_number)

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = channel_number.pop(0)
            first_block = True
            for i in range(n):
                layers.append(block(first_block, channel_number, input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
                first_block = False

        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights(model_weight, index_list)

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self, model_weight, index_list):
        my_weight = self.state_dict()
        my_keys = list(my_weight)
        model_keys = list(model_weight)
        # 1. copy the weight of first few filters
        for i in range(2*6):  # the first 2 layers
            my_weight[my_keys[i]] = model_weight[model_keys[i]]

        # 2. prune stage output
        stage_index_location = [0, 1, 4, 8, 13, 17, 21]
        stage_index = []
        block_index = []
        count = 0
        for i in range(len(index_list)):
            if i in stage_index_location:
                tmp = index_list[i]
                ind = np.argsort(-tmp)
                ind = ind[0:int(sum(tmp))]
                ind.sort()
                for j in range(self.cfgs[count][2]):
                    stage_index.append(ind)
                count += 1
            else:
                tmp = index_list[i]
                ind = np.argsort(-tmp)
                ind = ind[0:int(sum(tmp))]
                ind.sort()
                block_index.append(ind)
        # 2.1 process stage 1
        ind = stage_index.pop(0)
        v = model_weight[model_keys[12]]
        my_weight[my_keys[12]] = v[ind, :, :, :]
        v = model_weight[model_keys[13]]
        my_weight[my_keys[13]] = v[ind]
        v = model_weight[model_keys[14]]
        my_weight[my_keys[14]] = v[ind]
        v = model_weight[model_keys[15]]
        my_weight[my_keys[15]] = v[ind]
        v = model_weight[model_keys[16]]
        my_weight[my_keys[16]] = v[ind]
        v = model_weight[model_keys[17]]
        my_weight[my_keys[17]] = v

        # 2.2 process rest stages, 16 blocks*3 layers = 48 layers, each layer: conv+BN 6
        k_ind = 17
        for i in range(16):
            old_ind = ind.copy()
            ind = stage_index.pop(0)
            hidden_ind = block_index.pop(0)

            # layer 1
            # conv
            k_ind += 1
            v = model_weight[model_keys[k_ind]]
            v = v[hidden_ind, :, :, :]
            my_weight[my_keys[k_ind]] = v[:, old_ind, :, :]

            # BN 4
            for j in range(4):
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[hidden_ind]

            # num_batches_tracked
            k_ind += 1
            my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

            # layer 2
            # dw conv
            k_ind += 1
            v = model_weight[model_keys[k_ind]]  # 96x1x3x3
            my_weight[my_keys[k_ind]] = v[hidden_ind, :, :, :]

            # BN 4
            for j in range(4):
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[hidden_ind]

            # num_batches_tracked
            k_ind += 1
            my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

            # layer 3
            # conv
            k_ind += 1
            v = model_weight[model_keys[k_ind]]
            v = v[ind, :, :, :]
            my_weight[my_keys[k_ind]] = v[:, hidden_ind, :, :]

            # BN 4
            for j in range(4):
                k_ind += 1
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[ind]

            # num_batches_tracked
            k_ind += 1
            my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]

        # 1. copy the rest filters
        for i in range(8):
            k_ind += 1
            if i == 0:
                v = model_weight[model_keys[k_ind]]
                my_weight[my_keys[k_ind]] = v[:, ind, :, :]
            else:
                my_weight[my_keys[k_ind]] = model_weight[model_keys[k_ind]]
        self.load_state_dict(my_weight)


if __name__ == '__main__':
    model_weight = torch.load('/opt/luojh/pretrained_models/cub200/mobilenet_v2.pth')
    model = MobileNetV2(model_weight, num_classes=200)
    import flops_benchmark

    flops = flops_benchmark.count_flops(model)
    print('flops: {0}M, MAC: {1}M'.format(flops / 10 ** 6, (flops / 10 ** 6) / 2.0))

