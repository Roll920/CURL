import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from mobilenetv2 import MobileNetV2
import numpy as np
from my_dataset import MyDataSet
from test_acc import test_accuracy
import sys
import yaml


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
'''parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
'''
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=True, type=bool,
                    help='evaluate model on validation set')
parser.add_argument('--yaml_file', default='../config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
f = open(args.yaml_file)
FLAGS = yaml.load(f)
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS['gpu_id']
best_prec1 = -1
print(args)


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model_weight = torch.load(FLAGS['pretrain_model'])  # 78.77%
    model = MobileNetV2(model_weight, num_classes=FLAGS['class_num'])
    model = model.cuda()
    cudnn.benchmark = True
    test_accuracy(model, FLAGS)

    # Data loading code from lmdb
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            # transforms.Resize(256),
            # transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("| Preparing model...")
    dset_train = MyDataSet(data_transforms['val'], FLAGS['proxy_dataset_size'])
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=FLAGS['proxy_dataset_size'], shuffle=False, num_workers=8, pin_memory=True)
    print('data_loader_success!')

    # evaluate and train
    validate(train_loader, model)


def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()
    logsoftmax = torch.nn.LogSoftmax(dim=1).cuda()
    softmax = torch.nn.Softmax(dim=1).cuda()
    for _, (input, label) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()
            logits = model(input)

            # 1. stage block
            beta = []
            gamma = []
            for i in range(7):
                # 7 stage
                resblock = model.features._modules[str(i + 1)]
                score = np.zeros(resblock.dim)
                for j in range(resblock.dim):  # for each dim
                    for k in range(resblock.n):  # for each block
                        if k == 0:
                            block = resblock.block_1
                        elif k == 1:
                            block = resblock.block_2
                        elif k == 2:
                            block = resblock.block_3
                        elif k == 3:
                            block = resblock.block_4
                        if i == 0:
                            beta.append(block.bn2.weight[j].data.clone())
                            gamma.append(block.bn2.bias[j].data.clone())
                            block.bn2.weight.data[j] = 0
                            block.bn2.bias.data[j] = 0
                        else:
                            beta.append(block.bn3.weight[j].data.clone())
                            gamma.append(block.bn3.bias[j].data.clone())
                            block.bn3.weight.data[j] = 0
                            block.bn3.bias.data[j] = 0
                    # score
                    output = model(input)
                    kl_loss = torch.mean(
                        torch.sum(softmax(output) * (logsoftmax(output) - logsoftmax(logits)), dim=1))
                    score[j] = kl_loss
                    print('i={0}/7, j={1}/{2}, loss={3:.7f}'.format(i, j, resblock.dim, score[j]))
                    sys.stdout.flush()

                    # recover
                    for k in range(resblock.n):  # for each block
                        if k == 0:
                            block = resblock.block_1
                        elif k == 1:
                            block = resblock.block_2
                        elif k == 2:
                            block = resblock.block_3
                        elif k == 3:
                            block = resblock.block_4
                        if i == 0:
                            block.bn2.weight.data[j] = beta.pop(0)
                            block.bn2.bias.data[j] = gamma.pop(0)
                        else:
                            block.bn3.weight.data[j] = beta.pop(0)
                            block.bn3.bias.data[j] = gamma.pop(0)
                np.save('results/stage_{0}.npy'.format(i), score)
            test_accuracy(model, FLAGS)

            # 2. block index
            for i in range(1, 7):  # for each stage, skip the first stage (expansion factor=1)
                resblock = model.features._modules[str(i + 1)]
                for j in range(resblock.n):  # for each block
                    if j == 0:
                        block = resblock.block_1
                    elif j == 1:
                        block = resblock.block_2
                    elif j == 2:
                        block = resblock.block_3
                    elif j == 3:
                        block = resblock.block_4
                    # 2.1 mask
                    score = np.zeros(block.dim)
                    for k in range(block.dim):  # for each filter
                        beta.append(block.bn1.weight[k].data.clone())
                        gamma.append(block.bn1.bias[k].data.clone())
                        block.bn1.weight.data[k] = 0
                        block.bn1.bias.data[k] = 0

                        beta.append(block.bn2.weight[k].data.clone())
                        gamma.append(block.bn2.bias[k].data.clone())
                        block.bn2.weight.data[k] = 0
                        block.bn2.bias.data[k] = 0

                        # score
                        output = model(input)
                        kl_loss = torch.mean(
                            torch.sum(softmax(output) * (logsoftmax(output) - logsoftmax(logits)), dim=1))
                        score[k] = kl_loss
                        print('i={0}/7, j={1}/{2}, k={3}/{4}, loss={5:.7f}'.format(i, j, resblock.n, k,
                                                                                   block.dim, score[k]))
                        sys.stdout.flush()

                        # recover
                        block.bn1.weight.data[k] = beta.pop(0)
                        block.bn1.bias.data[k] = gamma.pop(0)
                        block.bn2.weight.data[k] = beta.pop(0)
                        block.bn2.bias.data[k] = gamma.pop(0)
                    np.save('results/stage_{0}_block_{1}.npy'.format(i, j), score)
    test_accuracy(model, FLAGS)


if __name__ == '__main__':
    main()
