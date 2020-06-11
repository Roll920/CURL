import argparse
import os
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from model import resnet50
import numpy as np
from my_dataset import MyDataSet
from test_acc import test_accuracy
import sys
from torchsummaryX import summary


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
parser.add_argument('--gpu_id', default='7', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--class_num', default=1000, type=int,
                    help='number of class')
parser.add_argument('--data_base', default='/mnt/ramdisk/ImageNet', type=str, help='the path of dataset')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
best_prec1 = -1
print(args)


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model_weight = torch.load('/home/luojh2/.torch/models/resnet50-19c8e357.pth')  # 76.130
    model = resnet50(model_weight, num_classes=1000)
    model = model.cuda()
    cudnn.benchmark = True

    # torch.save(model.state_dict(), '/opt/luojh/pretrained_models/ResNewt50_ImageNet.pth')
    model.eval()
    summary(model, torch.zeros((1, 3, 224, 224)).cuda())

    # test_accuracy(model)

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
    dset_train = MyDataSet(data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
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
            for i in range(4):
                # 4 stage
                if i == 0:
                    resblock = model.layer1  # [1, 256, 1, 1]
                elif i == 1:
                    resblock = model.layer2
                elif i == 2:
                    resblock = model.layer3
                else:
                    resblock = model.layer4

                index_mask = resblock.index_mask.clone()
                score = np.zeros(index_mask.size(1))
                for j in range(index_mask.size(1)):
                    tmp = index_mask.clone()
                    tmp[:, j, :, :] = 0
                    resblock.index_mask = tmp
                    output = model(input)

                    # ThiNet:77.598%, CE_loss_drop:65.602%, cross entropy:77.235%, KL1:77.373%, KL2:77.287%
                    # score[j] = torch.mean(-torch.sum(softmax(logits) * logsoftmax(output), dim=1)) # CE loss
                    kl_loss = torch.mean(
                            torch.sum(softmax(output) * (logsoftmax(output) - logsoftmax(logits)), dim=1))
                    score[j] = kl_loss
                    print('i={0}/4, j={1}/{2}, loss={3:.7f}'.format(i, j, index_mask.size(1), score[j]))
                    sys.stdout.flush()
                resblock.index_mask = index_mask
                np.save('results/stage_{0}.npy'.format(i), score)
            test_accuracy(model)

            # 2. block index
            for i in range(4):  # for each stage
                if i == 0:
                    resblock = model.layer1  # [1, 256, 1, 1]
                elif i == 1:
                    resblock = model.layer2
                elif i == 2:
                    resblock = model.layer3
                else:
                    resblock = model.layer4
                for j in range(resblock.n):  # for each block
                    if j == 0:
                        block = resblock.block_1
                    elif j == 1:
                        block = resblock.block_2
                    elif j == 2:
                        block = resblock.block_3
                    elif j == 3:
                        block = resblock.block_4
                    elif j == 4:
                        block = resblock.block_5
                    elif j == 5:
                        block = resblock.block_6
                    # 2.1 mask_1
                    index_mask = block.index_mask_1
                    score = np.zeros(index_mask.size(1))
                    for k in range(index_mask.size(1)):  # for each filter
                        tmp = index_mask.clone()
                        tmp[:, k, :, :] = 0
                        block.index_mask_1 = tmp
                        output = model(input)
                        # score[k] = torch.mean(-torch.sum(softmax(logits) * logsoftmax(output), dim=1))
                        kl_loss = torch.mean(
                            torch.sum(softmax(output) * (logsoftmax(output) - logsoftmax(logits)), dim=1))
                        score[k] = kl_loss
                        print('i={0}/4, j={1}/{2}, k={3}/{4}, loss={5:.7f}, mask_1'.format(i, j, resblock.n, k,
                                                                                           index_mask.size(1),
                                                                                           score[k]))
                        sys.stdout.flush()
                    np.save('results/stage_{0}_block_{1}_mask1.npy'.format(i, j), score)
                    block.index_mask_1 = index_mask

                    # 2.2 mask_2
                    index_mask = block.index_mask_2
                    score = np.zeros(index_mask.size(1))
                    for k in range(index_mask.size(1)):  # for each filter
                        tmp = index_mask.clone()
                        tmp[:, k, :, :] = 0
                        block.index_mask_2 = tmp
                        output = model(input)
                        # score[k] = torch.mean(-torch.sum(softmax(logits) * logsoftmax(output), dim=1))
                        kl_loss = torch.mean(
                            torch.sum(softmax(output) * (logsoftmax(output) - logsoftmax(logits)), dim=1))
                        score[k] = kl_loss
                        print('i={0}/4, j={1}/{2}, k={3}/{4}, loss={5:.7f}, mask_2'.format(i, j, resblock.n, k,
                                                                                           index_mask.size(1),
                                                                                           score[k]))
                        sys.stdout.flush()
                    np.save('results/stage_{0}_block_{1}_mask2.npy'.format(i, j), score)
                    block.index_mask_2 = index_mask
    test_accuracy(model)


if __name__ == '__main__':
    main()
