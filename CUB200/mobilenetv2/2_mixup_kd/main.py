import argparse
import os
import shutil
import time
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms
from torchsummaryX import summary
from mobilenetv2 import MobileNetV2
from ori_mobilenetv2 import OriMobileNetV2
from math import cos, pi
import numpy as np
import yaml


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
'''parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
'''
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool,
                    help='evaluate model on validation set')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--yaml_file', default='../config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
f = open(args.yaml_file)
FLAGS = yaml.load(f)
print(FLAGS)
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS['gpu_id']
best_prec1 = -1
print(args)


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    global args, best_prec1
    args = parser.parse_args()

    # create model
    model_weight = torch.load(FLAGS['pretrain_model'])  # 77.58%
    model = MobileNetV2(model_weight, num_classes=FLAGS['class_num'])
    model.eval()
    summary(model, torch.zeros((1, 3, 224, 224)))
    model = model.cuda()
    cudnn.benchmark = True

    model_teacher = OriMobileNetV2(num_classes=FLAGS['class_num'], width_mult=1.)
    model_teacher.load_state_dict(model_weight)
    model_teacher.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

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

    data_dir = FLAGS['data_base']
    print("| Preparing model...")
    dsets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    train_loader = torch.utils.data.DataLoader(dsets['train'], batch_size=FLAGS['batch_size'], shuffle=True, num_workers=8,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=4 * FLAGS['batch_size'], shuffle=False, num_workers=8,
                                             pin_memory=True)
    print('data_loader_success!')

    # evaluate and train
    validate(val_loader, model, criterion)
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model_teacher, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        best_prec1 = max(prec1, best_prec1)
        folder_path = 'checkpoint/fine_tune'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        torch.save(model.state_dict(), folder_path + '/model.pth')
        print('best acc is %.3f' % best_prec1)


def train(train_loader, model_teacher, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    model.train()
    model_teacher.eval()
    logsoftmax = nn.LogSoftmax(dim=1).cuda()
    softmax = nn.Softmax(dim=1).cuda()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()
        input, targets_a, targets_b, lam = mixup_data(input, target, args.alpha, True)

        # compute output
        output = model(input)
        logits = model_teacher(input)

        T = 2.0
        loss1 = torch.mean(
            T * T * torch.sum(softmax(logits / T) * (logsoftmax(logits / T) - logsoftmax(output / T)), dim=1))
        loss2 = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        loss = 0.7 * loss1 + 0.3 * loss2

        # measure accuracy and record loss
        losses.update(loss.data.item(), input.size(0))
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  | Acc: {3:.3f}% ({4:.3f}/{5})\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f}))'.format(
                epoch, i, len(train_loader), 100. * correct / total, correct, total, batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        with torch.no_grad():
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))
            sys.stdout.flush()

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']
    if iteration == 0:
        print('current learning rate:{0}'.format(lr))

    warmup_epoch = 5
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
