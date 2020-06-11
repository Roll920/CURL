import argparse
import os
import sys
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import transforms
from torchsummaryX import summary
import numpy as np
from my_dataset import MyDataSet
from ori_model import resnet50
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
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', default=False, type=bool,
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
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


def extract_logits():
    global args, best_prec1
    args = parser.parse_args()
    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    model_weight = torch.load(FLAGS['pretrain_model'])  # 78.771%
    model = resnet50(model_weight, num_classes=FLAGS['class_num']).cuda()

    model.eval()
    summary(model, torch.zeros((1, 3, 224, 224)).cuda())
    # test_accuracy(model)
    cudnn.benchmark = True

    # Data loading code from lmdb
    data_list = np.load(os.path.join(FLAGS['6x_larger_dataset'], 'data_list.npy'))
    np.save('data_list.npy', data_list)
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
    dsets = {}
    dsets['val'] = MyDataSet(data_transforms['val'])
    val_loader = torch.utils.data.DataLoader(dsets['val'], batch_size=4 * FLAGS['batch_size'], shuffle=False, num_workers=8,
                                             pin_memory=True)
    print('data_loader_success!')
    validate(val_loader, model)


def validate(val_loader, model):
    # switch to evaluate mode
    model.eval()
    label_list = []
    for i, (input, label, index) in enumerate(val_loader):
        with torch.no_grad():
            input = input.cuda()

            # compute output
            output = model(input)
            for ind_, item in enumerate(index):
                label_list.append(output[ind_].cpu().numpy())

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'.format(i, len(val_loader)))
            sys.stdout.flush()
    np.save(os.path.join('label_list.npy'), label_list)


if __name__ == '__main__':
    # data_list = np.load('/mnt/ramdisk/CUB/cub6w/data_list.npy')
    # np.save('data_list.npy', data_list)
    extract_logits()
