import os
import numpy as np
from torchsummaryX import summary
from pruned_mobilenetv2 import MobileNetV2
import torch
import yaml
import argparse


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
'''parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
'''
parser.add_argument('--yaml_file', default='../config.yaml', type=str, help='yaml config file')
args = parser.parse_args()

f = open(args.yaml_file)
FLAGS = yaml.load(f)


# sort score
filenames = os.listdir('results/')
filenames.sort()
score_list = []
index_list = []
for i, item in enumerate(filenames):
    data = np.load('results/'+item)
    index_list.append(np.ones(len(data)))
    for j in range(len(data)):
        score_list.append((i, j, data[j]))  # index_list_i, channel_id_j, score
score_list = sorted(score_list, key=lambda item_: item_[2])  # small to large, remove small filters


# generate index
top_k = 6890  # discard the top_k small filters, 96.1171M FLOPs
for i in range(top_k):
    item = score_list[i]
    # compression rate should not be smaller than threshold
    if index_list[item[0]].sum()/len(index_list[item[0]]) < 0.3:
        continue
    index_list[item[0]][item[1]] = 0
np.save('index.npy', index_list)

# generate channel number
channel_number = []
for i in range(len(filenames)):
    tmp = index_list[i]
    channel_number.append(int(tmp.sum()))
print(channel_number)
model = MobileNetV2(num_classes=FLAGS['class_num'], channel_number=channel_number)
summary(model, torch.zeros((1, 3, 224, 224)))
