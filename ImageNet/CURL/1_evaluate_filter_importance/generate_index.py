import os
import numpy as np
from torchsummaryX import summary
from pruned_model import resnet50
import torch


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
top_k = 8723  # discard the top_k small filters, original ResNet50, 4.08921G FLOPs, goal: 1.11G
for i in range(top_k):
    item = score_list[i]
    # compression rate should not be smaller than threshold
    if index_list[item[0]].sum()/len(index_list[item[0]]) < 0.3:
        continue
    if item[0] == 29:
        continue
    index_list[item[0]][item[1]] = 0
np.save('index.npy', index_list)

# generate channel number
channel_number = []
for i in range(len(filenames)):
    tmp = index_list[i]
    channel_number.append(int(tmp.sum()))
model = resnet50(channel_index=channel_number.copy(), num_classes=1000)
summary(model, torch.zeros((1, 3, 224, 224)))
print(channel_number)
