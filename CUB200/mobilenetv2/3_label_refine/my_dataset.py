import torch.utils.data as data
import numpy as np
import os
from torchvision import transforms
from PIL import Image


class MyDataSet(data.Dataset):
    def __init__(self, transform=None):
        self.file_list = np.load(os.path.join('data_list.npy'))
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_list[index][0]
        label = int(self.file_list[index][1])
        sample = Image.open(file_path).convert('RGB')

        if self.transform:
            sample = self.transform(sample)
        return sample, label, index

    def __len__(self):
        # return int(len(self.file_list)/10.0)
        return len(self.file_list)


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
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
    loader = MyDataSet(data_transforms['train'])
    print(loader.__len__())
