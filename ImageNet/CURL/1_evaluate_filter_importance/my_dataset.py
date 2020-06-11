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
        sample = Image.open(file_path)

        while(sample.layers != 3):
            index = np.random.randint(256, 120000)
            file_path = self.file_list[index][0]
            sample = Image.open(file_path)

        if self.transform:
            sample = self.transform(sample)
        return sample, int(self.file_list[index][1])

    def __len__(self):
        return 256
        # return 1


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
    loader.__getitem__(24)
