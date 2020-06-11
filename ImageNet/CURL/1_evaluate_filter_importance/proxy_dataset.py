import argparse
import os
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_base', default='/mnt/ramdisk/ImageNet', type=str, help='the path of dataset')
args = parser.parse_args()


def get_datalist():
    data = []
    label_list = os.listdir(os.path.join(args.data_base, 'train'))
    label_list.sort()
    label = 0
    for item in label_list:
        img_list = os.listdir(os.path.join(args.data_base, 'train', item))
        for img in img_list:
            data.append([os.path.join(args.data_base, 'train', item, img), label])
        label += 1
    return data


def main():
    data_list = get_datalist()
    np.random.shuffle(data_list)

    # save data_list
    np.save(os.path.join('data_list.npy'), data_list)


if __name__ == '__main__':
    main()
