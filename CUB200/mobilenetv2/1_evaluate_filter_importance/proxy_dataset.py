import argparse
import os
import numpy as np
import yaml


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--yaml_file', default='../config.yaml', type=str, help='yaml config file')
args = parser.parse_args()
f = open(args.yaml_file)
FLAGS = yaml.load(f)


def get_datalist():
    data = []
    label_list = os.listdir(os.path.join(FLAGS['data_base'], 'train'))
    label_list.sort()
    for label, item in enumerate(label_list):
        img_list = os.listdir(os.path.join(FLAGS['data_base'], 'train', item))
        for img in img_list:
            data.append([os.path.join(FLAGS['data_base'], 'train', item, img), label])
    return data


def main():
    data_list = get_datalist()
    np.random.shuffle(data_list)

    # save data_list
    np.save(os.path.join('data_list.npy'), data_list)


if __name__ == '__main__':
    main()
