import argparse
import os
import numpy as np
import random
from PIL import Image
import shutil


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_base', default='/mnt/ramdisk/CUB/trainval', type=str, help='the path of dataset')
parser.add_argument('--save_path', default='/mnt/ramdisk/CUB/cub6w', type=str, help='the path of dataset')
args = parser.parse_args()


def get_datalist():
    data = []
    label_list = os.listdir(os.path.join(args.data_base, 'train'))
    label_list.sort()
    for item in label_list:
        label = int(item.split('.')[0]) - 1
        img_list = os.listdir(os.path.join(args.data_base, 'train', item))
        for img in img_list:
            data.append([os.path.join(args.data_base, 'train', item, img), label])
    return data


def original_img(file_list, data_list):
    os.makedirs(os.path.join(args.save_path, 'ori'), exist_ok=True)
    for i, item in enumerate(data_list):
        print('[original image] processing {0} ...'.format(i))
        shutil.copyfile(item[0], os.path.join(args.save_path, 'ori', str(i)+'.jpg'))
        file_list.append([os.path.join(args.save_path, 'ori', str(i) + '.jpg'), item[1]])


def img_sum(file_list, data_list):
    os.makedirs(os.path.join(args.save_path, 'sum'), exist_ok=True)
    for i, item in enumerate(data_list):
        print('[sum image] processing {0} ...'.format(i))
        ind = random.sample(range(len(data_list)), 1)
        x_1 = Image.open(item[0])
        x_2 = Image.open(data_list[ind[0]][0])

        # mix up
        lam = np.random.beta(1, 1)
        x = lam * np.array(x_1) + (1 - lam) * np.array(x_2.resize(x_1.size))
        x = Image.fromarray(np.uint8(x))

        # save
        file_list.append(os.path.join(args.save_path, 'sum', str(i)+'.jpg'))
        x.save(os.path.join(args.save_path, 'sum', str(i)+'.jpg'))


def img_concate(file_list, data_list):
    os.makedirs(os.path.join(args.save_path, 'concate'), exist_ok=True)
    for i, item in enumerate(data_list):
        print('[concate image] processing {0} ...'.format(i))
        [ind1, ind2, ind3] = random.sample(range(len(data_list)), 3)
        x_1 = Image.open(item[0])
        x_2 = Image.open(data_list[ind1][0])
        x_3 = Image.open(data_list[ind2][0])
        x_4 = Image.open(data_list[ind3][0])

        # concate
        [x1, y1] = x_1.size

        x_2 = x_2.resize([x1, y1])
        x_3 = x_3.resize([x1, y1])
        x_3 = x_3.resize([x1, y1])
        x_4 = x_4.resize([x1, y1])

        x = Image.new(x_1.mode, (2 * x1, 2 * y1))
        x.paste(x_1, [0, 0])
        x.paste(x_2, (0, y1))
        x.paste(x_3, (x1, 0))
        x.paste(x_4, (x1, y1))

        # save
        file_list.append(os.path.join(args.save_path, 'concate', str(i)+'.jpg'))
        x.save(os.path.join(args.save_path, 'concate', str(i)+'.jpg'))


def img_shuffle(file_list, data_list, block_size):
    # to be continued
    os.makedirs(os.path.join(args.save_path, 'shuffle'), exist_ok=True)
    for i, item in enumerate(data_list):
        print('[shuffle image] processing {0} ...'.format(i))
        x_1 = Image.open(item[0])
        [x, y] = x_1.size
        x1 = block_size*int(x/block_size)
        y1 = block_size*int(y/block_size)

        # shuffle
        order_list = list(range(block_size**2))
        np.random.shuffle(order_list)
        data = np.array(x_1.resize([x1, y1]))
        data_new = np.ones(data.shape)
        for idx, item1 in enumerate(order_list):
            ind_i = int(item1 // block_size)
            ind_j = int(item1 % block_size)
            index_i = int(idx // block_size)
            index_j = int(idx % block_size)
            x_length = int(x1 / block_size)
            y_length = int(y1 / block_size)
            tmp = data[y_length * ind_j:y_length * (ind_j + 1), x_length * ind_i:x_length * (ind_i + 1), :]
            data_new[y_length * index_j:y_length * (index_j + 1), x_length * index_i:x_length * (index_i + 1), :] = tmp
        x = Image.fromarray(np.uint8(data_new))

        # save
        file_list.append([os.path.join(args.save_path, 'shuffle', str(i)+'_'+str(block_size)+'.jpg'), item[1]])
        x.save(os.path.join(args.save_path, 'shuffle', str(i)+'_'+str(block_size)+'.jpg'))


def cut_out(file_list, data_list):
    # to be continued
    os.makedirs(os.path.join(args.save_path, 'cut'), exist_ok=True)
    for ind_, item in enumerate(data_list):
        print('[cutout image] processing {0} ...'.format(ind_))
        x_1 = Image.open(item[0])
        [y, x] = x_1.size

        alpha = random.uniform(0.2, 0.5)
        x_l = int(alpha * x)
        y_l = int(alpha * y)
        x_1 = np.array(x_1)
        i = int(random.sample(range(x-x_l), 1)[0])
        j = int(random.sample(range(y-y_l), 1)[0])
        x_1[i:i+x_l, j:j+y_l] = 0
        x = Image.fromarray(np.uint8(x_1))

        # save
        file_list.append([os.path.join(args.save_path, 'cut', str(ind_) + '.jpg'), item[1]])
        x.save(os.path.join(args.save_path, 'cut', str(ind_) + '.jpg'))


def rotate(file_list, data_list, name):
    # to be continued
    os.makedirs(os.path.join(args.save_path, 'rotate'), exist_ok=True)
    for i, item in enumerate(data_list):
        print('[rotate image] processing {0} ...'.format(i))
        x = Image.open(item[0])

        angle = random.randint(0, 360)
        x = x.rotate(angle)

        # save
        file_list.append([os.path.join(args.save_path, 'rotate', str(i) + '_' + name + '.jpg'), item[1]])
        x.save(os.path.join(args.save_path, 'rotate', str(i) + '_' + name + '.jpg'))


def main():
    file_list = []
    data_list = get_datalist()
    np.random.shuffle(data_list)

    original_img(file_list, data_list)
    # img_sum(file_list, data_list)  # 1
    # img_concate(file_list, data_list)  # 2 PASS
    img_shuffle(file_list, data_list, 2)  # 3
    img_shuffle(file_list, data_list, 3)  # 4
    img_shuffle(file_list, data_list, 4)  # 5
    cut_out(file_list, data_list)  # 6
    rotate(file_list, data_list, '0')  # 7
    # rotate(file_list, data_list, '1')  # 8
    # rotate(file_list, data_list, '2')  # 9
    # rotate(file_list, data_list, '3')  # 9

    # save data_list
    np.save(os.path.join(args.save_path, 'data_list.npy'), file_list)


if __name__ == '__main__':
    main()
