import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import scipy.io as scio
import scipy.misc as misc
from PIL import Image
import shutil


class DATA_PAIR(Dataset):
    def __init__(self, data_dir, view1dim=3, view2dim=3):
        # 获取数据存放的dir
        # 例如d:/images/
        self.view1_list = os.listdir(os.path.join(data_dir,'view1'))
        self.view2_list = os.listdir(os.path.join(data_dir,'view2'))

        self.view1dim = view1dim
        self.view2dim = view2dim
        generate_map(data_dir)
        with open(os.path.join(data_dir, 'map.txt'), 'r') as fp:
            content = fp.readlines()
            str_list = [s.rstrip().split() for s in content]
            self.pair_list = [(x[0], x[1]) for x in str_list]

    def reset(self, NP_ratio):
        self.index_x, self.index_y, self.label = generate_negative(len(self.pair_list), NP_ratio)

    def __getitem__(self, index):

        img_view1 = self.view1_list[index]
        img_view2 = self.view2_list[index]

        data_totensor = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        img_view1 = data_totensor(img_view1)
        img_view2 = data_totensor(img_view2)


        data_transforms_view1 = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        data_transforms_view2 = transforms.Compose([
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        img_view1 = data_transforms_view1(img_view1)

        img_view2 = data_transforms_view2(img_view2)


        label = float(0)
        return img_view1, img_view2, label

    def __len__(self):
        return len(self.view1_list)



def generate_negative(num, NP_ratio):
    # print("Generating negative samples...")
    index_x = np.arange(num).repeat(1 + NP_ratio)
    index_y_ = np.random.randint(0, num, [num, NP_ratio])
    index_y = np.append(np.arange(num).reshape(num, 1), index_y_, 1)

    np_label = index_y == index_y[:, 0].reshape(num, 1)
    np_label = np.array(np_label).astype(float).reshape(num * (1 + NP_ratio), 1)
    index_y = index_y.reshape(num * (1 + NP_ratio))
    return index_x, index_y, np_label


def generate_map(root_dir):
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    with open(os.path.join(root_dir, 'map.txt'), 'w') as wfp:
        dir_view1 = os.path.join(root_dir, 'view1')
        dir_view2 = os.path.join(root_dir, 'view2')
        for file_name in os.listdir(dir_view1):
            view1_name = os.path.join(father_path, dir_view1, file_name)
            view2_name = os.path.join(father_path, dir_view2, file_name)

            linux_view1_name = view1_name.replace("\\", '/')
            linux_view2_name = view2_name.replace("\\", '/')
            wfp.write('{view1_dir} {view2_dir}\n'.format(view1_dir=linux_view1_name, view2_dir=linux_view2_name))


def generate_neg(root_dir, pair_list, NP_ratio):
    index_x, index_y, label = generate_negative(len(pair_list), NP_ratio)
    num = len(label)
    with open(os.path.join(root_dir, 'pairing.txt'), 'w') as wfp:
        for i in range(num):
            view1_name = pair_list[index_x[i]][0]
            view2_name = pair_list[index_y[i]][1]
            linux_view1_name = view1_name.replace("\\", '/')
            linux_view2_name = view2_name.replace("\\", '/')
            wfp.write(
                '{view1_dir} {view2_dir} {label}\n'.format(view1_dir=linux_view1_name, view2_dir=linux_view2_name,
                                                       label=label[i, 0]))


