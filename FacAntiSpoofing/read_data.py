from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset
import math
import cv2
import torchvision
import torch
from color_transform import adaptive_color_transform
from glob import glob


# CASIA-SURF training dataset and our private dataset
# data_root = '/home/zp/dataset/CASIA-CeFA/phase1/'
# depth_dir_train_file = data_root +'4@3_train.txt'

# # for IR train
# # depth_dir_train_file = os.getcwd() +'/data/ir_final_train.txt'
# # label_dir_train_file = os.getcwd() +'/data/label_ir_train.txt'

# # CASIA-SURF Val data 
# depth_dir_val_file = data_root +'4@2_train.txt'


# depth_dir_val_file = os.getcwd() +'/data/ir_val.txt'

# # CASIA-SURF Test data 
# depth_dir_test_file = data_root + '4@1_dev_res.txt'

# depth_dir_test_file = os.getcwd() +'/data/ir_test.txt'

class CASIA(Dataset):
    def __init__(self, args, transform=None, phase_train=True, data_dir=None, phase_test=False):
        self.data_root = args.data_root
        depth_dir_train_file = self.data_root.replace('phase2', 'phase1') + '4@' + args.sub_prot_train + '_train.txt'
        depth_dir_val_file = self.data_root.replace('phase2', 'phase1') + '4@' + args.sub_prot_val + '_dev.txt'
        if 'phase1' in self.data_root:
            txt = 'dev'
        else:
            txt = 'test'
        depth_dir_test_file = self.data_root + '4@' + args.sub_prot_test + '_' + txt + '_res.txt'
        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.mode = args.mode
        try:
            with open(depth_dir_train_file, 'r') as f:
                #                 self.depth_dir = f.read ().splitlines()
                self.depth_dir_train = f.read().splitlines()

            with open(depth_dir_val_file, 'r') as f:
                self.depth_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(depth_dir_test_file, 'r') as f:
                    depth_dirs = f.read().splitlines()
                    self.depth_dir_test = []
                    for dir_ in depth_dirs:
                        depth_dir_file = self.data_root + dir_
                        self.depth_dir_test = self.depth_dir_test + sorted(
                            glob(os.path.join(depth_dir_file, 'profile/*.jpg')))
                    # self.depth_dir_test = sorted(self.depth_dir_test)
        except:
            print('can not open files, may be filelist is not exist')
            exit()

    #         train_len = int(len(self.depth_dir) * 0.7)
    # #         np.random.shuffle(self.depth_dir)
    #         self.depth_dir_train = self.depth_dir[0:train_len]
    #         self.depth_dir_val = self.depth_dir[train_len:]
    #         self.depth_dir_test = self.depth_dir[train_len:]

    def __len__(self):
        if self.phase_train:
            return len(self.depth_dir_train)
        else:
            if self.phase_test:
                return len(self.depth_dir_test)
            else:
                return len(self.depth_dir_val)

    def __getitem__(self, idx):
        if self.phase_train:
            depth_dir, label = self.depth_dir_train[idx].split(' ')
            depth_dir = self.data_root + depth_dir
            label = np.array(int(label))
        else:
            if self.phase_test:
                depth_dir = self.depth_dir_test[idx]
                label = np.random.randint(0, 2, 1)
                label = np.array(label)
            else:
                depth_dir, label = self.depth_dir_val[idx].split(' ')
                depth_dir = self.data_root + depth_dir
                label = np.array(int(label))
        if self.mode == 'Depth':
            depth_dir = depth_dir.replace('profile', 'depth')
        elif self.mode == 'IR':
            depth_dir = depth_dir.replace('profile', 'ir')

        # if self.phase_train:
        #   depth = adaptive_color_transform(depth_dir, p=0.3)
        # else:
        #   depth = Image.open(depth_dir)
        depth = Image.open(depth_dir)
        depth = depth.convert('RGB')

        if self.transform:
            depth = self.transform(depth)
        if self.phase_train:
            return depth, label
        else:
            return depth, label, depth_dir

