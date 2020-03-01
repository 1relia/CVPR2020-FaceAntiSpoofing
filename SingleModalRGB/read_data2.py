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

class CASIA(Dataset):
    def __init__(self, args, transform=None, phase_train=True, data_dir=None,phase_test=False):
        self.data_root = args.data_root
        image_dir_train_file = self.data_root.replace('phase2', 'phase1') +'4@'+ args.sub_prot_train + '_train.txt'
        image_dir_val_file = self.data_root.replace('phase2', 'phase1') + '4@'+ args.sub_prot_val + '_dev.txt'
        # image_dir_test_file = self.data_root + '4@'+ args.sub_prot_test + '_dev_res.txt'
        if 'phase1' in self.data_root:
            txt = 'dev'
        else:
            txt = 'test'
        image_dir_test_file = self.data_root + '4@' + args.sub_prot_test + '_' + txt + '_res.txt'
        self.phase_train = phase_train
        self.phase_test = phase_test
        self.transform = transform
        self.mode = args.mode
        self.sample_duration = args.sample_duration
        try:
            with open(image_dir_train_file, 'r') as f:
                self.image_dir_train = f.read().splitlines()
                
            with open(image_dir_val_file, 'r') as f:
                 self.image_dir_val = f.read().splitlines()
            if self.phase_test:
                with open(image_dir_test_file, 'r') as f:
                    # self.image_dir_test = f.read().splitlines()
                    depth_dirs = f.read().splitlines()
                    self.image_dir_test = []
                    for dir_ in depth_dirs:
                        image_dir_file = self.data_root + dir_
                        images_each_folder = sorted(glob(os.path.join(image_dir_file, 'profile/*.jpg')))
                        length = len(images_each_folder)
                        # print(length,self.sample_duration,len(images_each_folder[:length-self.sample_duration]))
                        images_dir = [images_each_folder[0]] if  length < self.sample_duration else images_each_folder[:length-self.sample_duration+1]
                        self.image_dir_test = self.image_dir_test + images_dir
        except:
            print('can not open files, may be filelist is not exist')
            exit()
        
    def sample(self,idx, img_dirs, sample_duration=64):
        idx = min(len(img_dirs) - sample_duration,idx)
        length = len(img_dirs)
        samples_dir = []
        image_dir,label = img_dirs[idx].split(' ')
        if self.phase_train and int(label) == 1 and np.random.rand() < 0.3: # label
            samples_dir = samples_dir + [self.data_root + image_dir] * sample_duration
            label = 0.
            return samples_dir, np.array(label)
        label_ = 0.
        for line in img_dirs[idx: idx+sample_duration]:
            image_dir,label = line.split(' ')
            label_ = label_ + int(label)
            samples_dir = samples_dir + [self.data_root + image_dir]

        assert len(samples_dir) == sample_duration
        # samples_dir = sorted(samples_dir)
        return samples_dir, np.array(label_/sample_duration)

    def test_sample(self,idx, img_dirs, sample_duration=64):
        image_dir = img_dirs[idx]
        # print(image_dir[:-8],image_dir[-8:-4])
        parent_dir = image_dir[:-8]
        num = int(image_dir[-8:-4])-1
        label = np.random.randint(0,2,1)
        images_each_folder = sorted(glob(os.path.join(parent_dir, '*.jpg')))
        length = len(images_each_folder)
        samples_dir = []
        if length < sample_duration:
            # samples_dir = samples_dir + images_each_folder + [image_dir] * (len(images_each_folder) -sample_duration)
            samples_dir = samples_dir + [image_dir] * sample_duration
            print(len(samples_dir),sample_duration)
        else:
            samples_dir = images_each_folder[num:num+sample_duration]
        
        # assert len(samples_dir) == sample_duration
        return samples_dir, np.array(label)

    def __len__(self):
        if self.phase_train:
            return len(self.image_dir_train)
        else:
            if self.phase_test:
                return len(self.image_dir_test)
            else:
                return len(self.image_dir_val)
    def get_img_tensor(self,img_dir):
        if self.mode == 'Depth':
            img_dir = img_dir.replace('profile','depth')
        elif self.mode == 'IR':
            img_dir = img_dir.replace('profile','ir')

        if self.phase_train:
            image = adaptive_color_transform(img_dir, p=0.3)
        else:
            image = Image.open(img_dir)
        #depth = Image.open(image_dir)
        image = image.convert('RGB')

        if self.transform:
            image_tenosr = self.transform(image)
        return image_tenosr

    def __getitem__(self, idx):
        if self.phase_train:
            image_dir,label = self.image_dir_train[idx].split(' ')
            image_dirs = self.image_dir_train
            image_dir = self.data_root + image_dir
            label = np.array(int(label))
        else:
            if self.phase_test:
                image_dir = self.image_dir_test[idx]
                image_dirs = self.image_dir_test
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                image_dir,label = self.image_dir_val[idx].split(' ')
                image_dirs = self.image_dir_val
                image_dir = self.data_root + image_dir
                label = np.array(int(label))
        if self.phase_test:
            samples_dir,label = self.test_sample(idx,image_dirs, self.sample_duration)
        else:
            samples_dir,label = self.sample(idx,image_dirs, self.sample_duration)
        # print(samples_dir)
        samples_image = [self.get_img_tensor(img_dir) for img_dir in samples_dir]
        samples = torch.stack(samples_image,0).permute(1, 0, 2, 3) 
        # print(samples.shape) # torch.Size([3, 64, 224, 224])
        if self.phase_train:
            return samples,label
        elif self.phase_test:
            return samples,label,samples_dir[0]
        else:
            label = 0 if label < 0.5 else 1
            return samples,label,samples_dir[0]

