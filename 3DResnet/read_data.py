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
        image_dir_train_file = self.data_root.replace('phase2', 'phase1') +'4@'+ args.sub_prot_train + '_train_ref.txt'
        image_dir_val_file = self.data_root.replace('phase2', 'phase1') + '4@'+ args.sub_prot_val + '_dev_ref.txt'
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
                    self.image_dir_test = f.read().splitlines()
                print(len(self.image_dir_test))
        except:
            print('can not open files, may be filelist is not exist')
            exit()
        
    def sample(self,label, image_dir_file, sample_duration=64):
        print(image_dir_file)
        img_dirs = glob(os.path.join(image_dir_file,'profile/*.jpg'))
        length = len(img_dirs)
        samples_dir = []
        if self.phase_train and label == 1 and np.random.rand() < 0.3: # label
            samples_dir = samples_dir + [img_dirs[np.random.randint(length-1)]] * sample_duration
            label = 0
            return samples_dir, np.array(label)

        if length > sample_duration:
            # np.random.shuffle(img_dirs)
            # samples_dir = img_dirs[:sample_duration]
            start = np.random.randint(length-sample_duration-1)
            samples_dir = img_dirs[start:start+sample_duration]
        else:
            sample_ratio = int(sample_duration/length)
            for img_dir in img_dirs:
                samples_dir = samples_dir + [img_dir] * sample_ratio
            samples_dir = samples_dir + [img_dirs[np.random.randint(length-1)]] * (sample_duration - len(samples_dir))
        assert len(samples_dir) == sample_duration
        samples_dir = sorted(samples_dir)
        return samples_dir, label

    def test_sample(self, label, image_dir_file, sample_duration=64):
        #print(image_dir_file)
        img_dirs = glob(os.path.join(image_dir_file,'profile/*.jpg'))
        length = len(img_dirs)
        #print(length)
        samples_dir = []

        if length > sample_duration:
            # np.random.shuffle(img_dirs)
            # samples_dir = img_dirs[:sample_duration]
            #print(length-sample_duration-1)
            if length-sample_duration-1 == 0 :
                start = 0
            else:
                start = np.random.randint(length-sample_duration-1)
            samples_dir = img_dirs[start:start+sample_duration]
        else:
            sample_ratio = int(sample_duration/length)
            for img_dir in img_dirs:
                samples_dir = samples_dir + [img_dir] * sample_ratio
            samples_dir = samples_dir + [img_dirs[np.random.randint(length-1)]] * (sample_duration - len(samples_dir))
        assert len(samples_dir) == sample_duration
        samples_dir = sorted(samples_dir)
        return samples_dir, label


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
            image_dir = self.data_root + image_dir
            label = np.array(int(label))
        else:
            if self.phase_test:
                image_dir = self.image_dir_test[idx]
                image_dir = self.data_root + image_dir
                label = np.random.randint(0,2,1)
                label = np.array(label)
            else:
                image_dir,label = self.image_dir_val[idx].split(' ')
                image_dir = self.data_root + image_dir
                label = np.array(int(label))
        if self.phase_test:
            samples_dir, label = self.test_sample(label, image_dir, self.sample_duration)
        else:
            samples_dir,label = self.sample(label, image_dir, self.sample_duration)

        samples_image = [self.get_img_tensor(img_dir) for img_dir in samples_dir]
        samples = torch.stack(samples_image,0).permute(1, 0, 2, 3) 
        # print(samples.shape) # torch.Size([3, 64, 224, 224])
        if self.phase_train:
            return samples,label
        else:
            return samples,label,samples_dir[0]

