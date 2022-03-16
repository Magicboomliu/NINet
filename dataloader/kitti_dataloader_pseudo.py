from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch.utils.data import Dataset
import os
import sys
sys.path.append("../")
from utils import utils
from utils.file_io import read_img, read_disp,default_loader,converter,read_slant,read_kitti_step1,read_kitti_step2,load_psedo_kitti
from skimage import io, transform
import numpy as np

class StereoDataset(Dataset):
    def __init__(self, data_dir,
                 train_datalist,
                 test_datalist,
                 dataset_name='SceneFlow',
                 mode='train',
                 save_filename=False,
                 load_pseudo_gt=False,
                 transform=None):
        super(StereoDataset, self).__init__()

        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.mode = mode
        self.save_filename = save_filename
        self.transform = transform
        self.train_datalist = train_datalist
        self.test_datalist = test_datalist
        self.img_size=(256, 512)
        self.scale_size =(384,1280)
        self.original_size =(375,1280)
        

        sceneflow_finalpass_dict = {
            'train':  self.train_datalist,
            'val':    self.test_datalist,
            'test':   self.test_datalist 
        }

        kitti_2012_dict = {
            'train': 'filenames/KITTI_2012_train.txt',
            # 'train_all': 'filenames/KITTI_2012_train_all.txt',
            'val': 'filenames/KITTI_2012_val.txt',
            # 'test': 'filenames/KITTI_2012_test.txt'
        }

        kitti_2015_dict = {
            'train': 'filenames/KITTI_2015_train.txt',
            # 'train_all': 'filenames/KITTI_2015_train_all.txt',
            'val': 'filenames/KITTI_2015_val.txt',
            # 'test': 'filenames/KITTI_2015_test.txt'
        }

        kitti_mix_dict = {
            'train': self.train_datalist,
            'test': self.test_datalist
        }

        dataset_name_dict = {
            'SceneFlow': sceneflow_finalpass_dict,
            'KITTI2012': kitti_2012_dict,
            'KITTI2015': kitti_2015_dict,
            'KITTI_mix': kitti_mix_dict,
        }

        assert dataset_name in dataset_name_dict.keys()
        self.dataset_name = dataset_name

        self.samples = []

        data_filenames = dataset_name_dict[dataset_name][mode]

        lines = utils.read_text_lines(data_filenames)

        for line in lines:
            splits = line.split()

            left_img, right_img = splits[:2]
            gt_disp = None if len(splits) == 2 else splits[2]
            pseudo_disp = None if len(splits) ==3 else splits[3]
            sample = dict()

            if self.save_filename:
                sample['left_name'] = left_img.split('/', 1)[1]

            sample['left'] = os.path.join(data_dir, left_img)
            sample['right'] = os.path.join(data_dir, right_img)
            sample['disp'] = os.path.join(data_dir, gt_disp) if gt_disp is not None else None
            sample['pseudo_disp'] = os.path.join(data_dir,pseudo_disp) if pseudo_disp is not None else None
            self.samples.append(sample)

    def __getitem__(self, index):
        sample = {}
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        if self.mode=='train':
            sample['img_left'] = read_img(sample_path['left'])  # [H, W, 3]
            sample['img_right'] = read_img(sample_path['right'])
        
        # GT disparity of subset if negative, finalpass and cleanpass is positive
        subset = True if 'subset' in self.dataset_name else False
        if sample_path['disp'] is not None:
            sample['gt_disp'] = read_disp(sample_path['disp'], subset=subset)  # [H, W]
        if sample_path['pseudo_disp'] is not None:
            sample['pseudo_disp'] = load_psedo_kitti(sample_path['pseudo_disp'])
        
        if self.mode=='test' or self.mode=='val':
            # 在这里进行剪彩操作？
            img_left = default_loader(sample_path['left'])
            img_right = default_loader(sample_path['right'])
            w,h = img_left.size
            img_left = img_left.crop((w-1280, h-384, w, h))
            img_right = img_right.crop((w-1280, h-384, w, h))
            img_left = converter(img_left)
            img_right = converter(img_right)
            sample['img_left'] = img_left
            sample['img_right'] = img_right
            # Disparity GT 
            gt_disp = read_kitti_step1(sample_path['disp'])
            gt_disp = gt_disp.crop((w-1280, h-384, w, h))
            gt_disp = read_kitti_step2(gt_disp)
            sample['gt_disp']= gt_disp
            # Psedo GT
            pseudo_disp_scale = np.zeros_like(sample['gt_disp'])
            H,W = sample['pseudo_disp'].shape
            pseudo_disp_scale[384-H:,1280-W:] = sample['pseudo_disp']
            sample['pseudo_disp'] = pseudo_disp_scale
            

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.samples)
    
    def get_img_size(self):
        return self.img_size

    def get_scale_size(self):
        return self.scale_size
    
    def get_old_size(self):
        return self.original_size
