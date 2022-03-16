import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from networks.utilsNet.Disparity_warper import disp_warp
import os
from utils.file_io import read_img, read_disp
from dataloader import transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils.common import count_parameters 
from dataloader.dataloader import StereoDataset


# ImageNet Normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]# ImageNet Normalization


def prepare_dataset(file_path,train_list,val_list):
    test_batch =1
    num_works = 1
    val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
    val_transform = transforms.Compose(val_transform_list)

    test_dataset = StereoDataset(data_dir=file_path,train_datalist=train_list,test_datalist=val_list,
                                    dataset_name='SceneFlow',mode='val',transform=val_transform)
    scale_height, scale_width = test_dataset.get_scale_size()
    test_loader = DataLoader(test_dataset, batch_size = test_batch, \
                                shuffle = False, num_workers = num_works, \
                                pin_memory = True)
    return test_loader


def get_occlusion_mask(infer_disparity,left_image,right_image):
    assert infer_disparity.min()>=0
    scale = left_image.size()[2] / infer_disparity.size()[2]

    disp_ = infer_disparity / scale

    left_rec ,mask= disp_warp(right_image,disp_)
    error_map = left_rec - left_image

    return error_map

if __name__=="__main__":
    path ="/home/zliu/Desktop/liuzihua/codes/GIthub repos/clean/StereoMatching_CVPR2022/networks/utilsNet/Samples"
    left_image_path = os.path.join(path,"L.png")
    right_image_path = os.path.join(path,"R.png")
    disp_path = os.path.join(path,"disp.pfm")


    root_file_path = "/media/zliu/datagrid1/sceneflow"
    train_list = "filenames/SceneFlow.list"
    val_list = "filenames/FlyingThings3D_release_TEST.list"
    test_loader = prepare_dataset(root_file_path,train_list,val_list)

    for i, sample_batched in enumerate(test_loader):
        left_input = Variable(sample_batched['img_left'].cuda(), requires_grad=False)
        right_input = Variable(sample_batched['img_right'].cuda(), requires_grad=False)

        target_disp = sample_batched['gt_disp']
        target_disp = target_disp.cuda()
        target_disp =Variable(target_disp, requires_grad=False)

        print(left_input.shape)
        print(target_disp.shape)

        break