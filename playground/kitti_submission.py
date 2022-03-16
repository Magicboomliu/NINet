from __future__ import print_function
import sys
sys.path.append("../")
import argparse
import os, sys
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import skimage
import skimage.io
import numpy as np
import time
import math
from networks.EDNet.pretrain_affinity.pretrain_affinity_conf_nl import EDNet
from networks.EDNet.normalE.normal_estimation import NormalNet
from utils.file_io import read_disp,read_img,read_kitti_step1,read_kitti_step2
from dataloader import transforms
import matplotlib.pyplot as plt 



# Image RGB LIST
def default_transform():
    rgb_list = [transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    return transforms.Compose(rgb_list)




IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(description='KITTI_TEST')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/data/dataset/kitti_stereo/kitti_2015/testing',
                    help='select model')
parser.add_argument('--loadmodel', default=None,
                    help='loading model')
parser.add_argument('--norm_path', default=None,
                    help='loading Norm model')
parser.add_argument('--savepath', default='results/',
                    help='path to save the results.')
parser.add_argument('--model', default='fadnet',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--devices', type=str, help='indicates CUDA devices, e.g. 0,1,2', default='0')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Make Saved Path
if not os.path.exists(args.savepath):
    os.makedirs(args.savepath)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.KITTI == '2015':
    from dataloader import kitti_submission_dataloder as DA
else:
    raise NotImplementedError

# Read The Dataloader 
test_left_img, test_right_img = DA.dataloader(args.datapath)
devices = [int(item) for item in args.devices.split(',')]
ngpus = len(devices)

# Normal Model
normnet = NormalNet()
normnet = nn.DataParallel(normnet, device_ids=devices)
normnet.cuda()
if args.norm_path is not None:
    state_dict_norm = torch.load(args.norm_path)
    normnet.load_state_dict(state_dict_norm['state_dict'])
    print('load pretrained Normal model Successfully')

# Disp Model
model = EDNet(res_type='deform_norm',squeezed_volume=True,max_disp=192)
model = nn.DataParallel(model, device_ids=devices)
model.cuda()
if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    print('load pretrained Disparity model Successfully')

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# Image Normalization and Resize
def ImageScale(left_image,right_image,crop_height=384,crop_weight=1280):
    sample = dict()
    sample["img_left"] = left_image
    sample["img_right"] = right_image
    rgb_transform = default_transform()
    sample = rgb_transform(sample) # [3,H,W]
    # Resize 
    imgL = sample['img_left'].numpy()
    imgR = sample['img_right'].numpy()
    imgL = np.reshape(imgL,[1,3,imgL.shape[1],imgL.shape[2]])
    imgR = np.reshape(imgR,[1,3,imgR.shape[1],imgR.shape[2]])
    h,w = imgL.shape[-2:]
    top_pad = crop_height-imgL.shape[2]
    left_pad = crop_weight-imgL.shape[3]
    imgL = np.lib.pad(imgL,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    imgR = np.lib.pad(imgR,((0,0),(0,0),(top_pad,0),(0,left_pad)),mode='constant',constant_values=0)
    return imgL,imgR,h,w

def test(imgL,imgR):
    model.eval()
    normnet.eval()
    if args.cuda:
        imgL = torch.FloatTensor(imgL).cuda()
        imgR = torch.FloatTensor(imgR).cuda() 
    imgL, imgR= Variable(imgL), Variable(imgR)
    with torch.no_grad():
        assisted_normal = normnet(imgL,imgR,False)
        output,conf_infer = model(imgL, imgR, assisted_normal,False)
        output = torch.squeeze(output, dim=1)
        output = torch.squeeze(output, dim=0)

    pred_disp = output.data.cpu().numpy()
    
    return pred_disp
    
    


def main():
    time_total = 0
    for inx in range(len(test_left_img)):
        #print('image: %s'%test_left_img[inx])
        # Left Image and Right Image
        imgL_o = read_img(test_left_img[inx])
        imgR_o = read_img(test_right_img[inx])
        imgL,imgR,h,w = ImageScale(imgL_o,imgR_o,384,1280)
        pred_disp = test(imgL,imgR)
        
        top_pad = 384-h
        left_pad = 1280-w
        img = pred_disp[top_pad:,:-left_pad]
        round_img = np.round(img*256)
        
        skimage.io.imsave(os.path.join(args.savepath, test_left_img[inx].split('/')[-1]),round_img.astype('uint16'))
        
        plt.figure(figsize=(10,5))
        plt.imshow(round_img)
        plt.axis("off")
        plt.savefig("vis/kitti2015_1/{}.png".format(inx))


if __name__ == '__main__':
   main()
