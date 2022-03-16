import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import os
from dataloader.kitti_dataloader_pseudo import StereoDataset
from dataloader import transforms
import matplotlib.pyplot as plt
from utils.normal_cal import get_normal
import matplotlib.pyplot as plt
# from networks.EDNet.normalE.normal_estimation import NormalNet
# from networks.EDNet.pretrain_affinity.pretrain_affinity_conf_nl import EDNet
from utils.colormaps import apply_colormap
# Visualization Tensor to Numpy
def visualized(tensor):
    N,C,H,W = tensor.shape
    if C==1:
        array = tensor.squeeze(0).squeeze(0).cpu().numpy()
    elif C==3:
        array = tensor.squeeze(0).permute(1,2,0).cpu().numpy()
    else:
        raise NotImplementedError
    return array


#ImageNet Bias
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def prepare_dataset(dataset="KITTI_mix",datapath=None,trainlist=None,vallist=None):
    if dataset == 'KITTI_mix':
        train_transform_list = [transforms.RandomCrop(256, 512),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        train_transform = transforms.Compose(train_transform_list)

        val_transform_list = [transforms.ToTensor(),
                        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                        ]
        val_transform = transforms.Compose(val_transform_list)
        
        train_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                dataset_name='KITTI_mix',mode='train',transform=train_transform)
        test_dataset = StereoDataset(data_dir=datapath,train_datalist=trainlist,test_datalist=vallist,
                                dataset_name='KITTI_mix',mode='test',transform=val_transform)

    img_height, img_width = train_dataset.get_img_size()

    scale_height, scale_width = test_dataset.get_scale_size()

    datathread=4
    if os.environ.get('datathread') is not None:
        datathread = int(os.environ.get('datathread'))
    

    train_loader = DataLoader(train_dataset, batch_size = 1, \
                            shuffle = True, num_workers = datathread, \
                            pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = 1, \
                            shuffle = False, num_workers = datathread, \
                            pin_memory = True)
    num_batches_per_epoch = len(train_loader)

    return train_loader,test_loader

if __name__=="__main__":
    train_list = "../filenames/KITTI_mix_train_psedo.txt"
    val_list = "../filenames/KITTI_mix_val_psedo.txt"
    train_loader, test_loader = prepare_dataset(dataset='KITTI_mix',datapath='/datagrid01/liu/kitti_stereo',
                                    trainlist=train_list,vallist=val_list)
    # # Normal Net
    # normal_net = NormalNet().cuda()
    # saved_path = "/data/KITTI/new/models_saved/normal_only_LL1/model_best.pth"
    # # saved_path = "/data/KITTI/norm0016.pth"
    # model_data = torch.load(saved_path)
    # normal_net = torch.nn.DataParallel(normal_net, device_ids=[0]).cuda()
    # normal_net.load_state_dict(model_data['state_dict'])
    # # Disparity Net
    # model_path = "/data/codes/CODES_MODELS/NL_2/models_saved/kitti_320/model_best.pth"
    # # model_path = "/data/codes/CODES_MODELS/non_local_v2_m/non_local_conf/model_best.pth"
    # NSLPNet = EDNet(res_type='deform_norm',squeezed_volume=True).cuda()
    # NSLPNet = torch.nn.DataParallel(NSLPNet, device_ids=[0]).cuda()
    # model_data2 = torch.load(model_path)
    # NSLPNet.load_state_dict(model_data2["state_dict"])

    # normal_net.eval()
    # NSLPNet.eval()
    for i_batch, sample_batched in enumerate(test_loader):
    
        left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
        right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)

        target_disp = sample_batched['gt_disp'].unsqueeze(1)
        target_disp = target_disp.cuda()

        
        pseudo_disp = sample_batched['pseudo_disp'].unsqueeze(1)
        pseudo_disp = pseudo_disp.cuda()
        pseudo_disp = torch.autograd.Variable(pseudo_disp, requires_grad=False)
        

      
        # with torch.no_grad():
        #     assisted_normal =normal_net(left_input,right_input,False)
        #     output,conf_infer = NSLPNet(left_input, right_input,assisted_normal,False)
        #     plt.figure(figsize=(10,5))
        #     plt.subplot(1,2,1)
        #     output = torch.clip(output / 192 * 255, 0, 255).long()
        #     output = apply_colormap(output)
        #     print(output.shape)
        #     plt.imshow(output.squeeze(0).permute(1,2,0).cpu().numpy())
        #     plt.subplot(1,2,2)
        #     plt.imshow(conf_infer.squeeze(0).squeeze(0).cpu().numpy(),cmap='coolwarm')
        #     plt.savefig("vis/320/{}.png".format(i_batch))



            
