from __future__ import print_function
import sys
sys.path.append("../")
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.AverageMeter import AverageMeter
from utils.common import logger, check_path, write_pfm
from losss.disp_conf_loss_kitti import EPE_Loss
from utils.metric import P1_metric
from utils.visualization import disp_error_image_func
from dataloader.preprocess import scale_disp
import skimage
from utils.visual import disp_error_img,save_images
from networks.EDNet.pretrain_affinity.pretrain_affinity_conf_nl import EDNet
from networks.EDNet.normalE.normal_estimation import NormalNet
from utils.common import count_parameters
#from dataloader.kitti_dataloader_with_pse import StereoDataset
from dataloader.kitti_dataloader_pseudo import StereoDataset
from dataloader import transforms
from playground.gen_disp import gen_disp
from dataloader.preprocess import scale_disp
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class DisparityTrainer(object):
    def __init__(self, lr, devices, dataset, trainlist, vallist, datapath, batch_size, maxdisp,use_deform=False, pretrain=None, model='StereoNet', test_batch=4,normal_path=None):
        super(DisparityTrainer, self).__init__()
        self.lr = lr
        self.current_lr = lr
        self.devices = devices
        self.devices = [int(item) for item in devices.split(',')]
        ngpu = len(devices)
        self.ngpu = ngpu
        self.trainlist = trainlist
        self.vallist = vallist
        self.dataset = dataset
        self.datapath = datapath
        self.batch_size = batch_size
        self.test_batch = test_batch
        self.pretrain = pretrain 
        self.maxdisp = maxdisp
        self.use_deform= use_deform
        self.normal_path = normal_path

        self.criterion = None
        self.epe = EPE_Loss
        self.p1_error = P1_metric
        self.model = model
        self.initialize()

    def _prepare_dataset(self):
        if self.dataset == 'kitti':
            train_transform_list = [transforms.RandomCrop(320, 640),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
            train_transform = transforms.Compose(train_transform_list)

            val_transform_list = [transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
            val_transform = transforms.Compose(val_transform_list)
            
            train_dataset = StereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='KITTI_mix',mode='train',transform=train_transform)
            test_dataset = StereoDataset(data_dir=self.datapath,train_datalist=self.trainlist,test_datalist=self.vallist,
                                    dataset_name='KITTI_mix',mode='test',transform=val_transform)

        self.img_height, self.img_width = train_dataset.get_img_size()

        self.scale_height, self.scale_width = test_dataset.get_scale_size()


        datathread=4
        if os.environ.get('datathread') is not None:
            datathread = int(os.environ.get('datathread'))
        logger.info("Use %d processes to load data..." % datathread)

        self.train_loader = DataLoader(train_dataset, batch_size = self.batch_size, \
                                shuffle = True, num_workers = datathread, \
                                pin_memory = True)

        self.test_loader = DataLoader(test_dataset, batch_size = self.test_batch, \
                                shuffle = False, num_workers = datathread, \
                                pin_memory = True)
        self.num_batches_per_epoch = len(self.train_loader)

    
    def __load_the_normal_estimation_model(self):
        self.normal_net = NormalNet()
        self.normal_net = torch.nn.DataParallel(self.normal_net, device_ids=self.devices).cuda()
        model_data = torch.load(self.normal_path)
        print(model_data.keys())
        if 'state_dict' in model_data.keys():
            self.normal_net.load_state_dict(model_data['state_dict'])
        else:
            self.normal_net.load_state_dict(model_data)
        num_of_parameters = count_parameters(self.normal_net)
        print('Model: %s, # of parameters: %d' % ("Normal", num_of_parameters))
        self.normal_net.eval()
       
        
    def _build_net(self):
        # Build the Network architecture according to the model name
        if self.model == 'EDNet': 
            self.net= EDNet(res_type='deform_norm',squeezed_volume=True,max_disp=192)
            
        self.is_pretrain = False
        if self.ngpu > 1:
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        else:
            # self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=self.devices).cuda()
        print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in self.net.parameters()])))

        if self.pretrain == 'none':
            logger.info('Initial a new model...')
            pretrain_ckpt = "/home/zliu/Desktop/new_test/Backups/model_best.pth"
            ckpt = torch.load(pretrain_ckpt)
            current_model_dict = self.net.state_dict()

            useful_dict ={k:v for k,v in ckpt['state_dict'].items() if k in current_model_dict.keys()}
            current_model_dict.update(useful_dict)
            self.net.load_state_dict(current_model_dict)

        else:
            if os.path.isfile(self.pretrain):
                model_data = torch.load(self.pretrain)
                logger.info('Load pretrain model: %s', self.pretrain)
                if 'state_dict' in model_data.keys():
                    self.net.load_state_dict(model_data['state_dict'])
                    
                else:
                    self.net.load_state_dict(model_data)
                self.is_pretrain = True
            else:
                logger.warning('Can not find the specific model %s, initial a new model...', self.pretrain)

    def _build_optimizer(self):
        beta = 0.999
        momentum = 0.9
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), self.lr,
                                        betas=(momentum, beta), amsgrad=True)

    def initialize(self):
        self._prepare_dataset()
        self._build_net()
        self.__load_the_normal_estimation_model()
        self._build_optimizer()

    def adjust_learning_rate(self, epoch):
        if epoch <100:
            cur_lr = 1e-4
        elif epoch>=100 and epoch<300:
            cur_lr = 1e-4
        elif epoch>=300 and epoch<380:
            cur_lr = 4e-5
        else:
            cur_lr = 1e-5

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = cur_lr
        self.current_lr = cur_lr
        return cur_lr

    def set_criterion(self, criterion):
        self.criterion = criterion

    def train_one_epoch(self, epoch, round,iterations,summary_writer):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        flow2_EPEs = AverageMeter()

        # switch to train mode
        self.net.train()
        end = time.time()
        cur_lr = self.adjust_learning_rate(epoch)
        logger.info("learning rate of epoch %d: %f." % (epoch, cur_lr))
        summary_writer.add_scalar("Learning_Rate",cur_lr,epoch+1)
        for i_batch, sample_batched in enumerate(self.train_loader):
        
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)

            target_disp = sample_batched['gt_disp'].unsqueeze(1)
            target_disp = target_disp.cuda()
            target_disp = torch.autograd.Variable(target_disp, requires_grad=False)

            pseudo_disp = sample_batched['pseudo_disp'].unsqueeze(1)
            pseudo_disp = pseudo_disp.cuda()
            pseudo_disp = torch.autograd.Variable(pseudo_disp, requires_grad=False)
            
            pseudo_disp,gt_mask,left_mask = gen_disp(target_disp,pseudo_disp)

            data_time.update(time.time() - end)

            self.optimizer.zero_grad()

            '''
            output: predicted disparity: Here is a Pyramid:[1/8,1/4,1/2,1]
            loss : pyramid loss value :[1/8,1/4,1/2,1] and related weight [W1,W2,W3,W4]
            '''
            # Get the Assisted Surface Normal Here
            self.normal_net.eval()
            with torch.no_grad():
                assisted_normal = self.normal_net(left_input,right_input,False)
        
            output,conf_infer = self.net(left_input, right_input,assisted_normal,True)
    
            
            loss = self.criterion(output, target_disp, conf_infer)
            
            if type(loss) is list or type(loss) is tuple:
                loss = np.sum(loss)
            if type(output) is list or type(output) is tuple:
                
                flow2_EPE = self.epe(output[0], target_disp)
                
            else:
                flow2_EPE = self.epe(output, target_disp)

            # Record loss and EPE in the tfboard
            losses.update(loss.data.item(), left_input.size(0))
            flow2_EPEs.update(flow2_EPE.data.item(), left_input.size(0))

            summary_writer.add_scalar("total_loss",losses.val,iterations+1)
            summary_writer.add_scalar("epe_on_val",flow2_EPEs.val,iterations+1)
            
            # compute gradient and do SGD step
            loss.backward()
            self.optimizer.step()
            iterations = iterations+1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i_batch % 10 == 0:
                logger.info('this is round %d', round)
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'EPE {flow2_EPE.val:.3f} ({flow2_EPE.avg:.3f})\t'.format(
                  epoch, i_batch, self.num_batches_per_epoch, batch_time=batch_time, 
                  data_time=data_time, loss=losses, flow2_EPE=flow2_EPEs))

        return losses.avg, flow2_EPEs.avg,iterations

# Validation
    
    def validate(self,summary_writer,epoch,vis=False):
        
        batch_time = AverageMeter()
        flow2_EPEs = AverageMeter()
        P1_errors = AverageMeter()
        losses = AverageMeter()

        # switch to evaluate mode
        self.net.eval()
        end = time.time()
        inference_time = 0
        img_nums = 0

        for i, sample_batched in enumerate(self.test_loader):
            left_input = torch.autograd.Variable(sample_batched['img_left'].cuda(), requires_grad=False)
            right_input = torch.autograd.Variable(sample_batched['img_right'].cuda(), requires_grad=False)
            inputs = torch.cat((left_input, right_input), 1)
            input_var = torch.autograd.Variable(inputs, requires_grad=False)
            target_disp = sample_batched['gt_disp'].unsqueeze(1)
            target_disp = target_disp.cuda()
            target_disp = torch.autograd.Variable(target_disp, requires_grad=False)
            pseudo_disp = sample_batched['pseudo_disp'].unsqueeze(1)
            pseudo_disp = pseudo_disp.cuda()
            pseudo_disp = torch.autograd.Variable(pseudo_disp, requires_grad=False)
            
            pseudo_disp,gt_mask,left_mask = gen_disp(target_disp,pseudo_disp)
            
            with torch.no_grad():
                start_time = time.perf_counter()
                assisted_normal = self.normal_net(left_input,right_input,False)
                # Get the predicted disparity
                output,conf_infer = self.net(left_input, right_input, assisted_normal,False) 
                inference_time += time.perf_counter() - start_time
                img_nums += left_input.shape[0]
                loss = self.epe(output, target_disp)
                flow2_EPE = self.epe(output, target_disp)
                P1_error = self.p1_error(output, target_disp)

            if loss.data.item() == loss.data.item():
                losses.update(loss.data.item(), input_var.size(0))
            if flow2_EPE.data.item() == flow2_EPE.data.item():
                flow2_EPEs.update(flow2_EPE.data.item(), input_var.size(0))
            if P1_error.data.item() == P1_error.data.item():
                P1_errors.update(P1_error.data.item(), input_var.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                logger.info('Test: [{0}/{1}]\t Time {2}\t EPE {3}'
                      .format(i, len(self.test_loader), batch_time.val, flow2_EPEs.val))
                img_summary = dict()
                pred_disp =output
                pred_disp = pred_disp.squeeze(1)
                target_gt= target_disp.squeeze(1)
                img_summary['disp_error']=disp_error_img(pred_disp,target_gt)
                img_summary['left']=left_input
                img_summary['right'] =right_input
                img_summary['gt_disp']=target_gt
                img_summary['pred_disp']=pred_disp
                save_images(summary_writer,"images_info",img_summary,epoch+1)


        logger.info(' * EPE {:.3f}'.format(flow2_EPEs.avg))
        logger.info(' * P1_error {:.3f}'.format(P1_errors.avg))
        logger.info(' * avg inference time {:.3f}'.format(inference_time / img_nums))
        return flow2_EPEs.avg

    def get_model(self):
        return self.net.state_dict()