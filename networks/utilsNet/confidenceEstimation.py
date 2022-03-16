import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../..")
from networks.utilsNet.devtools import convt_bn_relu
from networks.utilsNet.Disparity_warper import disp_warp

def conv1x1(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU(inplace=True))

class DisparityConfEstimation(nn.Module):
    def __init__(self,input_channels,hidden_layer=32):
        super(DisparityConfEstimation,self).__init__()
        self.input_channels = input_channels
        self.hidden_layer = hidden_layer
        ''' 2 Branch + Concated + Little U-Net'''
        self.feature_encode = nn.Sequential(
            nn.Conv2d(self.input_channels,hidden_layer//2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer//2),
            nn.ReLU(True)
            )
        self.disp_encode = nn.Sequential(
            nn.Conv2d(3+1,hidden_layer//2,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer//2),
            nn.ReLU(True)
        )
        # Smaller U-Net?
        # Downsample Phase
        # 1/2
        self.conv1_a = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer,out_channels=hidden_layer,kernel_size=3,stride=2,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(True)
        )
        self.conv1_b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer,out_channels=hidden_layer,kernel_size=3,stride=1,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer),
            nn.ReLU(True)
        )
        # 1/4
        self.conv2_a = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer,out_channels=hidden_layer*2,kernel_size=3,stride=2,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer*2),
            nn.ReLU(True)
        )
        self.conv2_b = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layer*2,out_channels=hidden_layer*2,kernel_size=3,stride=1,
            padding=1,bias=False),
            nn.BatchNorm2d(hidden_layer*2),
            nn.ReLU(True)
        )
        # Upsample Phase
        self.upconv_1 = convt_bn_relu(hidden_layer*2,hidden_layer,3,stride=2,padding=1,output_padding=1)
        self.upconv_2 = convt_bn_relu(hidden_layer*2,hidden_layer,3,stride=2,padding=1,output_padding=1)
        
        # Final Prediction: Range from 0-1
        self.confidence_estimation=nn.Sequential(
            nn.Conv2d(hidden_layer,1,3,1,1,bias=False),
            nn.Sigmoid()
        )
        
    def forward(self,pred_disp,img_left,img_right,feature):
        '''
        Give a coarse disparity confidence result for propagation.
        Inputs: Current Feature + Pred_disparity + Warped Error(Option1)
        '''
        scale = img_left.size(-2)//pred_disp.size(-2)
        cur_left = F.interpolate(img_left,scale_factor=1.0/scale,mode='bilinear',align_corners=False)
        cur_right = F.interpolate(img_right,scale_factor=1.0/scale,mode='bilinear',align_corners=False)
        assert cur_left.size(-2)==pred_disp.size(-2)
        assert cur_left.size(-1)==pred_disp.size(-1)
        disp_ = pred_disp / scale
        warped_right = disp_warp(cur_right,disp_)[0]
        warped_error = warped_right - cur_left
        feature_part = self.feature_encode(feature)
        disp_part = self.disp_encode(torch.cat((disp_,warped_error),dim=1))
        

        # 1/2
        feat1_a = self.conv1_a(torch.cat((feature_part,disp_part),dim=1))
        feat1_b = self.conv1_b(feat1_a)
        #1/4
        feat2_a = self.conv2_a(feat1_b)
        feat2_b = self.conv2_b(feat2_a)
        # Upsampled to 1/2
        upconv2 = self.upconv_1(feat2_b) # 32
        # Upsampled to Full Size
        upconv1 = self.upconv_2(torch.cat((upconv2,feat1_b),dim=1))

        confidence = self.confidence_estimation(upconv1)

        return confidence
        



if __name__=="__main__":
    left_image = torch.randn(2,3,320,640)
    right_image = torch.randn(2,3,320,640)
    left_feature = torch.randn(2,32,160,320)
    pred_disparity = torch.abs(torch.rand(2,1,160,320))
    conf_branch = DisparityConfEstimation(input_channels=32,hidden_layer=32)

    confidnce = conf_branch(pred_disparity,left_image,right_image,left_feature)

    print(confidnce.shape)