import  torch
import torch.nn as nn
import torch.nn.functional as F

def get_affinity_matrix(feature):
    b,c,h,w = feature.shape
    # HWxC
    feature1 = feature.view(b,h*w,c)
    # CxHW
    feature2 = feature.view(b,c,h*w)

    feature =  torch.bmm(feature1,feature2)
    
    feature_pos = torch.exp(feature/2)

    # Row Normalization
    feature = F.normalize(feature_pos,dim=2)
    return feature


if __name__=="__main__":
    # Row is 2 and Column is 3
    feature = torch.randn(1,1,2,3).cuda()
    feature = get_affinity_matrix(feature)
    print(feature.shape)

