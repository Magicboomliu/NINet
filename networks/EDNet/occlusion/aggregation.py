import sys
sys.path.append("../../..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.EDNet.submodule import *

class InnerAggregation(nn.Module):
    def __init__(self,num_blocks,max_disp):
        super(InnerAggregation,self).__init__()
        self.num_blocks = num_blocks
        branch = nn.ModuleList()
        for _ in range(self.num_blocks):
            branch.append(DeconBlock(max_disp,max_disp,3,1))

        self.aggregation = nn.Sequential(*branch)
    def forward(self,x):
        x = self.aggregation(x)
        return x

# Cross Scale Aggregation
class CrossAggregation14(nn.Module):
    def __init__(self,num_blocks,max_disp_list,deform=False):
        super(CrossAggregation14,self).__init__()
        self.num_blocks = num_blocks
        self.max_disp_list = max_disp_list
        self.deform = deform
        self.aggregation = nn.Sequential(
            nn.Conv2d(max_disp_list[0],max_disp_list[1],kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(max_disp_list[1]),
            nn.LeakyReLU(0.2,inplace=True)
            )
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        if self.deform:
            self.aggregation_0 = InnerAggregation(num_blocks=self.num_blocks,max_disp = max_disp_list[1])
        
    def forward(self,cost_volume_list):
        x = self.aggregation(cost_volume_list[0])
        cost_volume = cost_volume_list[1] + x
        cost_volume = self.relu(cost_volume)

        if self.deform:
            cost_volume = self.aggregation_0(cost_volume)
        
        return cost_volume


class CrossAggregationCostVolume(nn.Module):
    def __init__(self,num_blocks,max_disp_list,deform=False):
        super(CrossAggregationCostVolume,self).__init__()
        self.num_blocks = num_blocks
        self.max_disp_list = max_disp_list
        self.deform = False
        self.inner_agg0= InnerAggregation(num_blocks=self.num_blocks,max_disp=max_disp_list[0])
        self.inner_agg1 = InnerAggregation(num_blocks=self.num_blocks,max_disp=max_disp_list[1])
        self.cross_agg0 = CrossAggregation14(num_blocks=self.num_blocks,max_disp_list=max_disp_list,deform = self.deform)

    def forward(self,cost_volume0,cost_volume1):

        assert cost_volume0.shape[1] == self.max_disp_list[0]
        assert cost_volume1.shape[1] == self.max_disp_list[1]
        
        # Get aggregated cost volume
        cost_volume0 = self.inner_agg0(cost_volume0)
        print(cost_volume0.shape)
        cost_volume1 = self.inner_agg1(cost_volume1)
     
        cost_volume = self.cross_agg0([cost_volume0,cost_volume1])
      

        return cost_volume



if __name__=="__main__":

    cost_volume_18 = torch.randn(1,24,160,320).cuda()
    cost_volume_14 = torch.randn(1,48,320,640).cuda()

    cost_volume_aggregation = CrossAggregationCostVolume(num_blocks=2,max_disp_list=[48,24],deform=True).cuda()

    cost_volume = cost_volume_aggregation(cost_volume_14,cost_volume_18)
    print(cost_volume.shape)
