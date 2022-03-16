from __future__ import print_function
import sys
sys.path.append("../")
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


# Loss Function
def get_costheta_loss(normal_infer,normal_gt):
  normal_infer_np = normal_infer.detach().cpu().numpy()
  nomral_gt_np = normal_gt.detach().cpu().numpy()
  norm_infer_norm = np.linalg.norm(normal_infer_np,axis=1,keepdims=True)
  norm_gt_norm = np.linalg.norm(nomral_gt_np,axis=1,keepdims=True)
  summation = np.sum(normal_infer_np*nomral_gt_np,axis=1)
  costheta_val = summation*1.0/(norm_gt_norm*norm_infer_norm+1e-10)
  costheta_val = np.clip(costheta_val,-1.0,1.0)
  total_angles = np.arccos(costheta_val)/np.pi*180

  total_angles = torch.Tensor(total_angles).to(normal_infer.device)
  target_angles = torch.zeros_like(total_angles).to(normal_infer.device)
  loss = F.smooth_l1_loss(total_angles,target_angles,size_average=True)

  return loss






if __name__=="__main__":
  
  target_norml = torch.randn(1,3,320,540)
  target_norml = F.normalize(target_norml,dim=1)
  
  inference_nomral = torch.ones(1,3,320,540)
  inference_nomral = F.normalize(inference_nomral,dim=1)
  
  losses = get_costheta_loss(inference_nomral,target_norml)
  print(losses)
  pass