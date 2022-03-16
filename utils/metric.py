import torch
import torch.nn.functional as F

def D1_metric(D_pred, D_gt):
    E = torch.abs(D_pred - D_gt)
    E_mask = (E > 3) & (E / D_gt.abs() > 0.05)
    return torch.mean(E_mask.float())

def P1_metric(D_pred, D_gt):
    E = torch.abs(D_pred - D_gt)
    E_mask = (E > 1)
    return torch.mean(E_mask.float())


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype('float'))
    else:
        mean = torch.mean(err_mask.float())

    return mean