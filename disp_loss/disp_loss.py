import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import SmoothL1Loss

#
# Disparity Loss
#
def EPE(output, target, maxdisp):
    mask = (target < maxdisp) & (0 < target)
    mask.detach()
    # d_diff = output[mask] - target[mask]
    # EPE_map = torch.abs(d_diff)
    # EPE_mean = torch.sum(EPE_map)/N
    
    output = output.squeeze(1)
    criterion = nn.MSELoss()
    
    # EPE_mean = F.smooth_l1_loss(output[mask], target[mask], size_average=True)
    EPE_mean = criterion(output[mask], target[mask])
    # print(EPE_mean)
    return EPE_mean

class multiscaleLoss(nn.modules.loss._Loss):
    def __init__(self, maxdisp):
        super(multiscaleLoss, self).__init__()
        
        self.maxdisp = maxdisp
        
    def forward(self, outputs, target):
        if type(outputs) not in [tuple, list]:
                outputs = [outputs]
        
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]
        # weights = [0.5, 0.3, 0.2]
        loss = 0
        
        for output, weight in zip(outputs, weights):
            loss += weight * self.one_scale(output, target, self.maxdisp)
        return loss


    def one_scale(self, output, target, maxdisp):


        B, _, h, w = output.size()
        
        target_scaled = nn.functional.adaptive_max_pool2d(target, (h, w))
        
        return EPE(output, target_scaled, maxdisp)

def CEloss(disp_gt, max_disp, gt_distribute, pred_distribute):
    mask = (disp_gt > 0) & (disp_gt < max_disp)

    pred_distribute = torch.log(pred_distribute + 1e-8)

    ce_loss = torch.sum(-gt_distribute * pred_distribute, dim=1)
    ce_loss = torch.mean(ce_loss[mask])
    return ce_loss

def disp2distribute(disp_gt, max_disp, b=2):
    disp_gt = disp_gt.unsqueeze(1)
    disp_range = torch.arange(0, max_disp).view(1, -1, 1, 1).float().cuda()
    gt_distribute = torch.exp(-torch.abs(disp_range - disp_gt) / b)
    gt_distribute = gt_distribute / (torch.sum(gt_distribute, dim=1, keepdim=True) + 1e-8)
    return gt_distribute


class smoothL1Loss(nn.modules.loss._Loss):
    def __init__(self):
        super(smoothL1Loss, self).__init__()
        
    def forward(self, outputs, target):
        # if type(outputs) not in [tuple, list]:
        #         outputs = [outputs]

        
        if len(outputs) == 5:
            weights = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]
        elif len(outputs) == 4:
            weights = [0.5, 0.5, 0.7, 1.0]
        elif len(outputs) == 3:
            weights = [0.5, 0.5, 1.0]
        elif len(outputs) == 8:
            weights = [0.5, 0.5, 1.0, 1.0]
            

        else:
            weights = 1
        # weights = [0.7, 1.0]
        loss = 0
        mask = (target < 80) & (0 < target)
        # mask = target != float('inf')

    
        # criterion = nn.MSELoss()
        if isinstance(weights, (list, tuple)):
            if len(outputs) == 8:
                outputs2 = outputs[4:]
                outputs1 = outputs[:4]
                for output, weight in zip(outputs1, weights):
                    output = torch.squeeze(output, 1)
                    loss += weight * F.smooth_l1_loss(output[mask], target[mask], size_average=True)

                target_distribute = disp2distribute(target, 48, b=2)
                for output, weight in zip(outputs2, weights):
                    loss += weight * CEloss(target, 48, target_distribute, output)
            else:
                for output, weight in zip(outputs, weights):
                    # import pdb; pdb.set_trace()
                    output = torch.squeeze(output, 1)

                    

                    loss += weight * F.smooth_l1_loss(output[mask], target[mask], size_average=True)
        else:
            loss = F.smooth_l1_loss(outputs[0][mask], target[mask], size_average=True)
            # loss += weight * criterion(output[mask], target[mask], size_average=True)
        # print(loss)
        return loss

class pasmLoss(nn.modules.loss._Loss):
    def __init__(self):
        super(pasmLoss, self).__init__()
        
    def forward(self, outputs, target):

        if len(outputs) == 1:
            loss = 0
            mask = target != float('inf')
            loss += F.smooth_l1_loss(outputs[mask], target[mask], size_average=True)
        else:
            loss = 0
            # mask = (target < 36) & (0 < target)
            mask = target != float('inf')

        
            output_disps, att, att_cycle, valid_mask = outputs
            loss += F.smooth_l1_loss(output_disps[mask], target[mask], size_average=True)

            loss_PAM_C = loss_pam_cycle(att_cycle, valid_mask).mean()
            loss_PAM_S = loss_pam_smoothness(att).mean()
            loss_PAM = loss_PAM_S + loss_PAM_C
            
            loss += loss_PAM
        return loss


def loss_pam_cycle(att_cycle, valid_mask):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att_cycle[0][0].device)

    for idx_scale in range(len(att_cycle)):
        b, c, h, w = valid_mask[idx_scale][0].shape
        I = torch.eye(w, w).repeat(b, h, 1, 1).to(att_cycle[0][0].device)

        att_left2right2left = att_cycle[idx_scale][0]
        att_right2left2right = att_cycle[idx_scale][1]
        valid_mask_left = valid_mask[idx_scale][0]
        valid_mask_right = valid_mask[idx_scale][1]

        loss_scale = L1Loss(att_left2right2left * valid_mask_left.permute(0, 2, 3, 1), I * valid_mask_left.permute(0, 2, 3, 1)) + \
                     L1Loss(att_right2left2right * valid_mask_right.permute(0, 2, 3, 1), I * valid_mask_right.permute(0, 2, 3, 1))

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def loss_pam_smoothness(att):
    weight = [0.2, 0.3, 0.5]
    loss = torch.zeros(1).to(att[0][0].device)

    for idx_scale in range(len(att)):
        att_right2left = att[idx_scale][0]
        att_left2right = att[idx_scale][1]

        loss_scale = L1Loss(att_right2left[:, :-1, :, :], att_right2left[:, 1:, :, :]) + \
                     L1Loss(att_left2right[:, :-1, :, :], att_left2right[:, 1:, :, :]) + \
                     L1Loss(att_right2left[:, :, :-1, :-1], att_right2left[:, :, 1:, 1:]) + \
                     L1Loss(att_left2right[:, :, :-1, :-1], att_left2right[:, :, 1:, 1:])

        loss = loss + weight[idx_scale] * loss_scale

    return loss


def L1Loss(input, target):
    return (input - target).abs().mean()
