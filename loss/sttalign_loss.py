import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

class GeometricReconstructionLoss(nn.Module):
    def __init__(self):
        super(GeometricReconstructionLoss, self).__init__()

    def forward(self, X_v, target_X_v, weights, device: torch.device):
        # X_v, target_X_v: (B,T,P,3)
        B, T, P, C = X_v.shape

        loss_bt = torch.zeros((B, T), device=device, dtype=X_v.dtype)

        for t in range(T):
            pred = X_v[:, t, :, :]        # (B,P,3)
            tag  = target_X_v[:, t, :, :] # (B,P,3)

            # dist: (B,P,P)
            dist = torch.cdist(pred, tag, p=2)

            # for each pred point -> nearest tag point
            idx_pred = dist.argmin(dim=2)  # (B,P)
            # for each tag point -> nearest pred point
            idx_tag  = dist.argmin(dim=1)  # (B,P)

            # gather
            tagp = torch.gather(tag, 1, idx_pred.unsqueeze(-1).expand(-1, -1, C))   # (B,P,3)
            predd = torch.gather(pred, 1, idx_tag.unsqueeze(-1).expand(-1, -1, C))  # (B,P,3)

            tmp1 = F.smooth_l1_loss(pred, tagp, reduction="none").mean(dim=(1, 2))  # (B,)
            tmp2 = F.smooth_l1_loss(tag, predd, reduction="none").mean(dim=(1, 2))  # (B,)

            loss_bt[:, t] = tmp1 + tmp2

        # 원본과 동일하게 weights 적용 후 /B
        loss = torch.sum(loss_bt * weights) / B

        # centroid losses (원본 로직 유지)
        prec = torch.mean(X_v, dim=2)          # (B,T,3)
        tarc = torch.mean(target_X_v, dim=2)   # (B,T,3)
        lossc = F.smooth_l1_loss(prec, tarc, reduction="sum") / (prec.shape[0] * 3)

        return loss, lossc

def symmetric_loss(X_v):
    nums = X_v.shape[1]//2
    rg = X_v[:, 0:nums, :, :]
    lg = X_v[:, nums:, :, :]
    lg = torch.flip(lg, dims=[1])
    rgc = torch.abs(torch.mean(rg, dim=2))
    lgc =  torch.abs(torch.mean(lg, dim=2))
    lossc = F.smooth_l1_loss(rgc[:, :, 0:2], lgc[:, :, 0:2], reduction="sum") / (rgc.shape[0] * 2)
    return lossc

def nearnest_index(pred_, tag_):
    pred = pred_ -torch.mean(pred_, dim=0)
    tag = tag_ -torch.mean(pred_, dim=0)
    pred = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
    tag = tag.unsqueeze(0).repeat(pred.shape[0], 1, 1)
    diff = torch.sqrt(torch.sum(torch.pow(torch.sub(pred, tag), 2), dim=-1))
    min_index = torch.argmin(diff, dim=1)
    minv = torch.min(diff, dim=1)[0]
    return min_index, minv

def nearnest_value(pred_, tag_):
    pred = pred_
    tag = tag_
    pred = pred.unsqueeze(1).repeat(1, tag.shape[0], 1)
    tag = tag.unsqueeze(0).repeat(pred.shape[0], 1, 1)
    diff = torch.sqrt(torch.sum(torch.pow(torch.sub(pred, tag), 2), dim=-1))
    min_index = torch.argmin(diff, dim=1)
    nearnestp = tag_[min_index]
    minv = pred_ - nearnestp
    return min_index, minv

def spatial_Relation_Loss(pred, target, weights, device):
    loss = torch.zeros([pred.shape[0], pred.shape[1]]).to(device)
    for bn in range(pred.shape[0]):
        for idx in range(pred.shape[1] -1):
            pred1 = pred[bn, idx, :, :]
            pred2 = pred[bn, idx+1, :, :]
            tag1 = target[bn, idx, :, :]
            tag2 = target[bn, idx+1, :, :]
            min_index1, _ = nearnest_index(pred1, tag1)
            min_index2, _ = nearnest_index(pred2, tag2)
            tag1_ = tag1[min_index1]
            tag2_ = tag2[min_index2]
            min_indexpp, minvp1 = nearnest_value(pred1, pred2)
            min_indextp, minvpt1= nearnest_value(tag1_, tag2)
            min_indexpp, minvp2 = nearnest_value(pred2, pred1)
            min_indextp, minvpt2= nearnest_value(tag2_, tag1)
            minvp_mask1 = minvp1
            minvpt_mask1 = minvpt1
            minvp_mask2 = minvp2
            minvpt_mask2 = minvpt2
            lossc1 = F.smooth_l1_loss(minvp_mask1, minvpt_mask1, reduction="mean")
            lossc2 = F.smooth_l1_loss(minvp_mask2, minvpt_mask2, reduction="mean")
            loss[bn, idx] = (lossc1 + lossc2)*0.5
    loss = torch.sum(loss) / weights.shape[0]
    return  loss

def interdental_occlusion_loss(assm_a, lab_a, assm_b, lab_b, device: torch.device):
    cox_coincide_loss_save = torch.zeros(size=[lab_a.shape[0], lab_a.shape[1]], device=device)
    sigma_groove_loss_save = torch.zeros(size=[lab_a.shape[0], lab_a.shape[1]], device=device)
    sigma_incisors_loss_save = torch.zeros(size=[lab_a.shape[0], lab_a.shape[1]], device=device)
    for bn in range(lab_a.shape[0]):
        for idx in range(lab_a.shape[1]):
            pot_a = lab_a[bn][idx]
            pot_a_z = pot_a.clone()[:, 2:]
            pot_a = pot_a[:, :2]
            tt = lab_a.shape[1] - 1
            pot_b = lab_b[bn][tt - idx]
            pot_b_alone = pot_b
            pot_b_z_alone = pot_b_alone.clone()[:, 2:]
            if idx == 0:
                pot_b = torch.cat([pot_b, lab_b[bn][tt - (idx + 1)]], dim=0)
            elif idx == lab_a.shape[1] - 1:
                pot_b = torch.cat([lab_b[bn][tt - (idx - 1)], pot_b], dim=0)
            else:
                pot_b = torch.cat([lab_b[bn][tt - (idx - 1)], pot_b], dim=0)
                pot_b = torch.cat([pot_b, lab_b[bn][tt - (idx + 1)]], dim=0)
            pot_b = pot_b[:, :2]
            
            diff = torch.cdist(pot_a, pot_b, p=2)   # (Pa,Pb)
            minvx = diff.argmin(dim=1)             # (Pa,)
            min_dist = diff.min(dim=1).values      # (Pa,)
            is_have_cos_label = (min_dist < 0.7)   # (Pa,) bool tensor

            as_pot_a = assm_a[bn][idx]
            as_pot_a_z = as_pot_a.clone()[:, 2:]
            as_pot_a = as_pot_a[:, :2]
            tt = assm_a.shape[1] - 1
            as_pot_b = assm_b[bn][tt - idx]
            as_pot_b_alone = as_pot_b
            as_pot_b_z_alone = as_pot_b_alone.clone()[:, 2:]
            if idx == 0:
                as_pot_b = torch.cat([as_pot_b, assm_b[bn][tt - (idx + 1)]], dim=0)
            elif idx == assm_a.shape[1] - 1:
                as_pot_b = torch.cat([assm_b[bn][tt - (idx - 1)], as_pot_b], dim=0)
            else:
                as_pot_b = torch.cat([assm_b[bn][tt - (idx - 1)], as_pot_b], dim=0)
                as_pot_b = torch.cat([as_pot_b, assm_b[bn][tt - (idx + 1)]], dim=0)
            as_pot_b_z = as_pot_b.clone()[:, 2:]
            as_pot_b = as_pot_b[:, :2]
            
            as_diff = torch.cdist(as_pot_a, as_pot_b, p=2)     # (Pa,Pb)
            as_minvx = as_diff.argmin(dim=1)                  # (Pa,)
            as_min_dist = as_diff.min(dim=1).values           # (Pa,)
            is_have_cos_assem = (as_min_dist < 0.7)           # (Pa,) bool tensor

            # mismatch count
            not_coincide_num = (is_have_cos_assem != is_have_cos_label).sum().item()

            # zvalue_dis: only where label says contact
            if is_have_cos_label.any():
                sel = is_have_cos_label
                zvalue_dis = as_pot_a_z[sel] - as_pot_b_z[as_minvx[sel]]
            else:
                zvalue_dis = []

            cox_coinc_lo = not_coincide_num
            cox_coincide_loss_save[bn][idx] = cox_coinc_lo
            if idx >= 5 and idx <= 10:
                canine_p1_ass = torch.mean(as_pot_a)
                a_xft2 = torch.argmin(as_pot_a_z, dim=0)
                canine_p2_ass = as_pot_a[a_xft2]
                canine_p3_ass = torch.mean(as_pot_b_alone)
                a_xft4 = torch.argmax(as_pot_b_z_alone, dim=0)
                canine_p4_ass = as_pot_b_alone[a_xft4]
                canine_p1_lab = torch.mean(pot_a)
                xft2 = torch.argmin(pot_a_z, dim=0)
                canine_p2_lab = pot_a[xft2]
                canine_p3_lab = torch.mean(pot_b_alone)
                xft4 = torch.argmax(pot_b_z_alone, dim=0)
                canine_p4_lab = pot_b_alone[xft4]
                gap_1 = torch.norm(canine_p1_ass - canine_p1_lab)
                gap_2 = torch.norm(canine_p2_ass - canine_p2_lab)
                gap_3 = torch.norm(canine_p3_ass - canine_p3_lab)
                gap_4 = torch.norm(canine_p4_ass - canine_p4_lab)
                sigma_z_loss = gap_1 + gap_2 + gap_3 + gap_4
                sigma_incisors_loss_save[bn][idx] = sigma_z_loss
            else:
                if isinstance(zvalue_dis, torch.Tensor) and zvalue_dis.numel() > 0:
                    sigma_z_loss = torch.var(zvalue_dis, unbiased=False)

    cox_coincide_loss = torch.mean(torch.mean(cox_coincide_loss_save, dim=1))
    sigma_groove_loss = torch.mean(torch.mean(sigma_groove_loss_save, dim=1))
    sigma_incisors_loss = torch.mean(torch.mean(sigma_incisors_loss_save, dim=1))
    return cox_coincide_loss, sigma_groove_loss, sigma_incisors_loss
