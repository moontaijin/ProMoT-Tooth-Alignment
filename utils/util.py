import numpy as np
import torch
import torch.nn.functional as F

def cal_loss(pred, gold, smoothing=True):
    gold = gold.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


from typing import Dict, List, Optional, Tuple, Callable

import torch
import torch.nn as nn
from pytorch3d.transforms import *

class Tooth_Assembler(nn.Module):
    def __init__(self):
        super(Tooth_Assembler, self).__init__()

    def forward(
        self,
        pred: torch.Tensor,     # (B,T,P,3) or (T,P,3)
        cenp: torch.Tensor,     # (B,T,1,3) or (T,1,3)
        dofs: torch.Tensor,     # (B,T,4)   or (T,4) quaternion
        ptrans: torch.Tensor,   # (B,T,3)   or (T,3)
        device: torch.device,
    ) -> torch.Tensor:

        # ---- handle both batched and unbatched ----
        if pred.dim() == 4:
            B, T, P, _ = pred.shape
            pred_ = pred.reshape(B * T, P, 3)
            cenp_ = cenp.reshape(B * T, 1, 3)
            dofs_ = dofs.reshape(B * T, 4)
            ptrans_ = ptrans.reshape(B * T, 3)
        else:
            T, P, _ = pred.shape
            pred_ = pred
            cenp_ = cenp
            dofs_ = dofs
            ptrans_ = ptrans

        # quaternion_to_matrix supports batch: (N,4)->(N,3,3)
        R = quaternion_to_matrix(dofs_)  # (BT,3,3) or (T,3,3)

        # assemble: (p - c)R + t + c
        pts = pred_ - cenp_
        pts = torch.bmm(pts, R)
        out = pts + ptrans_.unsqueeze(1) + cenp_

        if pred.dim() == 4:
            out = out.reshape(B, T, P, 3)

        return out
