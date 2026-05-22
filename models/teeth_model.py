import torch
from typing import Optional, Dict, Any
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.ssm.kalman import KalmanSmoothRefinementSSM
from models.ssm.s4 import S4RefinementSSM
from models.ssm import build_ssm
from models.encoder.pointnetpp import ToothPointNetPPFeatEncoder
from models.encoder.pointnext import ToothPointNeXtFeatEncoder
from pytorch3d.transforms import *


class teeth_reship():
    def __init__(self):
        super(teeth_reship, self).__init__()
        self.edge_index = torch.tensor(
        [[1, 2], [1, 16],
        [2, 1],   [2, 3],   [2, 15],
        [3, 2],   [3, 4],   [3, 14],
        [4, 3],   [4, 5],   [4, 13],
        [5, 4],   [5, 6],   [5, 12],
        [6, 5],   [6, 7],   [6, 11],
        [7, 6],   [7, 8],   [7, 10],
        [8, 7],   [8, 9],   [8, 9],
        [9, 8],   [9, 10],  [9, 8],
        [10, 9],  [10, 11], [10, 7],
        [11, 10], [11, 12], [11, 6],
        [12, 11], [12, 13], [12, 5],
        [13, 12], [13, 14], [13, 4],
        [14, 13], [14, 15], [14, 3],
        [15, 14], [15, 16], [15, 2],
        [16, 15], [16, 1],
        [17, 18],  [17, 32],
        [18, 17],  [18, 19], [18, 31],
        [19, 18],  [19, 20], [19, 30],
        [20, 19],  [20, 21], [20, 29],
        [21, 20],  [21, 22], [21, 28],
        [22, 21],  [22, 23], [22, 27],
        [23, 22],  [23, 24], [23, 26],
        [24, 23],  [24, 25], [24, 25],
        [25, 24],  [25, 26], [25, 24],
        [26, 25],  [26, 27], [26, 23],
        [27, 26],  [27, 28], [27, 22],
        [28, 27],  [28, 29], [28, 21],
        [29, 28],  [29, 30], [29, 20],
        [30, 29],  [30, 31], [30, 19],
        [31, 30],  [31, 32], [31, 18],
        [32, 31],  [32, 17]])


class teeth_arangement_model(nn.Module):
    def __init__(self, in_channel=3):
        super(teeth_arangement_model, self).__init__()
        from models.encoder.swin import mae_vit_base_patch16
        self.local_fea = mae_vit_base_patch16()

    def forward(self, teeth_points, centerp):
        # teeth_points: (B,T,N,3)
        # centerp:      (B,T,1,3)
        B, T, N, C = teeth_points.shape

        # (B,T,3,N)
        teeth_points = (teeth_points - centerp).permute(0, 1, 3, 2)
        # (B,T,3,N)
        cenv = centerp.permute(0, 1, 3, 2).expand(-1, -1, -1, N)

        # (B,T,6,N)
        x = torch.cat([teeth_points, cenv], dim=2)

        # vitnet이 이제 배치 입력(B,T,6,N)을 지원해야 함
        dofs, transv = self.local_fea(x)   # (B,T,4), (B,T,3)
        return dofs, transv


class STTAlignVitFeatEncoder(nn.Module):
    """
    STTAlign ViTNet을 'feature encoder'로만 사용:
      (B,T,P,3) -> (B,T,C)
    """
    def __init__(self, out_dim=256, freeze=False):
        super().__init__()
        from models.encoder.swin import mae_vit_base_patch16
        self.vit = mae_vit_base_patch16()
        self.out_dim = int(out_dim)

        # vit의 embed_dim은 256으로 구성돼 있으니, 다르면 proj
        embed_dim = 256
        self.proj = nn.Identity() if embed_dim == self.out_dim else nn.Linear(embed_dim, self.out_dim)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            for p in self.proj.parameters():
                p.requires_grad = False

    def forward(self, points_in, mask=None):
        """
        points_in: (B,T,P,3)
        mask: (B,T) bool
        """
        B, T, P, _ = points_in.shape
        centerp = points_in.mean(dim=2, keepdim=True)                  # (B,T,1,3)

        rel = (points_in - centerp).permute(0, 1, 3, 2).contiguous()   # (B,T,3,P)
        cen = centerp.permute(0, 1, 3, 2).expand(-1, -1, -1, P)        # (B,T,3,P)
        x6 = torch.cat([rel, cen], dim=2)                              # (B,T,6,P)

        # ✅ vitnet.py에서 return_feat 옵션 추가(1번 패치 필요)
        feat = self.vit.forward_encoder(x6, return_feat=True)          # (B,T,256)
        feat = self.proj(feat)                                         # (B,T,out_dim)

        if mask is not None:
            feat = feat * mask.unsqueeze(-1).float()
        return feat


class teeth_arangement_kalman_ssm_model(nn.Module):
    """
    STTAlign encoder -> KalmanSmoothRefinementSSM -> (quat, trans)
    기존 학습 코드의 호출 형태를 깨지 않게 forward를 (teeth_points, centerp)로 유지.
    """
    def __init__(
        self,
        use_tooth_id_emb=True,
        tooth_id_offset=0,
        freeze_encoder=False,
        encoder_type: str = "vit",
        encoder_out_dim: int = 256,
        pointnet_use_center: bool = True,
        ssm_cfg: Optional[Dict[str, Any]] = None,
        pointnext_k: int = 16,
        pointnext_stage_points=(128, 32, None),
        pointnext_stage_dims=(64, 128, 256),
        pointnext_expansion: int = 4,
    ):
        super().__init__()
        self.tooth_id_offset = int(tooth_id_offset)

        enc_type = str(encoder_type).lower()
        if enc_type in ["pointnet++", "pointnetpp", "pn++", "pointnet2", "pointnet2ssg", "pn2"]:
            enc = ToothPointNetPPFeatEncoder(
                out_dim=encoder_out_dim,
                use_center=pointnet_use_center,
                freeze=freeze_encoder,
            )
        elif enc_type in ["swin", "swin-t", "vit"]:
            enc = STTAlignVitFeatEncoder(out_dim=encoder_out_dim, freeze=freeze_encoder)
        elif enc_type in ["pointnext", "pnx", "pointneXt"]:
            enc = ToothPointNeXtFeatEncoder(
                out_dim=encoder_out_dim,
                use_center=pointnet_use_center,
                freeze=freeze_encoder,
                k=pointnext_k,
                stage_points=pointnext_stage_points,
                stage_dims=pointnext_stage_dims,
                expansion=pointnext_expansion,
            )

        self.ssm = build_ssm(
            ssm_cfg,
            encoder=enc,
            hidden_dim=encoder_out_dim,
            use_tooth_id_emb=use_tooth_id_emb,
        )

    @staticmethod
    def _canonicalize_quat(q):
        # quaternion sign ambiguity 제거 (w>=0)
        sign = torch.where(q[..., :1] < 0, -1.0, 1.0)
        return q * sign

    def forward(self, teeth_points, centerp=None, t=None, return_intermediates=False):
        """
        teeth_points: (B,T,P,3)
        centerp: (B,T,1,3) (호환용; 내부에선 points mean 사용)
        t: (B,1) 없으면 1로 둠
        return_intermediates=True면 aux(T_steps 등)도 반환
        """
        B, T, _, _ = teeth_points.shape
        device = teeth_points.device

        mask = torch.ones((B, T), device=device, dtype=torch.bool)

        # upper: 1..16, lower: 17..32 쓰고 싶으면 tooth_id_offset=16로
        tooth_id = torch.arange(1, T + 1, device=device).view(1, T).repeat(B, 1)
        tooth_id = tooth_id + self.tooth_id_offset

        if t is None:
            t = torch.ones((B, 1), device=device, dtype=teeth_points.dtype)

        if not return_intermediates:
            T_pred, deltas, h_last = self.ssm(teeth_points, tooth_id, mask, t, return_intermediates=False)
        else:
            T_pred, deltas, h_last, aux = self.ssm(teeth_points, tooth_id, mask, t, return_intermediates=True)

        R = T_pred[..., :3, :3]     # (B,T,3,3)
        trans = T_pred[..., 3, :3]  # (B,T,3)

        quat = matrix_to_quaternion(R)          # (B,T,4) (w,x,y,z)
        quat = self._canonicalize_quat(quat)

        if not return_intermediates:
            return quat, trans
        return quat, trans, aux


