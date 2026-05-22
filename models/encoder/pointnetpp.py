import torch
import torch.nn as nn
import torch.nn.functional as F

def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    src: (B,N,3), dst: (B,M,3) -> dist2: (B,N,M)
    """
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist2 = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist2 += torch.sum(src ** 2, dim=-1, keepdim=True)
    dist2 += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist2


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    points: (B,N,C)
    idx: (B,S) or (B,S,K)
    return: (B,S,C) or (B,S,K,C)
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    xyz: (B,N,3) -> centroids idx: (B,npoint)
    Pure PyTorch FPS (O(B*N*npoint)).
    """
    device = xyz.device
    B, N, _ = xyz.shape
    npoint = int(min(npoint, N))
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), device=device, dtype=torch.long)
    batch_indices = torch.arange(B, device=device, dtype=torch.long)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def _knn_point(nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """
    nsample: int
    xyz: (B,N,3), new_xyz: (B,S,3)
    return idx: (B,S,nsample) of kNN in xyz for each new_xyz
    """
    nsample = int(min(nsample, xyz.shape[1]))
    dist2 = _square_distance(new_xyz, xyz)  # (B,S,N)
    idx = dist2.topk(k=nsample, dim=-1, largest=False, sorted=False).indices
    return idx


class PointNetSetAbstractionKNN(nn.Module):
    """
    Set Abstraction layer: FPS (optional) + kNN grouping + shared MLP + maxpool.
    """
    def __init__(self, npoint, nsample, in_channel, mlp, use_xyz=True, group_all=False):
        super().__init__()
        self.npoint = None if npoint is None else int(npoint)
        self.nsample = None if nsample is None else int(nsample)
        self.use_xyz = bool(use_xyz)
        self.group_all = bool(group_all)

        last_channel = int(in_channel) + (3 if self.use_xyz else 0)
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, int(out_channel), 1, bias=False))
            self.mlp_bns.append(nn.BatchNorm2d(int(out_channel)))
            last_channel = int(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, xyz, points=None):
        """
        xyz: (B,N,3)
        points: (B,N,D) or None
        return:
          new_xyz: (B,S,3) or None
          new_points: (B,S,C)
        """
        B, N, _ = xyz.shape

        if self.group_all or self.npoint is None:
            # group all points into one
            new_xyz = xyz.mean(dim=1, keepdim=True)  # (B,1,3)
            grouped_xyz = xyz.view(B, 1, N, 3) - new_xyz.view(B, 1, 1, 3)  # (B,1,N,3)
            if points is not None:
                grouped_points = points.view(B, 1, N, -1)  # (B,1,N,D)
                if self.use_xyz:
                    new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B,1,N,3+D)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz if self.use_xyz else torch.zeros(B, 1, N, 0, device=xyz.device, dtype=xyz.dtype)
        else:
            # FPS
            fps_idx = _farthest_point_sample(xyz, self.npoint)  # (B,S)
            new_xyz = _index_points(xyz, fps_idx)               # (B,S,3)
            # kNN group
            idx = _knn_point(self.nsample, xyz, new_xyz)        # (B,S,K)
            grouped_xyz = _index_points(xyz, idx)               # (B,S,K,3)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)    # (B,S,K,3)

            if points is not None:
                grouped_points = _index_points(points, idx)     # (B,S,K,D)
                if self.use_xyz:
                    new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B,S,K,3+D)
                else:
                    new_points = grouped_points
            else:
                new_points = grouped_xyz if self.use_xyz else torch.zeros(B, self.npoint, self.nsample, 0, device=xyz.device, dtype=xyz.dtype)

        # (B,S,K,Cin) -> (B,Cin,S,K)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()

        for conv, bn in zip(self.mlp_convs, self.mlp_bns):
            new_points = self.act(bn(conv(new_points)))

        # maxpool over K
        new_points = torch.max(new_points, dim=-1).values  # (B,C,S)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B,S,C)
        return new_xyz, new_points


class ToothPointNetPPFeatEncoder(nn.Module):
    """
    PointNet++ per-tooth encoder producing a single feature per tooth.

    Input:
      points_in: (B,T,P,3)
      mask:      (B,T) bool (optional)
    Output:
      feat: (B,T,C)
    """
    def __init__(
        self,
        out_dim: int = 256,
        use_center: bool = True,
        freeze: bool = False,
        npoint1: int = 128,
        nsample1: int = 32,
        npoint2: int = 32,
        nsample2: int = 64,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.use_center = bool(use_center)

        # SA hierarchy (typical PointNet++ classification style)
        # xyz is centered (rel coords). optional per-point feature = absolute center (3D).
        in_ch0 = 3 if self.use_center else 0
        self.sa1 = PointNetSetAbstractionKNN(
            npoint=npoint1, nsample=nsample1, in_channel=in_ch0, mlp=[64, 64, 128], use_xyz=True, group_all=False
        )
        self.sa2 = PointNetSetAbstractionKNN(
            npoint=npoint2, nsample=nsample2, in_channel=128, mlp=[128, 128, 256], use_xyz=True, group_all=False
        )
        self.sa3 = PointNetSetAbstractionKNN(
            npoint=None, nsample=None, in_channel=256, mlp=[256, 512, self.out_dim], use_xyz=True, group_all=True
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _encode_single(self, pts: torch.Tensor) -> torch.Tensor:
        """
        pts: (B,P,3) -> feat: (B,C)
        """
        center = pts.mean(dim=1, keepdim=True)        # (B,1,3)
        xyz = pts - center                           # (B,P,3)
        if self.use_center:
            points = center.expand_as(pts)           # (B,P,3)
        else:
            points = None

        l1_xyz, l1_points = self.sa1(xyz, points)    # (B,n1,3), (B,n1,128)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  # (B,n2,3), (B,n2,256)
        _, l3_points = self.sa3(l2_xyz, l2_points)   # (B,1,out_dim)
        feat = l3_points.squeeze(1)                  # (B,out_dim)
        return feat

    def forward(self, points_in: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, P, _ = points_in.shape
        BT = B * T
        pts = points_in.reshape(BT, P, 3)

        if mask is None:
            feat = self._encode_single(pts).reshape(B, T, self.out_dim)
            return feat

        valid = mask.reshape(BT)
        feat_all = torch.zeros(BT, self.out_dim, device=points_in.device, dtype=points_in.dtype)

        if valid.any():
            feat_valid = self._encode_single(pts[valid])
            feat_all[valid] = feat_valid.to(feat_all.dtype)

        feat_all = feat_all.reshape(B, T, self.out_dim)
        
        return feat_all
