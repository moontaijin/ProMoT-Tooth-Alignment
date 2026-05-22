import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# EdgeConv helpers (pure PyTorch, no torch_geometric required)
# ---------------------------------------------------------------------------

def _knn(x, k):
    """Compute k-nearest neighbors using cdist.

    Args:
        x: (B, N, C) point features
        k: number of neighbors

    Returns:
        idx: (B, N, k) long tensor of neighbor indices
    """
    # x: (B, N, C)
    dist = torch.cdist(x, x)  # (B, N, N)
    _, idx = dist.topk(k, dim=-1, largest=False)  # (B, N, k) — includes self
    return idx


def _get_graph_feature(x, k, idx=None):
    """Build edge features for EdgeConv: concat(x_i, x_j - x_i).

    Args:
        x: (B, N, C)
        k: number of neighbors
        idx: optional precomputed knn indices (B, N, k)

    Returns:
        (B, N, k, 2*C) edge features
    """
    B, N, C = x.shape
    if idx is None:
        idx = _knn(x, k)  # (B, N, k)

    # Gather neighbors: idx_flat (B, N*k) -> expand for C dim -> gather from x
    idx_flat = idx.reshape(B, N * k)  # (B, N*k)
    idx_expanded = idx_flat.unsqueeze(-1).expand(-1, -1, C)  # (B, N*k, C)
    neighbors = torch.gather(x, 1, idx_expanded).reshape(B, N, k, C)

    x_i = x.unsqueeze(2).expand(-1, -1, k, -1)  # (B, N, k, C)
    edge_feat = torch.cat([x_i, neighbors - x_i], dim=-1)  # (B, N, k, 2C)
    return edge_feat


class EdgeConvBlock(nn.Module):
    """Single EdgeConv layer: kNN -> concat(x_i, x_j-x_i) -> MLP -> max aggregate."""

    def __init__(self, in_dim, out_dim, k=20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        """
        Args:
            x: (B, N, C)
        Returns:
            (B, N, out_dim)
        """
        B, N, C = x.shape
        edge_feat = _get_graph_feature(x, self.k)  # (B, N, k, 2C)
        # Apply MLP per edge
        edge_flat = edge_feat.reshape(B * N * self.k, -1)
        out_flat = self.mlp(edge_flat)  # (B*N*k, out_dim)
        out = out_flat.reshape(B, N, self.k, -1)
        # Max aggregate over neighbors
        out = out.max(dim=2).values  # (B, N, out_dim)
        return out


class EdgeConvEncoder(nn.Module):
    """Multi-stage EdgeConv encoder producing per-tooth global features.

    Processes (B*N, P, 3) point clouds through multiple EdgeConv stages
    then max-pools to get per-tooth features.
    """

    def __init__(self, out_dim=256, k=20, stages=3, hidden_dims=None, in_channels=3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        assert len(hidden_dims) == stages

        layers = []
        in_d = in_channels
        for i in range(stages):
            layers.append(EdgeConvBlock(in_d, hidden_dims[i], k=k))
            in_d = hidden_dims[i]
        self.stages = nn.ModuleList(layers)

        # Final projection
        total_dim = sum(hidden_dims)  # concat all stages
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, pts):
        """
        Args:
            pts: (M, P, 3) point clouds (M = B*N flattened teeth)

        Returns:
            feat: (M, out_dim) per-tooth features
        """
        x = pts  # (M, P, 3)
        stage_outs = []
        for layer in self.stages:
            x = layer(x)
            stage_outs.append(x)

        # Concat multi-scale features
        multi = torch.cat(stage_outs, dim=-1)  # (M, P, total_dim)
        # Max pool over points
        feat = multi.max(dim=1).values  # (M, total_dim)
        feat = self.proj(feat)  # (M, out_dim)
        return feat


# ---------------------------------------------------------------------------
# VN-PointNet: SO(3)-equivariant Vector Neuron encoder
# ---------------------------------------------------------------------------

class VNLinear(nn.Module):
    """Equivariant linear layer operating on vector features (..., 3, C_in) -> (..., 3, C_out)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * (2.0 / in_channels) ** 0.5)

    def forward(self, x):
        # x: (..., 3, C_in) -> (..., 3, C_out)
        return torch.einsum("...ci, oi -> ...co", x, self.weight)


class VNLeakyReLU(nn.Module):
    """Direction-aware activation for vector features."""

    def __init__(self, channels, negative_slope=0.2):
        super().__init__()
        self.negative_slope = negative_slope
        self.map_to_dir = VNLinear(channels, 1)

    def forward(self, x):
        # x: (..., 3, C)
        d = self.map_to_dir(x)  # (..., 3, 1)
        d_norm = F.normalize(d, dim=-2)  # normalize over 3D
        dot = (x * d_norm).sum(dim=-2, keepdim=True)  # (..., 1, C)
        mask = (dot >= 0).float()
        return x * mask + x * (1 - mask) * self.negative_slope


class VNMaxPool(nn.Module):
    """Invariant max-pool over points: (M, P, 3, C) -> (M, C)."""

    def forward(self, x):
        # x: (M, P, 3, C)
        norms = torch.linalg.norm(x, dim=2)  # (M, P, C)
        feat = norms.max(dim=1).values  # (M, C)
        return feat


class VNPointNetEncoder(nn.Module):
    """SO(3)-equivariant PointNet using Vector Neurons.

    Input: (M, P, 3) or (M, P, C) point clouds where C is divisible by 3
    Output: (M, feat_dim) invariant features
    """

    def __init__(self, feat_dim=256, in_channels=3):
        super().__init__()
        assert in_channels % 3 == 0, f"VNPointNet requires in_channels divisible by 3, got {in_channels}"
        self.in_channels = in_channels
        vn_in = in_channels // 3
        self.vn1 = VNLinear(vn_in, 64)
        self.act1 = VNLeakyReLU(64)
        self.vn2 = VNLinear(64, 128)
        self.act2 = VNLeakyReLU(128)
        self.vn3 = VNLinear(128, 256)
        self.act3 = VNLeakyReLU(256)
        self.pool = VNMaxPool()
        self.proj = nn.Linear(256, feat_dim) if feat_dim != 256 else nn.Identity()

    def forward(self, pts):
        """
        Args:
            pts: (M, P, C) point clouds where C is divisible by 3

        Returns:
            feat: (M, feat_dim)
        """
        # Reshape to vector features: (M, P, C) -> (M, P, 3, C//3)
        M, P, C = pts.shape
        x = pts.view(M, P, 3, C // 3)
        x = self.act1(self.vn1(x))
        x = self.act2(self.vn2(x))
        x = self.act3(self.vn3(x))
        feat = self.pool(x)  # (M, 256)
        return self.proj(feat)


# ---------------------------------------------------------------------------
# PointNeXt: SA blocks with inverted bottleneck residual MLP
# ---------------------------------------------------------------------------

def _fps_torch(pts, n_sample):
    """Farthest Point Sampling (pure PyTorch).

    Args:
        pts: (B, N, 3) point clouds
        n_sample: number of points to sample

    Returns:
        idx: (B, n_sample) long tensor of sampled indices
    """
    B, N, _ = pts.shape
    device = pts.device
    idx = torch.zeros(B, n_sample, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), device=device)

    for i in range(n_sample):
        idx[:, i] = farthest
        centroid = pts[torch.arange(B, device=device), farthest].unsqueeze(1)  # (B, 1, 3)
        d = torch.sum((pts - centroid) ** 2, dim=-1)  # (B, N)
        dist = torch.min(dist, d)
        farthest = dist.argmax(dim=-1)  # (B,)

    return idx


class PointNeXtSABlock(nn.Module):
    """PointNeXt Set Abstraction block.

    FPS downsample -> kNN group -> pre-project (in_dim+3 -> out_dim) ->
    max-pool over neighbors -> inverted bottleneck residual MLP.
    """

    def __init__(self, in_dim, out_dim, n_sample, k=16, expansion=4):
        super().__init__()
        self.n_sample = n_sample
        self.k = k

        # Pre-projection: input features + relative coordinates
        self.pre_proj = nn.Sequential(
            nn.Linear(in_dim + 3, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )

        # Skip connection projection (when dims don't match)
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        # Inverted bottleneck residual MLP
        mid_dim = out_dim * expansion
        self.res_mlp = nn.Sequential(
            nn.Linear(out_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, pts, feat):
        """
        Args:
            pts: (B, N, 3) point coordinates
            feat: (B, N, C_in) point features

        Returns:
            pts_new: (B, n_sample, 3) downsampled points
            feat_new: (B, n_sample, C_out) new features
        """
        B, N, C_in = feat.shape

        # FPS downsample
        n_out = min(self.n_sample, N) if self.n_sample is not None else N
        if n_out < N:
            fps_idx = _fps_torch(pts, n_out)  # (B, n_out)
            pts_new = torch.gather(pts, 1, fps_idx.unsqueeze(-1).expand(-1, -1, 3))
            feat_center = torch.gather(feat, 1, fps_idx.unsqueeze(-1).expand(-1, -1, C_in))
        else:
            pts_new = pts
            feat_center = feat

        # kNN grouping
        k = min(self.k, N)
        dist = torch.cdist(pts_new, pts)  # (B, n_out, N)
        _, knn_idx = dist.topk(k, dim=-1, largest=False)  # (B, n_out, k)

        # Gather neighbor features and relative coordinates
        feat_neighbors = torch.gather(
            feat.unsqueeze(1).expand(-1, n_out, -1, -1), 2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, C_in)
        )  # (B, n_out, k, C_in)
        pts_neighbors = torch.gather(
            pts.unsqueeze(1).expand(-1, n_out, -1, -1), 2,
            knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        )  # (B, n_out, k, 3)
        rel_pos = pts_neighbors - pts_new.unsqueeze(2)  # (B, n_out, k, 3)

        # Concatenate features with relative positions
        grouped = torch.cat([feat_neighbors, rel_pos], dim=-1)  # (B, n_out, k, C_in+3)

        # Pre-project and max-pool
        Bn = B * n_out * k
        grouped_flat = grouped.reshape(Bn, -1)
        proj_flat = self.pre_proj(grouped_flat)  # (B*n_out*k, C_out)
        C_out = proj_flat.shape[-1]
        proj = proj_flat.reshape(B, n_out, k, C_out)
        pooled = proj.max(dim=2).values  # (B, n_out, C_out)

        # Skip connection
        skip = self.skip(feat_center)  # (B, n_out, C_out)

        # Inverted bottleneck residual
        res_input = pooled + skip
        res_flat = res_input.reshape(B * n_out, C_out)
        res_out = self.res_mlp(res_flat).reshape(B, n_out, C_out)
        feat_new = F.relu(res_out + res_input)

        return pts_new, feat_new


class PointNeXtEncoder(nn.Module):
    """PointNeXt-style encoder for per-tooth point cloud features.

    3 SA stages with progressive downsampling, then global max-pool
    and projection to feat_dim.

    Input: (M, P, 3) -> Output: (M, feat_dim)
    """

    def __init__(self, out_dim=256, k=16, stage_points=None, stage_dims=None, in_channels=3, expansion=4):
        super().__init__()
        if stage_points is None:
            stage_points = [128, 32, None]  # None = global (keep all remaining)
        if stage_dims is None:
            stage_dims = [64, 128, 256]

        assert len(stage_points) == len(stage_dims)
        self.in_channels = in_channels

        stages = []
        in_d = in_channels  # initial feature dim (3 for xyz, 6 for xyz+center)
        for i, (n_pts, dim) in enumerate(zip(stage_points, stage_dims)):
            stages.append(PointNeXtSABlock(
                in_dim=in_d, out_dim=dim, n_sample=n_pts, k=k, expansion=expansion
            ))
            in_d = dim
        self.stages = nn.ModuleList(stages)

        # Final projection to output dim
        self.proj = nn.Sequential(
            nn.Linear(stage_dims[-1], out_dim),
            nn.ReLU(),
        )

    def forward(self, pts):
        """
        Args:
            pts: (M, P, C) point clouds (C = in_channels)

        Returns:
            feat: (M, out_dim) per-tooth features
        """
        x_pts = pts[..., :3]  # coordinates always first 3 dims
        x_feat = pts  # initial features = all channels

        for stage in self.stages:
            x_pts, x_feat = stage(x_pts, x_feat)

        # Global max pool over remaining points
        feat = x_feat.max(dim=1).values  # (M, last_dim)
        feat = self.proj(feat)  # (M, out_dim)
        return feat


# ---------------------------------------------------------------------------
# ToothNodeEncoder
# ---------------------------------------------------------------------------

class ToothNodeEncoder(nn.Module):
    """Pluggable per-tooth encoder.

    Args:
        mode: "pointnet" | "edgeconv" | "vn_pointnet" | "pointnext"
        feat_dim: output feature dimension
        edgeconv_k: kNN neighbors for edgeconv mode
        edgeconv_stages: number of EdgeConv stages
        edgeconv_hidden_dims: hidden dims per stage (list)
        pointnext_k: kNN neighbors for pointnext mode (default 16)
        pointnext_stage_points: downsample targets per stage (list)
        pointnext_stage_dims: feature dims per stage (list)
    """

    def __init__(self, mode="pointnet", feat_dim=256,
                 edgeconv_k=20, edgeconv_stages=3, edgeconv_hidden_dims=None,
                 pointnext_k=16, pointnext_stage_points=None,
                 pointnext_stage_dims=None,
                 in_channels=3,
                 **kwargs):
        super().__init__()
        self.mode = mode
        self.feat_dim = feat_dim
        self.in_channels = in_channels

        if mode == "pointnet":
            from net.backbone import PointNetEncoder
            self.enc = PointNetEncoder(out_dim=feat_dim, in_channels=in_channels)
        elif mode == "edgeconv":
            self.enc = EdgeConvEncoder(
                out_dim=feat_dim, k=edgeconv_k,
                stages=edgeconv_stages, hidden_dims=edgeconv_hidden_dims,
                in_channels=in_channels,
            )
        elif mode == "vn_pointnet":
            self.enc = VNPointNetEncoder(feat_dim=feat_dim, in_channels=in_channels)
        elif mode == "pointnext":
            self.enc = PointNeXtEncoder(
                out_dim=feat_dim, k=pointnext_k,
                stage_points=pointnext_stage_points,
                stage_dims=pointnext_stage_dims,
                in_channels=in_channels,
            )
        elif mode == "swin":
            from net.swin_encoder import SwinToothEncoder
            self.enc = SwinToothEncoder(
                feat_dim=feat_dim, in_channels=in_channels,
                embed_dim=kwargs.get("swin_embed_dim", 256),
                depths=kwargs.get("swin_depths", (2, 2, 6, 2)),
                num_heads=kwargs.get("swin_num_heads", 4),
                window_size=kwargs.get("swin_window_size", 8),
                mlp_ratio=kwargs.get("swin_mlp_ratio", 4.0),
                center_depth=kwargs.get("swin_center_depth", 4),
                fusion_depth=kwargs.get("swin_fusion_depth", 4),
            )
        else:
            raise ValueError(f"Unknown encoder mode: {mode}")

    @property
    def out_dim(self):
        return self.feat_dim

    def forward(self, pts, mask=None):
        """
        Args:
            pts: (B, N, P, C) or (B, P, C) point clouds
            mask: (B, N) optional mask

        Returns:
            feat: (B, N, feat_dim) or (B, feat_dim)
        """
        if self.mode in ("pointnet", "swin"):
            return self.enc(pts, mask=mask)

        if pts.ndim == 3:
            return self.enc(pts)
        B, N, P, C = pts.shape
        flat = pts.reshape(B * N, P, C)
        feat = self.enc(flat).view(B, N, -1)
        if mask is not None:
            feat = feat * mask.unsqueeze(-1).float()
        return feat



class ToothPointNeXtFeatEncoder(nn.Module):
    """
    ProMoT용 per-tooth PointNeXt feature encoder.
    - 항상 tooth-wise centering: rel = xyz - centroid
    - use_center=True면 center(centroid)를 per-point feature로 concat: [rel(3), center(3)] => 6ch
    Input : points_in (B,T,P,3), mask (B,T) optional
    Output: feat (B,T,out_dim)
    """
    def __init__(
        self,
        out_dim: int = 256,
        use_center: bool = True,
        freeze: bool = False,
        k: int = 16,
        stage_points=(128, 32, None),
        stage_dims=(64, 128, 256),
        expansion: int = 4,
    ):
        super().__init__()
        self.out_dim = int(out_dim)
        self.use_center = bool(use_center)

        in_channels = 6 if self.use_center else 3

        self.backbone = PointNeXtEncoder(
            out_dim=self.out_dim,
            k=int(k),
            stage_points=list(stage_points),
            stage_dims=list(stage_dims),
            in_channels=in_channels,
            expansion=int(expansion),
        )

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, points_in: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # points_in: (B,T,P,3)
        B, T, P, C = points_in.shape
        assert C >= 3, f"points_in last dim must be >=3, got {C}"

        pts = points_in[..., :3].reshape(B * T, P, 3)  # (BT,P,3)

        center = pts.mean(dim=1, keepdim=True)         # (BT,1,3)
        rel = pts - center                              # (BT,P,3)

        if self.use_center:
            cen_feat = center.expand_as(rel)            # (BT,P,3)
            x = torch.cat([rel, cen_feat], dim=-1)      # (BT,P,6)
        else:
            x = rel                                     # (BT,P,3)

        feat = self.backbone(x).reshape(B, T, self.out_dim)  # (B,T,C)

        if mask is not None:
            feat = feat * mask.unsqueeze(-1).float()
        return feat