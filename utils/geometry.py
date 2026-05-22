import torch

def hat_so3(w):
    wx, wy, wz = w[...,0], w[...,1], w[...,2]
    O = torch.zeros_like(wx)
    return torch.stack([
        torch.stack([ O, -wz,  wy], dim=-1),
        torch.stack([ wz,  O, -wx], dim=-1),
        torch.stack([-wy, wx,  O], dim=-1),
    ], dim=-2)

def _broadcast_I(n: int, like: torch.Tensor):
    """Create an identity matrix broadcastable to `like` batch shape."""
    batch_shape = like.shape[:-1]  # last dim is vector dim
    I = torch.eye(n, device=like.device, dtype=like.dtype)
    return I.view(*([1] * len(batch_shape)), n, n)


def exp_so3(w, eps=1e-8):
    """Stable SO(3) exponential map.

    w: (...,3) axis-angle vector.
    returns R: (...,3,3)
    """
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (...,1)
    W = hat_so3(w)                                      # (...,3,3)
    W2 = W @ W

    I = _broadcast_I(3, w)

    # coefficients: a = sin(theta)/theta, b = (1-cos(theta))/theta^2
    theta2 = theta * theta
    theta4 = theta2 * theta2

    small = theta < eps
    # Taylor expansions around 0
    a_taylor = 1.0 - theta2 / 6.0 + theta4 / 120.0
    b_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0

    a = torch.where(small, a_taylor, torch.sin(theta) / theta.clamp_min(eps))
    b = torch.where(small, b_taylor, (1.0 - torch.cos(theta)) / theta2.clamp_min(eps))

    R = I + a[..., None] * W + b[..., None] * W2
    return R


def exp_se3(xi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Stable SE(3) exponential map (row-vector convention).

    Row convention in this project:
      - points are row vectors: [x y z 1]
      - apply as: p' = p @ T
      - T layout:
          [ R  0 ]
          [ t  1 ]   where t is a row vector.

    xi: (...,6) with [w(3), v(3)]
    returns T: (...,4,4)

    Standard SE(3) exp under **row-vector / right-multiply** convention.
    With the se(3) hat operator arranged as:
        [ [w]   0 ]
        [  v    0 ]   (v is a row vector)
    the exponential is:
        R = exp([w])
        t = v · J(w)
    where J(w) is the SO(3) left Jacobian:
        J = I + (1-cosθ)/θ^2 [w] + (θ-sinθ)/θ^3 [w]^2
    """
    w = xi[..., :3]
    v = xi[..., 3:]

    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (...,1)
    W = hat_so3(w)
    W2 = W @ W

    R = exp_so3(w, eps=eps)
    I3 = _broadcast_I(3, w)

    theta2 = theta * theta
    theta3 = theta2 * theta
    theta4 = theta2 * theta2

    small = theta < eps

    # a = (1-cosθ)/θ^2, b = (θ-sinθ)/θ^3
    a_taylor = 0.5 - theta2 / 24.0 + theta4 / 720.0
    b_taylor = 1.0 / 6.0 - theta2 / 120.0 + theta4 / 5040.0

    a = torch.where(small, a_taylor, (1.0 - torch.cos(theta)) / theta2.clamp_min(eps))
    b = torch.where(small, b_taylor, (theta - torch.sin(theta)) / theta3.clamp_min(eps))

    J = I3 + a[..., None] * W + b[..., None] * W2
    # row translation: t = v @ J
    t = (v.unsqueeze(-2) @ J).squeeze(-2)  # (...,3)

    batch_shape = xi.shape[:-1]
    I4 = torch.eye(4, device=xi.device, dtype=xi.dtype)
    T = I4.view(*([1] * len(batch_shape)), 4, 4).repeat(*batch_shape, 1, 1).clone()
    T[..., :3, :3] = R
    T[..., 3, :3] = t
    return T

def exp_se3_small(xi):
    """Deprecated alias.

    Previously used a small-angle SE(3) approximation that ignores the
    V-matrix coupling between rotation and translation.
    Kept for backward-compatibility; now calls exp_se3.
    """
    return exp_se3(xi)

def rotation_geodesic_loss(R_pred, R_gt, eps=1e-6):
    R_rel = R_pred.transpose(-1,-2) @ R_gt
    tr = R_rel[...,0,0] + R_rel[...,1,1] + R_rel[...,2,2]
    cos = ((tr - 1.0) / 2.0).clamp(-1+eps, 1-eps)
    return torch.acos(cos)  # (...,)

def apply_T_points(T, points):
    """
    T: (B,N,4,4) row convention
    points: (B,N,P,3)
    """
    B,N,P,_ = points.shape
    ones = torch.ones((B,N,P,1), device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=-1)  # (B,N,P,4)
    out = (homo @ T)[..., :3]
    return out

def log_so3(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)

    vee = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    )

    sin_theta = torch.sin(theta)
    small = theta < 1e-4
    near_pi = (torch.pi - theta) < 1e-3

    scale = theta / (2.0 * (sin_theta + eps))
    w = vee * scale.unsqueeze(-1)

    w_small = 0.5 * vee
    w = torch.where(small.unsqueeze(-1), w_small, w)

    if torch.any(near_pi):
        I = torch.eye(3, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (3, 3))
        A = (R + I) * 0.5
        axis = torch.sqrt(torch.clamp(torch.stack([A[..., 0, 0], A[..., 1, 1], A[..., 2, 2]], dim=-1), min=0.0))

        k = torch.argmax(axis, dim=-1)
        ax, ay, az = axis.unbind(-1)
        ax = torch.where(k == 0, ax, ax * torch.sign(A[..., 0, 1] + A[..., 0, 2] + eps))
        ay = torch.where(k == 1, ay, ay * torch.sign(A[..., 0, 1] + A[..., 1, 2] + eps))
        az = torch.where(k == 2, az, az * torch.sign(A[..., 0, 2] + A[..., 1, 2] + eps))
        axis = torch.stack([ax, ay, az], dim=-1)
        axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + eps)

        w_pi = axis * theta.unsqueeze(-1)
        w = torch.where(near_pi.unsqueeze(-1), w_pi, w)

    return w


def inv_left_jacobian_so3(w: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # exp_se3(): t = v @ J(w)  (row convention)
    theta = torch.linalg.norm(w, dim=-1, keepdim=True)  # (...,1)
    W = hat_so3(w)
    W2 = W @ W
    I = _broadcast_I(3, w)

    theta2 = (theta * theta).clamp_min(eps)
    sin_t = torch.sin(theta).clamp_min(eps)
    cos_t = torch.cos(theta)

    # J^{-1} = I - 0.5W + A W^2
    # A = 1/theta^2 - (1+cos)/(2*theta*sin)
    A = (1.0 / theta2) - ((1.0 + cos_t) / (2.0 * theta * sin_t))

    Jinv = I - 0.5 * W + A[..., None] * W2

    small = (theta.squeeze(-1) < 1e-4)
    Jinv_small = I - 0.5 * W + (1.0 / 12.0) * W2
    Jinv = torch.where(small[..., None, None], Jinv_small, Jinv)
    return Jinv


def invert_se3(T: torch.Tensor) -> torch.Tensor:
    # row convention inverse: [[R^T,0],[-t R^T,1]]
    R = T[..., :3, :3]
    t = T[..., 3, :3]
    Rt = R.transpose(-1, -2)
    tinv = -(t.unsqueeze(-2) @ Rt).squeeze(-2)

    I4 = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape)
    out = I4.clone()
    out[..., :3, :3] = Rt
    out[..., 3, :3] = tinv
    return out


def log_se3(T: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    # row convention: T[:3,:3]=R, T[3,:3]=t(row)
    R = T[..., :3, :3]
    t = T[..., 3, :3]

    w = log_so3(R, eps=eps)
    Jinv = inv_left_jacobian_so3(w, eps=eps)
    v = (t.unsqueeze(-2) @ Jinv).squeeze(-2)
    return torch.cat([w, v], dim=-1)