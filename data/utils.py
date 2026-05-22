import os
import math
import numpy as np
import torch

def walkFile(path_root, file_list):
    for root, dirs, files in os.walk(path_root):
        for d in dirs:
            path_file = os.path.join(root, d)
            file_list.append(path_file)

def get_files(file_dir, file_list, type_str):
    for file_ in os.listdir(file_dir):
        path = os.path.join(file_dir, file_)
        if os.path.isdir(path):
            get_files(path, file_list, type_str)
        else:
            if file_.rfind(type_str) != -1:
                file_list.append(path)

def rotation_matrix(rotate_axis, rotate_angle):
    M_PI = math.pi
    axis = rotate_axis
    angle = rotate_angle
    m = np.zeros((4,4) ,np.float64)
    a = angle * (M_PI / 180.0)
    c = math.cos(a)
    s = math.sin(a)
    one_m_c = 1 - c
    ax = axis / np.sqrt(np.sum(np.power(axis, 2)))
    m[0, 0] = ax[0] * ax[0] * one_m_c + c
    m[0, 1] = ax[0] * ax[1] * one_m_c - ax[2] * s
    m[0, 2] = ax[0] * ax[2] * one_m_c + ax[1] * s
    m[1, 0] = ax[1] * ax[0] * one_m_c + ax[2] * s
    m[1, 1] = ax[1] * ax[1] * one_m_c + c
    m[1, 2] = ax[1] * ax[2] * one_m_c - ax[0] * s
    m[2, 0] = ax[2] * ax[0] * one_m_c - ax[1] * s
    m[2, 1] = ax[2] * ax[1] * one_m_c + ax[0] * s
    m[2, 2] = ax[2] * ax[2] * one_m_c + c
    m[3, 3] = 1.0
    return m

def rotate_maxtrix(rotaxis, angle_):
    M_PI = math.pi
    rt = np.eye(4)
    if (np.sqrt(rotaxis.dot(rotaxis)) > 0.001):
        rotaxis = rotaxis / np.sqrt(np.sum(np.power(rotaxis, 2)))
        rotangle = angle_
        rt = rotation_matrix(rotaxis, rotangle)
    return rt


def _skew(w: torch.Tensor) -> torch.Tensor:
    # w: (..., 3)
    wx, wy, wz = w.unbind(-1)
    O = torch.zeros_like(wx)
    return torch.stack(
        [
            torch.stack([O, -wz, wy], dim=-1),
            torch.stack([wz, O, -wx], dim=-1),
            torch.stack([-wy, wx, O], dim=-1),
        ],
        dim=-2,
    )


def so3_log_map(R: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Robust-ish SO(3) log map.
    R: (..., 3, 3) -> w: (..., 3) (axis-angle)
    """
    # trace to angle
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta)  # (...,)

    # vee(R - R^T)
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

    # 일반 구간: w = theta/(2 sin theta) * vee
    scale = theta / (2.0 * (sin_theta + eps))
    w = vee * scale.unsqueeze(-1)

    # 소각: w ~ 0.5*vee
    w_small = 0.5 * vee
    w = torch.where(small.unsqueeze(-1), w_small, w)

    # theta ~ pi 부근 안정화: axis를 (R+I)/2에서 추정
    if torch.any(near_pi):
        I = torch.eye(3, device=R.device, dtype=R.dtype).expand(R.shape[:-2] + (3, 3))
        A = (R + I) * 0.5  # (...,3,3)
        axis = torch.sqrt(torch.clamp(torch.stack([A[..., 0, 0], A[..., 1, 1], A[..., 2, 2]], dim=-1), min=0.0))

        # 부호 보정(대각이 가장 큰 축 기준)
        k = torch.argmax(axis, dim=-1)  # (...)
        # off-diagonal 기반 부호
        ax = axis[..., 0]
        ay = axis[..., 1]
        az = axis[..., 2]
        ax = torch.where(k == 0, ax, ax * torch.sign(A[..., 0, 1] + A[..., 0, 2] + eps))
        ay = torch.where(k == 1, ay, ay * torch.sign(A[..., 0, 1] + A[..., 1, 2] + eps))
        az = torch.where(k == 2, az, az * torch.sign(A[..., 0, 2] + A[..., 1, 2] + eps))
        axis = torch.stack([ax, ay, az], dim=-1)
        axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + eps)

        w_pi = axis * theta.unsqueeze(-1)
        w = torch.where(near_pi.unsqueeze(-1), w_pi, w)

    return w


def se3_log_map(T: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    SE(3) log map.
    T: (..., 4, 4) where T[:3,:3]=R, T[:3,3]=t (last column)
    returns xi: (..., 6) = [w(3), v(3)]
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]

    w = so3_log_map(R, eps=eps)                     # (...,3)
    theta = torch.linalg.norm(w, dim=-1)            # (...,)

    W = _skew(w)                                    # (...,3,3)
    W2 = W @ W

    I = torch.eye(3, device=T.device, dtype=T.dtype).expand(T.shape[:-2] + (3, 3))

    # V^{-1} = I + 0.5 W + A W^2
    # A = (1/theta^2) * (1 - theta*sin(theta)/(2*(1-cos(theta))))
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    denom = 2.0 * (1.0 - cos_theta) + eps
    A = (1.0 - (theta * sin_theta) / denom) / (theta * theta + eps)

    Vinv = I + 0.5 * W + A.unsqueeze(-1).unsqueeze(-1) * W2

    # small-angle Taylor: Vinv ≈ I + 0.5 W + 1/12 W^2
    small = theta < 1e-4
    Vinv_small = I + 0.5 * W + (1.0 / 12.0) * W2
    Vinv = torch.where(small.unsqueeze(-1).unsqueeze(-1), Vinv_small, Vinv)

    v = (Vinv @ t.unsqueeze(-1)).squeeze(-1)        # (...,3)
    return torch.cat([w, v], dim=-1)

def quaternion_to_axis_angle(q: torch.Tensor, eps: float = 1e-8, order: str = "wxyz") -> torch.Tensor:
    """
    q: (..., 4) quaternion
    returns: (..., 3) axis-angle vector (axis * angle), angle in [0, pi]
    - order="wxyz": q = [w, x, y, z]
    - order="xyzw": q = [x, y, z, w]
    """
    if q.shape[-1] != 4:
        raise ValueError(f"q must have shape (...,4). Got {q.shape}")

    if order == "wxyz":
        w, x, y, z = q.unbind(-1)
    elif order == "xyzw":
        x, y, z, w = q.unbind(-1)
    else:
        raise ValueError("order must be 'wxyz' or 'xyzw'")

    # normalize
    norm = torch.sqrt(w*w + x*x + y*y + z*z + eps)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # enforce w >= 0 for shortest rotation (angle <= pi)
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))
    w, x, y, z = w * sign, x * sign, y * sign, z * sign

    v = torch.stack([x, y, z], dim=-1)                  # (...,3)
    v_norm = torch.linalg.norm(v, dim=-1)               # (...)

    # angle = 2*atan2(||v||, w)
    angle = 2.0 * torch.atan2(v_norm, torch.clamp(w, min=eps))

    # axis = v / ||v||, axis-angle = axis * angle
    axis = v / (v_norm.unsqueeze(-1) + eps)
    axis_angle = axis * angle.unsqueeze(-1)

    # small-angle: sin(theta/2) ~ theta/2 => v ~ axis*(theta/2) => axis_angle ~ 2*v
    small = v_norm < 1e-6
    axis_angle_small = 2.0 * v
    axis_angle = torch.where(small.unsqueeze(-1), axis_angle_small, axis_angle)

    return axis_angle