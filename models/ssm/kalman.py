# net/kalman_smooth_ssm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.geometry import exp_se3

class KalmanSmoothRefinementSSM(nn.Module):
    """
    K-step latent rollout + neural measurements + diagonal Kalman smoothing + Δxi integration.
    encoder는 반드시 forward(points_in, mask=None)->(B,N,C), 그리고 out_dim 속성이 있어야 함.
    """
    def __init__(
        self,
        hidden_dim=256,
        K=8,
        use_tooth_id_emb=True,
        encoder=None,  # ✅ STTAlign encoder 주입
    ):
        super().__init__()
        if encoder is None:
            raise ValueError("KalmanSmoothRefinementSSM requires an injected encoder in STTAlign project.")
        self.enc = encoder
        self.K = int(K)
        self.hidden_dim = int(hidden_dim)

        feat_dim = int(self.enc.out_dim)

        self.use_tooth_id_emb = bool(use_tooth_id_emb)
        id_dim = 0
        if self.use_tooth_id_emb:
            self.id_emb = nn.Embedding(33, 32)  # 0 padding, 1..32
            id_dim = 32

        self.step_emb = nn.Embedding(self.K, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        obs_in = feat_dim + feat_dim + id_dim + 32 + 32  # e, g, id, step, time
        self.obs_net = nn.Sequential(
            nn.Linear(obs_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6),
        )

        self.a_raw = nn.Parameter(torch.zeros(hidden_dim))
        self.q_raw = nn.Parameter(torch.zeros(hidden_dim))
        self.r_raw = nn.Parameter(torch.zeros(hidden_dim))
        self.eps = 1e-6

    def _kalman_smooth_diag(self, y_seq, dt_seq):
        """
        y_seq: (K, BN, H)
        dt_seq: (BN, 1)
        return: (K, BN, H)
        """
        K, BN, H = y_seq.shape
        device, dtype = y_seq.device, y_seq.dtype

        a_diag = -F.softplus(self.a_raw).to(device=device, dtype=dtype)     # (H,)
        Ad = torch.exp(dt_seq * a_diag.view(1, H))                          # (BN,H)

        q = F.softplus(self.q_raw).to(device=device, dtype=dtype) + self.eps
        r = F.softplus(self.r_raw).to(device=device, dtype=dtype) + self.eps
        q = q.view(1, H).expand(BN, H)
        r = r.view(1, H).expand(BN, H)

        m_pred = torch.zeros((K, BN, H), device=device, dtype=dtype)
        P_pred = torch.zeros((K, BN, H), device=device, dtype=dtype)
        m_filt = torch.zeros((K, BN, H), device=device, dtype=dtype)
        P_filt = torch.zeros((K, BN, H), device=device, dtype=dtype)

        m = torch.zeros((BN, H), device=device, dtype=dtype)   # m0
        P = torch.ones((BN, H), device=device, dtype=dtype)    # P0

        for k in range(K):
            m = Ad * m
            P = (Ad * Ad) * P + q
            m_pred[k], P_pred[k] = m, P

            y = y_seq[k]
            S = P + r
            Kgain = P / S
            m = m + Kgain * (y - m)
            P = (1.0 - Kgain) * P
            m_filt[k], P_filt[k] = m, P

        m_s = m_filt[-1].clone()
        P_s = P_filt[-1].clone()

        m_smooth = torch.zeros_like(m_filt)
        P_smooth = torch.zeros_like(P_filt)
        m_smooth[-1] = m_s
        P_smooth[-1] = P_s

        for k in range(K - 2, -1, -1):
            denom = P_pred[k + 1].clamp_min(self.eps)
            Ck = P_filt[k] * Ad / denom
            m_s = m_filt[k] + Ck * (m_s - m_pred[k + 1])
            P_s = P_filt[k] + (Ck * Ck) * (P_s - P_pred[k + 1])
            m_smooth[k] = m_s
            P_smooth[k] = P_s

        return m_smooth

    def forward(self, points_in, tooth_id, mask, t, return_intermediates=False):
        """
        points_in: (B,N,P,3)
        tooth_id:  (B,N)
        mask:      (B,N) bool
        t:         (B,1)
        """
        B, N, _, _ = points_in.shape
        device, dtype = points_in.device, points_in.dtype

        e = self.enc(points_in, mask=mask)                         # (B,N,C)
        e_masked = e.clone()
        e_masked[~mask] = -1e9
        g = e_masked.max(dim=1).values.unsqueeze(1).repeat(1, N, 1)  # (B,N,C)

        if self.use_tooth_id_emb:
            tid = torch.clamp(tooth_id, 0, 32)
            emb = self.id_emb(tid)
        else:
            emb = None

        dt = (t / float(self.K)).to(device=device, dtype=dtype)     # (B,1)
        dt_bn = dt.repeat(1, N).reshape(B * N, 1)                   # (BN,1)

        time_feat = self.time_mlp(t.to(device=device, dtype=dtype)) # (B,32)
        time_feat = time_feat.unsqueeze(1).repeat(1, N, 1)          # (B,N,32)

        y_seq = []
        for k in range(self.K):
            step_vec = self.step_emb.weight[k].view(1, 1, -1).expand(B, N, -1)  # (B,N,32)
            if self.use_tooth_id_emb:
                obs_in = torch.cat([e, g, emb, step_vec, time_feat], dim=-1)
            else:
                obs_in = torch.cat([e, g, step_vec, time_feat], dim=-1)

            y = self.obs_net(obs_in).reshape(B * N, self.hidden_dim)  # (BN,H)
            y = y * mask.reshape(B * N, 1).float()
            y_seq.append(y)

        y_seq = torch.stack(y_seq, dim=0)                            # (K,BN,H)
        h_smooth = self._kalman_smooth_diag(y_seq, dt_bn)             # (K,BN,H)

        T = torch.eye(4, device=device, dtype=dtype).view(1, 1, 4, 4).repeat(B, N, 1, 1)
        deltas = []
        T_steps = [] if return_intermediates else None

        for k in range(self.K):
            h_k = h_smooth[k].reshape(B, N, self.hidden_dim)
            xi = self.delta_head(h_k.reshape(B * N, self.hidden_dim)).reshape(B, N, 6)
            xi = xi * mask.unsqueeze(-1).float()

            # 안정화 clamp (원본 그대로)
            w = torch.clamp(xi[..., :3], -0.1, 0.1)
            v = torch.clamp(xi[..., 3:], -0.5, 0.5)
            xi = torch.cat([w, v], dim=-1)

            dT = exp_se3(xi)
            T = T @ dT
            deltas.append(xi)

            if return_intermediates:
                T_steps.append(T)

        h_last = h_smooth[-1].reshape(B, N, self.hidden_dim)

        if not return_intermediates:
            return T, deltas, h_last

        aux = {
            "y_seq": y_seq.reshape(self.K, B, N, self.hidden_dim),
            "h_smooth": h_smooth.reshape(self.K, B, N, self.hidden_dim),
            "T_steps": torch.stack(T_steps, dim=0),        # (K,B,N,4,4)
            "deltas": torch.stack(deltas, dim=0),          # (K,B,N,6)
        }
        return T, deltas, h_last, aux
