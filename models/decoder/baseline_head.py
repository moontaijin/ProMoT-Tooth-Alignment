import torch
import torch.nn as nn


class BaselinePoseHead(nn.Module):
    """Baseline quat/trans head used in the original ViTNet.

    This is a *direct extraction* of the head logic (linear21/22/23 + optional per-tooth biases)
    so it can be reused consistently across ablations.

    Input:  h (B,T,C)
    Output: trans (B,T,3), quat (B,T,4)
    """

    def __init__(self, embed_dim: int = 256, use_bias: bool = True, decoder_t=None, decoder_r=None):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.linear21 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.linear22 = nn.Linear(self.embed_dim, 3)
        self.linear23 = nn.Linear(self.embed_dim, 4)

        # store optional biases (copied from cfg.decoder_t / cfg.decoder_r behavior)
        if use_bias and decoder_t is not None and decoder_r is not None:
            dt = torch.as_tensor(decoder_t, dtype=torch.float32).flatten()
            dr = torch.as_tensor(decoder_r, dtype=torch.float32).flatten()
        else:
            dt = torch.zeros(0, dtype=torch.float32)
            dr = torch.zeros(0, dtype=torch.float32)

        self.register_buffer('decoder_t_raw', dt)
        self.register_buffer('decoder_r_raw', dr)

    @staticmethod
    def _bias_1TD(raw_flat: torch.Tensor, T: int, D: int, device, dtype):
        raw = raw_flat.to(device=device, dtype=dtype)
        if raw.numel() == 0:
            return None
        if raw.numel() == D:
            return raw.view(1, 1, D).expand(1, T, D)
        if raw.numel() % D == 0:
            n = raw.numel() // D
            b = raw.view(1, n, D)
            if n < T:
                pad = torch.zeros((1, T - n, D), device=device, dtype=dtype)
                b = torch.cat([b, pad], dim=1)
            elif n > T:
                b = b[:, :T, :]
            return b
        return None

    def forward(self, h2: torch.Tensor):
        """h2: (B,T,2*embed_dim)"""
        B, T, _ = h2.shape
        device, dtype = h2.device, h2.dtype

        h = self.linear21(h2)
        trans = self.linear22(h)
        quat = self.linear23(h)

        bt = self._bias_1TD(self.decoder_t_raw, T=T, D=3, device=device, dtype=dtype)
        br = self._bias_1TD(self.decoder_r_raw, T=T, D=4, device=device, dtype=dtype)
        if bt is not None:
            trans = trans + bt
        if br is not None:
            quat = quat + br

        return quat, trans
