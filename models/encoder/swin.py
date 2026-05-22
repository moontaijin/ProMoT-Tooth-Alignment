from functools import partial
import torch
import torch.nn as nn
import  torch.nn.functional as F
from image_models.timm.models.vision_transformer import PatchEmbed, Block
from image_models.timm.models.swin_transformer import SwinTransformer, SwinTransformerBlock
from utils.pos_embed import PositionalEncoding
import data.data_config as cfg

class MaskedAutoencoderViT(nn.Module):
    def __init__(self, embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cplinear1 = nn.Linear(3, embed_dim)
        self.plinear1 = nn.Linear(3, embed_dim)
        self.encoder_embed = nn.Parameter(torch.zeros(1, cfg.teeth_nums, embed_dim), requires_grad=False)
        self.decoder_embed = nn.Parameter(torch.zeros(1, cfg.teeth_nums, embed_dim*5), requires_grad=False)
        self.teeth1 = nn.Parameter(torch.zeros(cfg.teeth_nums, 1, 1),requires_grad=True)
        self.teeth_blocks = nn.ModuleList([
            SwinTransformer(img_size=(cfg.teeth_nums, cfg.sam_points), patch_size=1, in_chans=3, window_size=8, num_classes=embed_dim, embed_dim=embed_dim, depths=(2, 2, 6, 2), qkv_bias=True, norm_layer=norm_layer)
            for i in range(1)])
        self.cp_blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim, input_resolution=(cfg.teeth_nums, 1), num_heads=num_heads, window_size=8, shift_size=4, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim*2)
        self.tb_props = nn.ModuleList([
            SwinTransformerBlock(embed_dim*2, input_resolution=(cfg.teeth_nums, 1), num_heads=num_heads, window_size=8, shift_size=4, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.linear21 = nn.Linear(embed_dim*2, embed_dim)
        self.linear22 = nn.Linear(embed_dim, 3)
        self.linear23 = nn.Linear(embed_dim, 4)

        # net/vitnet.py inside MaskedAutoencoderViT.__init__(...)

        self.register_buffer("decoder_t_raw", torch.as_tensor(cfg.decoder_t, dtype=torch.float32).flatten())
        self.register_buffer("decoder_r_raw", torch.as_tensor(cfg.decoder_r, dtype=torch.float32).flatten())

        # 원본의 "거의 0이면 더하지 않음" 조건을 forward 밖에서 한번만 결정(동기화/아이템 호출은 init에서만)
        self._use_decoder_t = int((self.decoder_t_raw != 0).sum().cpu()) >= 5
        self._use_decoder_r = int((self.decoder_r_raw != 0).sum().cpu()) >= 5

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed1 = PositionalEncoding(self.encoder_embed.shape[1], self.encoder_embed.shape[2], self.device)
        self.encoder_embed.data.copy_(pos_embed1.float().unsqueeze(0))
        pos_embed2 = PositionalEncoding(self.decoder_embed.shape[1], self.decoder_embed.shape[2], self.device)
        self.decoder_embed.data.copy_(pos_embed2.float().unsqueeze(0))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _bias_1TD(self, raw_flat: torch.Tensor, T: int, D: int, device, dtype):
        """
        raw_flat: (K,)  -> (1,T,D)로 변환 (필요 시 pad/trim)
        - K==D: global bias (1,1,D) broadcast
        - K==n*D: n개 tooth bias -> n<T면 뒤를 0으로 pad, n>T면 앞에서 trim
        - 그 외: None (추가하지 않음)
        """
        raw = raw_flat.to(device=device, dtype=dtype)

        if raw.numel() == 0:
            return None

        # global bias (3 or 4)
        if raw.numel() == D:
            return raw.view(1, 1, D).expand(1, T, D)

        # per-tooth bias (n*D)
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
    
    # --- net/vitnet.py ---

    def forward_encoder(self, x, return_feat: bool = False):
        """
        x:
        - (T,C,N) or (B,T,C,N)
        return_feat=True이면 head 직전 feature(h)만 반환: (B,T,embed_dim)
        """
        squeeze_out = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_out = True

        B, T, C, N = x.shape
        assert T == cfg.teeth_nums, f"T must be cfg.teeth_nums={cfg.teeth_nums}, got {T}"

        device = x.device
        dtype = x.dtype

        # -------- center branch --------
        center = x[:, :, 3:6, 0].to(dtype)          # (B,T,3)
        xc = self.cplinear1(center)                 # (B,T,embed)
        enc = self.encoder_embed.to(device=device, dtype=dtype)
        xc = xc + enc
        xc = xc.unsqueeze(2)                        # (B,T,1,embed)
        for cpb in self.cp_blocks:
            xc = cpb(xc)
        xc = xc.squeeze(2)                          # (B,T,embed)

        # -------- tooth branch --------
        coords = x[:, :, 0:3, :].to(dtype)          # (B,T,3,N)
        img = coords.permute(0, 2, 1, 3)            # (B,3,T,N)

        teeth_feats = []
        out = img
        for blk in self.teeth_blocks:
            out = blk(out)                          # (B,T,N,embed)
            teeth_feats.append(out)

        out = torch.cat(teeth_feats, dim=-1)        # (B,T,N,embed*L)
        out = out.mean(dim=2)                       # (B,T,embed*L)

        # -------- fuse + tb_props --------
        x2 = torch.cat([out, xc], dim=-1)           # (B,T,2*embed) when L=1
        x_res = x2.unsqueeze(2)                     # (B,T,1,2*embed)
        h = x_res
        for blk in self.tb_props:
            h = blk(h)

        teeth1 = self.teeth1.to(device=device, dtype=dtype).unsqueeze(0)  # (1,T,1,1)
        h = teeth1 * h + x_res
        h = h.squeeze(2)                            # (B,T,2*embed)

        h = F.relu(self.linear21(h))                # (B,T,embed)

        # ✅ 여기서 feature만 반환
        if return_feat:
            return h.squeeze(0) if squeeze_out else h

        # -------- 기존 head --------
        transv = 10.0 * torch.tanh(self.linear22(h))
        dec_t = self._bias_1TD(self.decoder_t_raw, T=cfg.teeth_nums, D=3,
                            device=transv.device, dtype=transv.dtype)
        if self._use_decoder_t and (dec_t is not None):
            transv = transv + dec_t

        q = torch.tanh(self.linear23(h))
        dofx = torch.nn.functional.normalize(q, dim=-1)
        dec_r = self._bias_1TD(self.decoder_r_raw, T=cfg.teeth_nums, D=4,
                            device=dofx.device, dtype=dofx.dtype)
        if self._use_decoder_r and (dec_r is not None):
            dofx = dofx + dec_r

        if squeeze_out:
            return dofx.squeeze(0), transv.squeeze(0)
        return dofx, transv


    # def forward_encoder(self, x):
    #     TB, C, N = x.shape
    #     x = x.permute(0, 2, 1)
    #     xc = self.cplinear1(x[:, 0:1, 3:]).permute(1, 0, 2)
    #     # self.decoder_r = torch.tensor(cfg.decoder_r)
    #     # self.decoder_r = self.decoder_r.to(torch.device('cuda', 0))
    #     # self.decoder_t = torch.tensor(cfg.decoder_t)
    #     # self.decoder_t = self.decoder_t.to(torch.device('cuda', 0))
    #     x = x[:, :, :3]
    #     xc = self.encoder_embed + xc
    #     xc = xc.permute(1, 0, 2)
    #     xc = torch.unsqueeze(xc, dim=0)
    #     for cpb in self.cp_blocks:
    #         xc = cpb(xc)
    #     xc = torch.squeeze(xc, dim=0)
    #     xc = xc.permute(1, 0, 2)
    #     teeths = []
    #     x = x.permute(2, 0, 1)
    #     x = torch.unsqueeze(x, dim=0)
    #     for blk in self.teeth_blocks:
    #         x = blk(x)
    #         t_x = torch.squeeze(x, dim=0)
    #         teeths.append(t_x)
    #     x = torch.squeeze(x, dim=0)
    #     x = x.permute(1, 2, 0)
    #     x = torch.cat(teeths, dim=-1)
    #     x = torch.mean(x, dim=1, keepdim=True)
    #     x = x.permute(1, 0, 2)
    #     x = torch.cat([x, xc],  dim=-1)
    #     x_ = x.clone().permute(1, 0, 2)
    #     x = x.permute(1, 0, 2)
    #     x = torch.unsqueeze(x, dim=0)
    #     for blk in self.tb_props:
    #         x = blk(x)
    #     x = torch.squeeze(x, dim=0)
    #     x = self.teeth1 * x + x_
    #     x = torch.mean(x, dim=1)
    #     x = F.relu(self.linear21(x))
    #     # if torch.count_nonzero(self.decoder_t).item() < 5:
    #     #     transv = 10*F.tanh(self.linear22(x))
    #     # else:
    #     #     transv = 10*F.tanh(self.linear22(x)) + self.decoder_t
    #     # x = F.tanh(self.linear23(x))
    #     # if torch.count_nonzero(self.decoder_r).item() < 5:
    #     #     dofx = torch.nn.functional.normalize(x, dim=-1)
    #     # else:
    #     #     dofx = torch.nn.functional.normalize(x, dim=-1) + self.decoder_r

    #     decoder_t = self.decoder_t.to(device=x.device, dtype=x.dtype)
    #     decoder_r = self.decoder_r.to(device=x.device, dtype=x.dtype)

    #     transv = 10.0 * F.tanh(self.linear22(x)) + decoder_t

    #     q = F.tanh(self.linear23(x))
    #     dofx = torch.nn.functional.normalize(q, dim=-1) + decoder_r

    #     return dofx, transv

    def forward(self, imgs, mask_ratio=0.75):
        dofx, transv = self.forward_encoder(imgs)
        return dofx, transv

def mae_vit_base_patch16(**kwargs):
    model = MaskedAutoencoderViT(embed_dim=256, depth=4, num_heads=4,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model