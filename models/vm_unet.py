

import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from einops import rearrange, repeat
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    print("[VM-UNet] einops not installed – some ops will use manual reshape.")

try:
    from timm.models.layers import DropPath, trunc_normal_
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

    class DropPath(nn.Module):
        def __init__(self, drop_prob=0.):
            super().__init__()
            self.drop_prob = drop_prob
        def forward(self, x):
            if self.drop_prob == 0. or not self.training:
                return x
            keep = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            r = torch.rand(shape, dtype=x.dtype, device=x.device)
            return x * (r < keep).float() / keep

    def trunc_normal_(tensor, std=0.02):
        nn.init.trunc_normal_(tensor, std=std)

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    SELECTIVE_SCAN_AVAILABLE = True
except ImportError:
    SELECTIVE_SCAN_AVAILABLE = False
    print("[VM-UNet] mamba_ssm not installed – SS2D will use a Conv fallback.")

class PatchEmbed2D(nn.Module):
    """4×4 patch embedding (official: Conv2d + optional LN)."""
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=None):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    """Downsample ×2, channels ×2 (official PatchMerging2D)."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        H2, W2 = H // 2, W // 2
        x0 = x[:, 0::2, 0::2, :][:, :H2, :W2, :]
        x1 = x[:, 1::2, 0::2, :][:, :H2, :W2, :]
        x2 = x[:, 0::2, 1::2, :][:, :H2, :W2, :]
        x3 = x[:, 1::2, 1::2, :][:, :H2, :W2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        return self.reduction(self.norm(x))

class PatchExpand2D(nn.Module):
    """Upsample ×2, channels ÷2 (official PatchExpand2D)."""
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim * 2, dim_scale * dim * 2, bias=False)
        self.norm   = norm_layer(dim * 2 // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        p = 2
        c = x.shape[-1] // (p * p)
        x = x.view(B, H, W, p, p, c).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * p, W * p, c)
        return self.norm(x)

class Final_PatchExpand2D(nn.Module):
    """Final 4× upsample (official Final_PatchExpand2D)."""
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm   = norm_layer(dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        p = 4
        c = x.shape[-1] // (p * p)
        x = x.view(B, H, W, p, p, c).permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * p, W * p, c)
        return self.norm(x)

class SS2D(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0., conv_bias=True, bias=False, **kwargs):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_conv   = d_conv
        self.expand   = expand
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)

        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner,
                                groups=self.d_inner, bias=conv_bias,
                                kernel_size=d_conv,
                                padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()

        if SELECTIVE_SCAN_AVAILABLE:

            x_projs = [nn.Linear(self.d_inner,
                                 self.dt_rank + self.d_state * 2,
                                 bias=False) for _ in range(4)]
            self.x_proj_weight = nn.Parameter(
                torch.stack([t.weight for t in x_projs], dim=0))

            dt_projs = [self._dt_init(self.dt_rank, self.d_inner,
                                      dt_scale, dt_init,
                                      dt_min, dt_max, dt_init_floor)
                        for _ in range(4)]
            self.dt_projs_weight = nn.Parameter(
                torch.stack([t.weight for t in dt_projs], dim=0))
            self.dt_projs_bias   = nn.Parameter(
                torch.stack([t.bias  for t in dt_projs], dim=0))

            self.A_logs = self._A_log_init(d_state, self.d_inner,
                                           copies=4, merge=True)
            self.Ds     = self._D_init(self.d_inner,
                                       copies=4, merge=True)

            self.out_norm = nn.LayerNorm(self.d_inner)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
            self.dropout  = nn.Dropout(dropout) if dropout > 0. else None
            self._use_scan = True
        else:

            self.fallback = nn.Sequential(
                nn.Conv2d(self.d_inner, self.d_inner, 3, padding=1,
                          groups=self.d_inner, bias=False),
                nn.GELU(),
                nn.Conv2d(self.d_inner, self.d_inner, 1, bias=False),
            )
            self.out_norm = nn.LayerNorm(self.d_inner)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
            self.dropout  = nn.Dropout(dropout) if dropout > 0. else None
            self._use_scan = False

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1., dt_init="random",
                 dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        else:
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = torch.arange(1, d_state + 1, dtype=torch.float32,
                         device=device).unsqueeze(0).expand(d_inner, -1)
        A_log = torch.log(A)
        if copies > 1:
            A_log = A_log.unsqueeze(0).expand(copies, -1, -1)
            if merge:
                A_log = A_log.flatten(0, 1)
        p = nn.Parameter(A_log)
        p._no_weight_decay = True
        return p

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = D.unsqueeze(0).expand(copies, -1)
            if merge:
                D = D.flatten(0, 1)
        p = nn.Parameter(D)
        p._no_weight_decay = True
        return p

    def forward(self, x: torch.Tensor):
        """x: (B, H, W, C)  →  (B, H, W, C)"""
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x_, z = xz.chunk(2, dim=-1)

        x_ = x_.permute(0, 3, 1, 2).contiguous()
        x_ = self.act(self.conv2d(x_))

        if self._use_scan:
            out = self._forward_scan(x_, B, H, W)
        else:
            out = self.fallback(x_)
            out = out.permute(0, 2, 3, 1)

        out = self.out_norm(out)
        out = out * F.silu(z)
        out = self.out_proj(out)
        if self.dropout:
            out = self.dropout(out)
        return out

    def _forward_scan(self, x_, B, H, W):
        """4-direction selective scan — mirrors official forward_corev0."""
        L   = H * W
        K   = 4

        x_hw = x_.view(B, -1, L)
        x_wh = x_.transpose(2, 3).contiguous().view(B, -1, L)
        xs   = torch.cat([
            x_hw.unsqueeze(1),
            x_wh.unsqueeze(1),
            torch.flip(x_hw, [-1]).unsqueeze(1),
            torch.flip(x_wh, [-1]).unsqueeze(1),
        ], dim=1)

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs.view(B, K, -1, L),
            self.x_proj_weight,
        )
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts.view(B, K, -1, L),
            self.dt_projs_weight,
        )

        xs   = xs.float().view(B, -1, L)
        dts  = dts.contiguous().float().view(B, -1, L)
        Bs   = Bs.float().view(B, K, -1, L)
        Cs   = Cs.float().view(B, K, -1, L)
        Ds   = self.Ds.float().view(-1)
        As   = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan_fn(
            xs, dts, As, Bs, Cs, Ds,
            z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y  = torch.flip(out_y[:, 2:4], dims=[-1])
        wh_y   = out_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)
        invwh_y = inv_y[:, 1].view(B, -1, W, H).transpose(2, 3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        return y

class VSSBlock(nn.Module):
 
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0.,
                 norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
                 attn_drop_rate: float = 0., d_state: int = 16, **kwargs):
        super().__init__()
        self.ln_1           = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim,
                                   dropout=attn_drop_rate,
                                   d_state=d_state, **kwargs)
        self.drop_path      = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))

class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            ) for i in range(depth)
        ])
        self.downsample = downsample(dim=dim, norm_layer=norm_layer)\
            if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, upsample=None,
                 use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            ) for i in range(depth)
        ])
        self.upsample = upsample(dim=dim, norm_layer=norm_layer)\
            if upsample else None

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        for blk in self.blocks:
            x = checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
        return x

class VSSM(nn.Module):

    def __init__(self, patch_size=4, in_chans=3, num_classes=1,
                 depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                 dims=[96, 192, 384, 768],
                 dims_decoder=[768, 384, 192, 96],
                 d_state=16, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers  = len(depths)
        self.embed_dim   = dims[0]
        self.dims        = dims

        self.patch_embed = PatchEmbed2D(
            patch_size=patch_size, in_chans=in_chans,
            embed_dim=self.embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_dec = list(reversed(
            [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))]))

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(VSSLayer(
                dim=dims[i], depth=depths[i], d_state=d_state,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i+1])],
                norm_layer=norm_layer,
                downsample=PatchMerging2D if i < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            ))

        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers_up.append(VSSLayer_up(
                dim=dims_decoder[i], depth=depths_decoder[i], d_state=d_state,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr_dec[sum(depths_decoder[:i]):sum(depths_decoder[:i+1])],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if i != 0 else None,
                use_checkpoint=use_checkpoint,
            ))

        self.final_up   = Final_PatchExpand2D(dim=dims_decoder[-1],
                                               dim_scale=4,
                                               norm_layer=norm_layer)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        skip_list = []
        x = self.patch_embed(x)
        for layer in self.layers:
            skip_list.append(x)
            x = layer(x)

        for i, layer_up in enumerate(self.layers_up):
            if i == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skip_list[-i])

        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        x = self.final_conv(x)
        return x

class VMUNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1,
                 depths=(2, 2, 9, 2), depths_decoder=(2, 9, 2, 2),
                 drop_path_rate=0.2, d_state=16,

                 C=96, enc_blocks=None, dec_blocks=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.vmunet = VSSM(
            in_chans=in_channels,
            num_classes=num_classes,
            depths=list(depths),
            depths_decoder=list(depths_decoder),
            d_state=d_state,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.vmunet(x)

if __name__ == "__main__":
    model = VMUNet(in_channels=3, num_classes=1, d_state=16)
    x     = torch.randn(2, 3, 352, 352)

