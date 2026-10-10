import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra import initialize_config_module
from sam2.build_sam import build_sam2
from mamba_ssm import Mamba

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 4, scale: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank   = rank
        self.scale  = scale
        in_f  = linear.in_features
        out_f = linear.out_features
        self.lora_A = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f))
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A) @ self.lora_B * self.scale

def apply_lora_to_block(block, rank=4, scale=1.0):
    if hasattr(block, 'attn'):
        attn = block.attn
        if hasattr(attn, 'qkv') and isinstance(attn.qkv, nn.Linear):
            attn.qkv = LoRALinear(attn.qkv, rank=rank, scale=scale)
        if hasattr(attn, 'proj') and isinstance(attn.proj, nn.Linear):
            attn.proj = LoRALinear(attn.proj, rank=rank, scale=scale)
    return block

class MambaAdapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.linear.in_features if isinstance(blk.attn.qkv, LoRALinear) else blk.attn.qkv.in_features
        self.mlp_branch = nn.Sequential(
            nn.Linear(dim, 32), nn.GELU(),
            nn.Linear(32, dim), nn.GELU(),
        )
        self.mamba_branch = Mamba(d_model=dim, d_state=8, d_conv=4, expand=1)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta  = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, H, W, C = x.shape
        mlp_out   = self.mlp_branch(x)
        mamba_out = self.mamba_branch(x.reshape(B, H * W, C)).reshape(B, H, W, C)
        return self.block(x + self.alpha * mamba_out + self.beta * mlp_out)

class CS3MambaBlock(nn.Module):
    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.channels    = channels
        self.d_state     = d_state
        self.norm        = nn.LayerNorm(channels)
        self.mamba_h_fwd = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_h_bwd = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_v_fwd = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        self.mamba_v_bwd = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        for m in [self.mamba_h_bwd, self.mamba_v_fwd, self.mamba_v_bwd]:
            del m.A_log, m.D
            m.A_log = self.mamba_h_fwd.A_log
            m.D     = self.mamba_h_fwd.D
        self.importance_net = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 3, padding=1, groups=channels // 4, bias=False),
            nn.Conv2d(channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
        self.mamba_deform = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=1)
        del self.mamba_deform.A_log, self.mamba_deform.D
        self.mamba_deform.A_log = self.mamba_h_fwd.A_log
        self.mamba_deform.D     = self.mamba_h_fwd.D
        d_small = max(4, d_state // 2)
        self.mamba_small = Mamba(d_model=channels, d_state=d_small, d_conv=d_conv, expand=1)
        self.ms_gate  = nn.Sequential(nn.Linear(channels, channels), nn.Sigmoid())
        self.fusion_w = nn.Linear(channels, 6)
        self.fusion   = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        assert channels % 4 == 0
        self.pos_proj = nn.Linear(channels, channels, bias=False)
        self._pos_cache = {}
        self.num_registers   = 4
        self.register_tokens = nn.Parameter(torch.randn(1, self.num_registers, channels) * 0.02)

    def _get_sincos_pos(self, H, W, C, device, dtype):
        key = (H, W, C, str(device))
        if key not in self._pos_cache:
            d     = C // 4
            h_pos = torch.arange(H, dtype=torch.float32, device=device).unsqueeze(1)
            w_pos = torch.arange(W, dtype=torch.float32, device=device).unsqueeze(1)
            omega = 1.0 / (10000 ** (torch.arange(d, dtype=torch.float32, device=device) / d))
            enc_h = torch.cat([(h_pos * omega).sin(), (h_pos * omega).cos()], dim=-1)
            enc_w = torch.cat([(w_pos * omega).sin(), (w_pos * omega).cos()], dim=-1)
            pos   = torch.cat([
                enc_h.unsqueeze(1).expand(H, W, -1),
                enc_w.unsqueeze(0).expand(H, W, -1)
            ], dim=-1).reshape(H * W, C)
            self._pos_cache[key] = pos
        return self._pos_cache[key].to(dtype=dtype, device=device)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x
        x_seq = self.norm(x.flatten(2).transpose(1, 2))
        regs  = self.register_tokens.expand(B, -1, -1)

        def scan_with_reg(mamba, seq):
            seq_r = torch.cat([regs, seq], dim=1)
            return mamba(seq_r)[:, self.num_registers:]

        y_h_fwd = scan_with_reg(self.mamba_h_fwd, x_seq)
        y_h_bwd = torch.flip(scan_with_reg(self.mamba_h_bwd, torch.flip(x_seq, [1])), [1])
        x_vt    = x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)
        y_v_fwd = scan_with_reg(self.mamba_v_fwd, x_vt)
        y_v_fwd = y_v_fwd.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2).flatten(2).transpose(1, 2)
        y_v_bwd = torch.flip(scan_with_reg(self.mamba_v_bwd, torch.flip(x_vt, [1])), [1])
        y_v_bwd = y_v_bwd.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2).flatten(2).transpose(1, 2)

        importance      = self.importance_net(x)
        importance_flat = importance.flatten(2).squeeze(1)
        sort_idx = torch.argsort(importance_flat, dim=1, descending=True)
        inv_idx  = torch.argsort(sort_idx, dim=1)
        x_sorted = torch.gather(x_seq, 1, sort_idx.unsqueeze(-1).expand(-1, -1, C))
        pos      = self._get_sincos_pos(H, W, C, x.device, x.dtype)
        x_sorted = x_sorted + self.pos_proj(pos[sort_idx])
        y_deform = torch.gather(self.mamba_deform(x_sorted), 1, inv_idx.unsqueeze(-1).expand(-1, -1, C))

        y_small = self.mamba_small(x_seq)
        y_ms    = y_small + self.ms_gate(y_small) * y_h_fwd

        ys = torch.stack([y_h_fwd, y_h_bwd, y_v_fwd, y_v_bwd, y_deform, y_ms], dim=-1)
        w  = torch.softmax(self.fusion_w(x_seq), dim=-1).unsqueeze(2)
        y  = self.fusion((ys * w).sum(dim=-1))
        return y.transpose(1, 2).reshape(B, C, H, W) + identity

class MSCA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        mid  = out_ch // 4
        self.b1 = nn.Sequential(nn.Conv2d(out_ch, mid, 1),                         nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv2d(out_ch, mid, 3, padding=2, dilation=2),  nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b3 = nn.Sequential(nn.Conv2d(out_ch, mid, 3, padding=4, dilation=4),  nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.b4 = nn.Sequential(nn.Conv2d(out_ch, mid, 3, padding=8, dilation=8),  nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.gc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, mid, 1), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(nn.Conv2d(out_ch + mid, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        gc     = self.gc(x).expand(-1, -1, x.shape[2], x.shape[3])
        x_proj = self.proj(x)
        return self.fuse(torch.cat([self.b1(x_proj), self.b2(x_proj), self.b3(x_proj), self.b4(x_proj), gc], dim=1)) + x_proj

class HybridMambaAttn(nn.Module):
    def __init__(self, channels, d_state=16, num_heads=4):
        super().__init__()
        self.norm1   = nn.LayerNorm(channels)
        self.norm2   = nn.LayerNorm(channels)
        self.norm3   = nn.LayerNorm(channels)
        self.mamba   = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=1)
        self.attn    = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, batch_first=True, dropout=0.0)
        self.mamba_w = nn.Parameter(torch.tensor(0.5))
        self.attn_w  = nn.Parameter(torch.tensor(0.5))
        self.ffn     = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, x):
        B, C, H, W = x.shape
        x_seq   = self.norm1(x.flatten(2).transpose(1, 2))
        y_mamba = self.mamba(x_seq)
        y_attn, _ = self.attn(x_seq, x_seq, x_seq)
        y       = self.mamba_w * y_mamba + self.attn_w * y_attn
        y       = y + self.ffn(self.norm2(y))
        return self.norm3(y).transpose(1, 2).reshape(B, C, H, W) + x

class ECA(nn.Module):
    def __init__(self, channels, k=3):
        super().__init__()
        self.avg     = nn.AdaptiveAvgPool2d(1)
        self.conv    = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

class BoundaryAwareDecoder(nn.Module):
    def __init__(self, in_ch, out_ch, use_skip=True, scale=2):
        super().__init__()
        self.use_skip = use_skip
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * scale * scale, 1, bias=False),
            nn.PixelShuffle(scale),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        if use_skip:
            self.skip_eca      = ECA(out_ch)
            self.boundary_gate = nn.Sequential(nn.Conv2d(out_ch * 2 + 1, out_ch, 1, bias=False), nn.Sigmoid())
            self.conv = nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),     nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        self.no_skip_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.up(x)
        if self.use_skip and skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            skip  = self.skip_eca(skip)
            gate  = self.boundary_gate(torch.cat([x, skip, torch.zeros(x.shape[0], 1, *x.shape[2:], device=x.device)], dim=1))
            fused = gate * skip + (1 - gate) * x
            x     = self.conv(torch.cat([fused, skip], dim=1))
        else:
            x = self.no_skip_conv(x)
        return x

class CrossScaleFeatureAggregation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.proj2 = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj3 = nn.Conv2d(channels, channels, 1, bias=False)
        self.proj4 = nn.Conv2d(channels, channels, 1, bias=False)
        self.w1 = nn.Parameter(torch.zeros(1))
        self.w2 = nn.Parameter(torch.zeros(1))
        self.w3 = nn.Parameter(torch.zeros(1))
        self.w4 = nn.Parameter(torch.zeros(1))
        self.refine = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, f1, f2, f3, f4):
        H, W   = f1.shape[2], f1.shape[3]
        f2_up  = F.interpolate(self.proj2(f2), size=(H, W), mode='bilinear', align_corners=False)
        f3_up  = F.interpolate(self.proj3(f3), size=(H, W), mode='bilinear', align_corners=False)
        f4_up  = F.interpolate(self.proj4(f4), size=(H, W), mode='bilinear', align_corners=False)
        fused  = torch.sigmoid(self.w1)*f1 + torch.sigmoid(self.w2)*f2_up + torch.sigmoid(self.w3)*f3_up + torch.sigmoid(self.w4)*f4_up
        return self.refine(fused) + f1

class VMambaSAM(nn.Module):
    def __init__(self, checkpoint_path=None, num_classes=1,
                 use_mamba=True, use_skip_connections=True,
                 use_auxiliary_loss=True, use_msca=True, use_csfa=True,
                 use_mamba_adapter=True, d_state=4):
        super().__init__()
        self.use_mamba            = use_mamba
        self.use_skip_connections = use_skip_connections
        self.use_auxiliary_loss   = use_auxiliary_loss
        self.use_msca             = use_msca
        self.use_csfa             = use_csfa
        self.use_mamba_adapter    = use_mamba_adapter

        model = self._init_sam2(checkpoint_path)
        del model.sam_mask_decoder, model.sam_prompt_encoder
        del model.memory_encoder,   model.memory_attention
        del model.mask_downsample,  model.obj_ptr_tpos_proj
        del model.obj_ptr_proj,     model.image_encoder.neck

        self.encoder = model.image_encoder.trunk
        for p in self.encoder.parameters():
            p.requires_grad = False

        total_blocks = len(self.encoder.blocks)
        lora_start   = total_blocks * 2 // 3
        lora_rank    = 8
        for i, blk in enumerate(self.encoder.blocks):
            if i >= lora_start:
                apply_lora_to_block(blk, rank=lora_rank, scale=1.0)
        print(f"[LoRA] Applied to blocks {lora_start}~{total_blocks-1} (rank={lora_rank}), total={total_blocks} blocks")

        if use_mamba_adapter:
            self.encoder.blocks = nn.Sequential(*[MambaAdapter(b) for b in self.encoder.blocks])
            print("[Adapter] Using MambaAdapter (Mamba + MLP hybrid)")
        else:
            self.encoder.blocks = nn.Sequential(*self.encoder.blocks)
            print("[Adapter] NOT using MambaAdapter (keep original SAM2 blocks only)")

        chs = ([144, 288, 576, 1152] if checkpoint_path and "large"     in checkpoint_path else
               [112, 224, 448, 896]  if checkpoint_path and "base_plus" in checkpoint_path else
               [96, 192, 384, 768])
        C = 96

        if use_msca:
            self.msca1 = MSCA(chs[0], C)
            self.msca2 = MSCA(chs[1], C)
            self.msca3 = MSCA(chs[2], C)
            self.msca4 = MSCA(chs[3], C)
        else:
            self.msca1 = nn.Sequential(nn.Conv2d(chs[0], C, 1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
            self.msca2 = nn.Sequential(nn.Conv2d(chs[1], C, 1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
            self.msca3 = nn.Sequential(nn.Conv2d(chs[2], C, 1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))
            self.msca4 = nn.Sequential(nn.Conv2d(chs[3], C, 1), nn.BatchNorm2d(C), nn.ReLU(inplace=True))

        if use_mamba:
            self.mamba1 = CS3MambaBlock(C, d_state=d_state)
            self.mamba2 = CS3MambaBlock(C, d_state=d_state + 4)
            self.mamba3 = CS3MambaBlock(C, d_state=d_state + 8)
            self.mamba4 = HybridMambaAttn(C, d_state=d_state + 12, num_heads=4)
        else:
            class SimpleConvBlock(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(dim, dim, 3, padding=1, groups=dim), nn.BatchNorm2d(dim), nn.GELU(),
                        nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim),
                    )
                def forward(self, x): return x + self.conv(x)
            self.mamba1 = SimpleConvBlock(C)
            self.mamba2 = SimpleConvBlock(C)
            self.mamba3 = SimpleConvBlock(C)
            self.mamba4 = SimpleConvBlock(C)

        self.up1 = BoundaryAwareDecoder(C, C, use_skip_connections)
        self.up2 = BoundaryAwareDecoder(C, C, use_skip_connections)
        self.up3 = BoundaryAwareDecoder(C, C, use_skip_connections)
        self.up4 = BoundaryAwareDecoder(C, C, use_skip=False)

        self.csfa = CrossScaleFeatureAggregation(C) if use_csfa else nn.Identity()

        def _head():
            return nn.Sequential(
                nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
                nn.Conv2d(32, num_classes, 1),
            )

        if use_auxiliary_loss:
            self.side1 = _head()
            self.side2 = _head()
            self.side3 = _head()
        else:
            self.side1 = self.side2 = self.side3 = None

        self.head = _head()

    def _init_sam2(self, checkpoint_path):
        from hydra.core.global_hydra import GlobalHydra
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_map = {
            "large":     ["sam2_hiera_l.yaml", "sam2_hiera_large.yaml"],
            "base_plus": ["sam2_hiera_b+.yaml", "sam2_hiera_base_plus.yaml", "sam2_hiera_b.yaml"],
            "small":     ["sam2_hiera_s.yaml",  "sam2_hiera_small.yaml"],
            "tiny":      ["sam2_hiera_t.yaml",  "sam2_hiera_tiny.yaml"],
        }
        model_type = "small"
        if checkpoint_path:
            for key in cfg_map:
                if key in checkpoint_path.lower():
                    model_type = key
                    break
        last_error = None
        for cfg_path in [os.path.join(current_dir, "sam2_configs"), "sam2_configs", "../sam2_configs"]:
            for model_cfg in cfg_map[model_type]:
                try:
                    with initialize_config_module(cfg_path, version_base="1.2"):
                        m = build_sam2(model_cfg, checkpoint_path) if checkpoint_path else build_sam2(model_cfg)
                    print(f"[SAM2] Loaded: {cfg_path}/{model_cfg}")
                    return m
                except Exception as e:
                    last_error = e
                    GlobalHydra.instance().clear()
        raise RuntimeError(f"Cannot load SAM2. Last error: {last_error}")

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        f1 = self.mamba1(self.msca1(x1))
        f2 = self.mamba2(self.msca2(x2))
        f3 = self.mamba3(self.msca3(x3))
        f4 = self.mamba4(self.msca4(x4))

        d3    = self.up1(f4, f3)
        side1 = self.side1(d3) if (self.use_auxiliary_loss and self.side1 is not None) else None
        d2    = self.up2(d3, f2)
        side2 = self.side2(d2) if (self.use_auxiliary_loss and self.side2 is not None) else None

        f1 = self.csfa(f1, f2, f3, f4) if self.use_csfa else self.csfa(f1)

        d1    = self.up3(d2, f1)
        side3 = self.side3(d1) if (self.use_auxiliary_loss and self.side3 is not None) else None
        d0    = self.up4(d1)
        out   = F.interpolate(self.head(d0), scale_factor=2, mode="bilinear", align_corners=False)

        if self.use_auxiliary_loss and side1 is not None:
            size  = out.shape[2:]
            side1 = F.interpolate(side1, size=size, mode="bilinear", align_corners=False)
            side2 = F.interpolate(side2, size=size, mode="bilinear", align_corners=False)
            side3 = F.interpolate(side3, size=size, mode="bilinear", align_corners=False)
            return out, side1, side2, side3
        return out

    def get_model_complexity(self, input_size=(1, 3, 352, 352)):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        gflops    = 0.0
        try:
            from fvcore.nn import FlopCountAnalysis
            self.eval()
            dummy = torch.randn(*input_size)
            if next(self.parameters()).is_cuda:
                dummy = dummy.cuda()
            with torch.no_grad():
                gflops = FlopCountAnalysis(self, dummy).total() / 1e9
        except Exception:
            try:
                from thop import profile
                self.eval()
                dummy = torch.randn(*input_size)
                if next(self.parameters()).is_cuda:
                    dummy = dummy.cuda()
                with torch.no_grad():
                    flops, _ = profile(self, inputs=(dummy,), verbose=False)
                    gflops = flops / 1e9
            except Exception:
                H, W   = input_size[2], input_size[3]
                gflops = 3.0 * total * (H * W) / 1e12
        return {
            "total_params":     total,
            "trainable_params": trainable,
            "model_size_mb":    round(total * 4 / 1024 ** 2, 2),
            "gflops":           round(gflops, 2),
        }

MambaSAM2UNetV3 = VMambaSAM
