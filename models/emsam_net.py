

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LFEMBlock(nn.Module):
    """单个 LFEM 卷积组：3×3 主卷积 + 空洞卷积并行分支"""
    def __init__(self, in_ch, out_ch, dilation=2):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.main(x), self.dilated(x)], dim=1))

class LFEM(nn.Module):

    def __init__(self, in_ch=3, base_ch=64):
        super().__init__()
        self.stage1 = LFEMBlock(in_ch,      base_ch,     dilation=2)
        self.pool1  = nn.MaxPool2d(2)
        self.stage2 = LFEMBlock(base_ch,    base_ch * 2, dilation=2)
        self.pool2  = nn.MaxPool2d(2)
        self.stage3 = LFEMBlock(base_ch*2,  base_ch * 4, dilation=4)
        self.pool3  = nn.MaxPool2d(2)
        self.stage4 = LFEMBlock(base_ch*4,  base_ch * 8, dilation=4)

        self.align = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1),
            nn.LayerNorm([base_ch * 4, 1, 1]),
            nn.GELU()
        )

        self.align = nn.Sequential(
            nn.Conv2d(base_ch * 8, base_ch * 4, 3, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.GELU()
        )

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(self.pool1(f1))
        f3 = self.stage3(self.pool2(f2))
        f4 = self.stage4(self.pool3(f3))
        return self.align(f4)

class MAA(nn.Module):

    def __init__(self, dim, r=4, scales=(1, 2, 4, 8)):
        super().__init__()
        mid = dim // r
        self.down = nn.Linear(dim, mid)
        self.scales = scales

        self.dw_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid, mid, 1),
                nn.Sigmoid()
            ) for _ in scales
        ])
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(mid * len(scales), mid, 3, padding=1, groups=mid),
            nn.GELU(),
            nn.Conv2d(mid, mid, 1)
        )
        self.up = nn.Linear(mid, dim)

    def forward(self, x):

        B, L, D = x.shape
        H = W = int(math.sqrt(L))

        h = F.relu(self.down(x))
        h_2d = h.transpose(1, 2).reshape(B, -1, H, W)

        outs = []
        for pool_size, dw in zip(self.scales, self.dw_convs):
            pooled = F.adaptive_avg_pool2d(h_2d, pool_size)
            w = dw(pooled)
            up = F.interpolate(pooled * w, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(up)

        fused = self.fuse_conv(torch.cat(outs, dim=1))
        out = fused.reshape(B, -1, H * W).transpose(1, 2)
        return self.up(out)

class TransformerBlockWithMAA(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio), nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.maa = MAA(dim)

    def forward(self, x):

        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.maa(x)
        x = x + self.mlp(self.norm2(x))
        return x

class GFEM(nn.Module):

    def __init__(self, img_size=352, patch_size=8, in_ch=3, dim=256, depth=4, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_ch, dim, patch_size, stride=patch_size)
        self.pos_embed   = nn.Parameter(torch.zeros(1, num_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.Sequential(*[
            TransformerBlockWithMAA(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :x.shape[1], :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)
        return x

class CrossBranchAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, global_feat, local_feat):

        out, _ = self.attn(
            query=self.norm(global_feat),
            key=self.norm(local_feat),
            value=self.norm(local_feat)
        )
        return global_feat + out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid()
        )

    def forward(self, x, epoch_scale=1.0):
        return x * (self.fc(x) * epoch_scale)

class FFM(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.cba  = CrossBranchAttention(dim)
        self.se   = SEBlock(dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, global_feat, local_feat, epoch_scale=1.0):
        B, C, H, W = global_feat.shape
        g = global_feat.flatten(2).transpose(1, 2)
        l = local_feat.flatten(2).transpose(1, 2)

        fused = self.cba(g, l)
        fused = fused.transpose(1, 2).reshape(B, C, H, W)
        fused = self.se(fused, epoch_scale)
        return fused

class EMSAM(nn.Module):


    def __init__(self, in_channels=3, num_classes=1, base_ch=64,
                 img_size=352, patch_size=8, transformer_dim=256, transformer_depth=4):
        super().__init__()

        self.lfem = LFEM(in_channels, base_ch)
        local_dim = base_ch * 4

        self.gfem = GFEM(img_size, patch_size, in_channels,
                         dim=transformer_dim, depth=transformer_depth)

        self.local_align  = nn.Conv2d(local_dim, transformer_dim, 1)
        self.global_align = nn.Conv2d(transformer_dim, transformer_dim, 1)

        self.ffm = FFM(transformer_dim)

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, base_ch * 4, 2, stride=2),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(base_ch, num_classes, 1)

        self._current_epoch = 0

    def set_epoch(self, epoch, total_epochs=200):
        """设置当前 epoch，用于 SE Block 的动态权重"""
        import math
        beta = 0.2
        self._epoch_scale = 1 - math.exp(-beta * epoch)

    def forward(self, x):
        B, C, H, W = x.shape

        local_feat = self.lfem(x)

        global_feat = self.gfem(x)

        target_h = H // 8
        target_w = W // 8
        if global_feat.shape[2:] != (target_h, target_w):
            global_feat = F.interpolate(global_feat, size=(target_h, target_w),
                                        mode='bilinear', align_corners=False)

        local_feat  = self.local_align(local_feat)
        global_feat = self.global_align(global_feat)

        epoch_scale = getattr(self, '_epoch_scale', 1.0)
        fused = self.ffm(global_feat, local_feat, epoch_scale)

        d3 = self.dec3(fused)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        out = self.head(d1)
        return out

if __name__ == "__main__":
    model = EMSAM(num_classes=1, img_size=352)
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} ({total*4/1024**2:.1f} MB)")
