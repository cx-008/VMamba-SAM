

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[GARDEN] mamba_ssm 未安装，BASS 模块将使用卷积替代")

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class MSCA(nn.Module):
    def __init__(self, channels, pool_sizes=(1, 2, 6)):
        super().__init__()
        self.pool_sizes = pool_sizes
        mid = channels // 4

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(s),
                nn.Conv2d(channels, mid, 1), nn.ReLU(inplace=True)
            ) for s in pool_sizes
        ])

        self.fuse = nn.Sequential(
            nn.Conv2d(channels + len(pool_sizes) * mid, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        outs = [x]
        for branch in self.branches:
            out = branch(x)
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(out)
        return self.fuse(torch.cat(outs, dim=1))

class GGBP(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2

        self.grad_x = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False)
        self.grad_y = nn.Conv2d(channels, channels, kernel_size, padding=pad, groups=channels, bias=False)

        self._init_sobel(kernel_size)

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, 1, 1),
            nn.Sigmoid()
        )

    def _init_sobel(self, k):
        with torch.no_grad():
            if k == 3:
                sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
                sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            else:
                sx = torch.zeros(5, 5)
                sx[2, 0], sx[2, 4] = -1, 1
                sy = sx.T
            for conv, s in [(self.grad_x, sx), (self.grad_y, sy)]:
                w = s.view(1, 1, k, k).expand(conv.weight.shape[0], -1, -1, -1)
                conv.weight.copy_(w)

    def forward(self, x):
        gx = self.grad_x(x)
        gy = self.grad_y(x)
        return self.fuse(torch.cat([gx, gy], dim=1))

class BASS(nn.Module):
    def __init__(self, channels, d_state=16, sharpening=1):
        super().__init__()
        self.channels = channels
        self.sharpening = sharpening

        self.ggbp = GGBP(channels)

        self.in_proj = nn.Conv2d(channels, channels, 1)

        if MAMBA_AVAILABLE:

            self.mamba_h_fwd = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=1)
            self.mamba_h_bwd = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=1)
            self.mamba_v_fwd = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=1)
            self.mamba_v_bwd = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=1)
        else:

            self.fallback = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
                nn.Conv2d(channels, channels, 1),
                nn.GELU()
            )

        self.gamma = nn.Parameter(torch.ones(1))
        self.beta  = nn.Parameter(torch.zeros(1))

        self.out_proj = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels)
        )

    def _scan_4dir(self, x_2d):
      
        B, C, H, W = x_2d.shape
        x_flat = x_2d.reshape(B, C, H * W).transpose(1, 2)

        y0 = self.mamba_h_fwd(x_flat)
        y1 = torch.flip(self.mamba_h_bwd(torch.flip(x_flat, [1])), [1])

        x_vt = x_2d.permute(0, 1, 3, 2).reshape(B, C, H * W).transpose(1, 2)
        y2_raw = self.mamba_v_fwd(x_vt)
        y2 = y2_raw.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        y2 = y2.reshape(B, C, H * W).transpose(1, 2)

        y3_raw = torch.flip(self.mamba_v_bwd(torch.flip(x_vt, [1])), [1])
        y3 = y3_raw.transpose(1, 2).reshape(B, C, W, H).permute(0, 1, 3, 2)
        y3 = y3.reshape(B, C, H * W).transpose(1, 2)

        y = (y0 + y1 + y2 + y3) / 4.0
        return y.transpose(1, 2).reshape(B, C, H, W)

    def forward(self, x):
        B, C, H, W = x.shape
        identity = x

        boundary = self.ggbp(x)

        x_proj = self.in_proj(x)

        if MAMBA_AVAILABLE:

            scanned = self._scan_4dir(x_proj)
        else:
            scanned = self.fallback(x_proj)

        gate = torch.sigmoid(self.gamma * boundary.pow(self.sharpening) + self.beta)

        out = gate * scanned + (1 - gate) * x_proj

        out = self.out_proj(out)
        return F.relu(out + identity)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(self.avg(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, d_state=16):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.cca = ChannelAttention(skip_ch)
        self.conv = double_conv(out_ch + skip_ch, out_ch)
        self.bass = BASS(out_ch, d_state=d_state)

    def forward(self, x, skip):
        x = self.up(x)
        skip = self.cca(skip)
        x = self.conv(torch.cat([x, skip], dim=1))
        x = self.bass(x)
        return x

class GARDEN(nn.Module):


    def __init__(self, in_channels=3, num_classes=1, base_ch=64, d_state=16):
        super().__init__()

        self.enc0 = double_conv(in_channels, base_ch)
        self.enc1 = DownBlock(base_ch,     base_ch * 2)
        self.enc2 = DownBlock(base_ch * 2, base_ch * 4)
        self.enc3 = DownBlock(base_ch * 4, base_ch * 8)
        self.enc4 = DownBlock(base_ch * 8, base_ch * 16)

        self.msca = MSCA(base_ch * 16, pool_sizes=(1, 2, 6))

        self.up4 = UpBlock(base_ch * 16, base_ch * 8,  base_ch * 8,  d_state)
        self.up3 = UpBlock(base_ch * 8,  base_ch * 4,  base_ch * 4,  d_state)
        self.up2 = UpBlock(base_ch * 4,  base_ch * 2,  base_ch * 2,  d_state)
        self.up1 = UpBlock(base_ch * 2,  base_ch,      base_ch,      d_state)

        self.head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):

        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.msca(e4)

        d4 = self.up4(b,  e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d1 = self.up1(d2, e0)

        return self.head(d1)

if __name__ == "__main__":
    model = GARDEN(num_classes=1)
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} ({total*4/1024**2:.1f} MB)")
