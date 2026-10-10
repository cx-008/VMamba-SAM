

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("[LightM-UNet] mamba_ssm not installed. Using Conv fallback.")
    print("Install: pip install causal-conv1d mamba-ssm")

class ResidualVisionMambaLayer(nn.Module):

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:

            self.mamba = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1),
            )

    def forward(self, x):
   
        B, C, H, W = x.shape
        shortcut = x

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        if MAMBA_AVAILABLE:
            x = self.mamba(x)
        else:

            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.mamba(x)
            x = x.flatten(2).transpose(1, 2)

        x = x.transpose(1, 2).reshape(B, C, H, W)

        return x + shortcut

class EncoderBlock(nn.Module):
  
    def __init__(self, in_channels, out_channels, d_state=16):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.rvm = ResidualVisionMambaLayer(out_channels, d_state=d_state)

    def forward(self, x):
        x = self.downsample(x)
        x = self.rvm(x)
        return x

class DecoderBlock(nn.Module):
   
    def __init__(self, in_channels, skip_channels, out_channels, d_state=16):
        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2
        )

        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)

        self.rvm = ResidualVisionMambaLayer(out_channels, d_state=d_state)

    def forward(self, x, skip):

        x = self.upsample(x)
        skip = self.skip_conv(skip)
        x = x + skip
        x = self.rvm(x)
        return x

class LightMUNet(nn.Module):

    def __init__(self, in_channels=3, num_classes=1,
                 base_channels=32, d_state=16):
        super().__init__()

        self.num_classes = num_classes
        c = base_channels

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, padding=1),
            nn.BatchNorm2d(c),
            nn.GELU(),
        )
        self.stem_rvm = ResidualVisionMambaLayer(c, d_state=d_state)

        self.enc1 = EncoderBlock(c, c*2, d_state)
        self.enc2 = EncoderBlock(c*2, c*4, d_state)
        self.enc3 = EncoderBlock(c*4, c*8, d_state)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(c*8, c*8, 3, padding=1),
            nn.BatchNorm2d(c*8),
            nn.GELU(),
            ResidualVisionMambaLayer(c*8, d_state=d_state),
        )

        self.dec1 = DecoderBlock(c*8, c*4, c*4, d_state)
        self.dec2 = DecoderBlock(c*4, c*2, c*2, d_state)
        self.dec3 = DecoderBlock(c*2, c, c, d_state)

        self.head = nn.Conv2d(c, num_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):


        x0 = self.stem(x)
        x0 = self.stem_rvm(x0)

        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        x_bottleneck = self.bottleneck(x3)

        x = self.dec1(x_bottleneck, x2)
        x = self.dec2(x, x1)
        x = self.dec3(x, x0)

        x = self.head(x)

        return x

if __name__ == "__main__":
    print("=" * 70)
    print("LightM-UNet: Lightweight Mamba-UNet Test")
    print("=" * 70)

    model = LightMUNet(in_channels=3, num_classes=1, base_channels=32, d_state=16)
    x = torch.randn(2, 3, 352, 352)

    print(f"Input shape : {x.shape}")
    print(f"Mamba available: {MAMBA_AVAILABLE}")
    print()

    with torch.no_grad():
        out = model(x)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Output shape: {out.shape}")
    print(f"Parameters  : {params:,} ({params/1e6:.2f}M)")
    print(f"Trainable   : {trainable:,} ({trainable/1e6:.2f}M)")
    print()

