
import torch
import torch.nn as nn
import torch.nn.functional as F

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )

class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):
 
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1, bias=False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):


        g1 = self.W_g(g)
        if g1.shape[2:] != x.shape[2:]:
            g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=False)

        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        feat = self.conv(x)
        return feat, self.pool(feat)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.ag = AttentionGate(F_g=out_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = double_conv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        skip = self.ag(g=x, x=skip)
        return self.conv(torch.cat([x, skip], dim=1))

class AttentionUNet(nn.Module):


    def __init__(self, in_channels=3, num_classes=1, base_ch=64):
        super().__init__()

        self.enc1 = EncoderBlock(in_channels, base_ch)
        self.enc2 = EncoderBlock(base_ch,     base_ch * 2)
        self.enc3 = EncoderBlock(base_ch * 2, base_ch * 4)
        self.enc4 = EncoderBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = double_conv(base_ch * 8, base_ch * 16)

        self.dec4 = DecoderBlock(base_ch * 16, base_ch * 8,  base_ch * 8)
        self.dec3 = DecoderBlock(base_ch * 8,  base_ch * 4,  base_ch * 4)
        self.dec2 = DecoderBlock(base_ch * 4,  base_ch * 2,  base_ch * 2)
        self.dec1 = DecoderBlock(base_ch * 2,  base_ch,      base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):

        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        x = self.bottleneck(x)

        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)

if __name__ == "__main__":
    model = AttentionUNet(num_classes=1)
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} ({total * 4 / 1024 ** 2:.1f} MB)")
