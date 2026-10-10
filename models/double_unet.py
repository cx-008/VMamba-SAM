

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn_relu(in_ch, out_ch, k=3, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
    )

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        conv_bn_relu(in_ch, out_ch),
        conv_bn_relu(out_ch, out_ch),
    )

class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling
    4路并行：1×1 / d=6 / d=12 / d=18 + 全局平均池化
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = out_ch // 5

        self.b0 = conv_bn_relu(in_ch, mid, 1, p=0)
        self.b1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=6,  dilation=6,  bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.global_avg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fuse = conv_bn_relu(mid * 5, out_ch, 1, p=0)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        b0 = self.b0(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        bg = F.interpolate(self.global_avg(x), size=(H, W), mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([b0, b1, b2, b3, bg], dim=1))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // reduction, 4), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 4), channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class Encoder(nn.Module):
    def __init__(self, in_ch, base_ch):
        super().__init__()
        self.e1 = double_conv(in_ch,      base_ch)
        self.e2 = double_conv(base_ch,    base_ch * 2)
        self.e3 = double_conv(base_ch*2,  base_ch * 4)
        self.e4 = double_conv(base_ch*4,  base_ch * 8)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        return s1, s2, s3, s4, self.pool(s4)

class Decoder(nn.Module):
    def __init__(self, base_ch):
        super().__init__()
        self.up4 = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, stride=2)
        self.d4  = double_conv(base_ch * 16, base_ch * 8)
        self.se4 = SEBlock(base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.d3  = double_conv(base_ch * 8,  base_ch * 4)
        self.se3 = SEBlock(base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.d2  = double_conv(base_ch * 4,  base_ch * 2)
        self.se2 = SEBlock(base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.d1  = double_conv(base_ch * 2,  base_ch)
        self.se1 = SEBlock(base_ch)

    def _align(self, x, ref):
        if x.shape[2:] != ref.shape[2:]:
            x = F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
        return x

    def forward(self, bottleneck, s1, s2, s3, s4):
        x = self._align(self.up4(bottleneck), s4)
        x = self.se4(self.d4(torch.cat([x, s4], dim=1)))

        x = self._align(self.up3(x), s3)
        x = self.se3(self.d3(torch.cat([x, s3], dim=1)))

        x = self._align(self.up2(x), s2)
        x = self.se2(self.d2(torch.cat([x, s2], dim=1)))

        x = self._align(self.up1(x), s1)
        x = self.se1(self.d1(torch.cat([x, s1], dim=1)))
        return x

class DoubleUNet(nn.Module):
    """
    DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation
    论文：Jha et al., CBMS 2020
    https://arxiv.org/abs/2006.04868

    农业应用：植物病害精细分割、果实检测
    特点：双网络级联，第二网络聚焦第一网络的预测区域
    """

    def __init__(self, in_channels=3, num_classes=1, base_ch=32):
        super().__init__()

        self.enc1 = Encoder(in_channels, base_ch)
        self.aspp1 = ASPP(base_ch * 8, base_ch * 8)
        self.dec1 = Decoder(base_ch)
        self.head1 = nn.Conv2d(base_ch, num_classes, 1)

        self.enc2 = Encoder(in_channels, base_ch)
        self.aspp2 = ASPP(base_ch * 8, base_ch * 8)
        self.dec2 = Decoder(base_ch)
        self.head2 = nn.Conv2d(base_ch, num_classes, 1)

    def forward(self, x):

        s1_1, s1_2, s1_3, s1_4, pool1 = self.enc1(x)
        b1 = self.aspp1(pool1)
        d1 = self.dec1(b1, s1_1, s1_2, s1_3, s1_4)
        mask1 = self.head1(d1)

        x2 = x * torch.sigmoid(mask1)
        s2_1, s2_2, s2_3, s2_4, pool2 = self.enc2(x2)
        b2 = self.aspp2(pool2)
        d2 = self.dec2(b2, s2_1, s2_2, s2_3, s2_4)
        mask2 = self.head2(d2)

        return mask1 + mask2

if __name__ == "__main__":
    model = DoubleUNet(num_classes=1)
    x = torch.randn(2, 3, 352, 352)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,} ({total * 4 / 1024 ** 2:.1f} MB)")
