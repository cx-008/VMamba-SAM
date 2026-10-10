"""
EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation

Reference:
    Md Mostafijur Rahman et al., "EMCAD: Efficient Multi-scale Convolutional Attention
    Decoding for Medical Image Segmentation"
    IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2024
    arXiv:2405.06880
    https://github.com/SLDGroup/EMCAD

Key Features:
    - Efficient decoder with only 1.91M parameters and 0.381G FLOPs
    - 79.4% reduction in parameters and 80.3% reduction in FLOPs vs. state-of-the-art
    - Multi-scale convolutional attention module (MSCAM)
    - Efficient up-convolution blocks (EUCB)
    - Large-kernel grouped attention gates (LGAG)
    - State-of-the-art performance across 12 medical image segmentation datasets

Architecture:
    - Works with any hierarchical encoder (CNN or Transformer)
    - EMCAD decoder with multi-scale attention
    - Cascaded feature fusion
    - Efficient design using group and depth-wise convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleDepthwiseConv(nn.Module):
    """
    Multi-scale parallel depth-wise convolution
    Captures features at different scales efficiently
    """
    def __init__(self, channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(channels, channels, k, padding=k//2, groups=channels, bias=False)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(channels * len(kernel_sizes))
        self.fusion = nn.Conv2d(channels * len(kernel_sizes), channels, 1, bias=False)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.bn(out)
        out = self.fusion(out)
        return out

class ChannelAttention(nn.Module):
    """Channel attention using global pooling"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

class SpatialAttention(nn.Module):
    """Spatial attention using channel pooling"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return x * out

class LargeKernelGroupedAttentionGate(nn.Module):
    """
    Attention gate with large kernel and grouped convolution
    Efficiently captures spatial relationships
    """
    def __init__(self, F_g, F_l, F_int, kernel_size=7, groups=4):
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
            nn.Conv2d(F_int, 1, kernel_size, padding=kernel_size//2, groups=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: gating signal from coarser scale
        x: feature from encoder (skip connection)
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolution block with parallel branches
    """
    def __init__(self, channels):
        super().__init__()
        self.msdc = MultiScaleDepthwiseConv(channels)
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.msdc(x)
        out = self.conv(out)
        return out + x

class MultiScaleConvAttentionModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mscb = MultiScaleConvBlock(channels)
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.mscb(x)
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class EfficientUpConvBlock(nn.Module):

    def __init__(self, in_ch, skip_ch, out_ch, use_attention_gate=True):
        super().__init__()
        self.use_attention_gate = use_attention_gate

        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bn_up = nn.BatchNorm2d(out_ch)

        if use_attention_gate:
            self.att_gate = LargeKernelGroupedAttentionGate(out_ch, skip_ch, out_ch // 2)

        self.conv_fusion = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.mscam = MultiScaleConvAttentionModule(out_ch)

    def forward(self, x, skip):

        x = self.up(x)
        x = self.bn_up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)

        if self.use_attention_gate:
            skip = self.att_gate(x, skip)

        x = torch.cat([x, skip], dim=1)
        x = self.conv_fusion(x)

        x = self.mscam(x)

        return x

class SimpleEncoder(nn.Module):

    def __init__(self, in_channels=3, base_ch=32):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True)
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 2),
            nn.ReLU(inplace=True)
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 4),
            nn.ReLU(inplace=True)
        )

        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 4, base_ch * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 8, base_ch * 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 8),
            nn.ReLU(inplace=True)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_ch * 8, base_ch * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 16, base_ch * 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch * 16),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        return e1, e2, e3, e4, b

class EMCAD(nn.Module):


    def __init__(
        self,
        in_channels=3,
        num_classes=1,
        base_ch=32,
        use_attention_gates=True
    ):
        super().__init__()

        self.encoder = SimpleEncoder(in_channels, base_ch)

        self.dec4 = EfficientUpConvBlock(base_ch * 16, base_ch * 8, base_ch * 8, use_attention_gates)
        self.dec3 = EfficientUpConvBlock(base_ch * 8, base_ch * 4, base_ch * 4, use_attention_gates)
        self.dec2 = EfficientUpConvBlock(base_ch * 4, base_ch * 2, base_ch * 2, use_attention_gates)
        self.dec1 = EfficientUpConvBlock(base_ch * 2, base_ch, base_ch, use_attention_gates)

        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, num_classes, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, num_classes, H, W)
        """

        e1, e2, e3, e4, b = self.encoder(x)

        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        out = self.head(d1)

        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out

if __name__ == '__main__':

    model = EMCAD(in_channels=3, num_classes=1, base_ch=32)
    x = torch.randn(2, 3, 352, 352)

    with torch.no_grad():
        y = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")

    from thop import profile
    try:
        flops, params = profile(model, inputs=(x,))
        print(f"FLOPs: {flops / 1e9:.2f}G")
    except:
        print("Install thop to calculate FLOPs: pip install thop")
