

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling) 模块"""

    def __init__(self, in_channels, output_stride):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18] if output_stride == 16 else [1, 12, 24, 36]

        self.aspp1 = self._make_branch(in_channels, 256, 1, dilations[0])
        self.aspp2 = self._make_branch(in_channels, 256, 3, dilations[1])
        self.aspp3 = self._make_branch(in_channels, 256, 3, dilations[2])
        self.aspp4 = self._make_branch(in_channels, 256, 3, dilations[3])

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 256, 1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.conv1 = nn.Conv2d(256 * 5, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

    def _make_branch(self, in_channels, out_channels, kernel_size, dilation):
        padding = 0 if kernel_size == 1 else dilation
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)

        global_feat = self.global_pool(x)
        x5 = F.interpolate(global_feat, size=x.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        return self.dropout(self.relu(x))

class Decoder(nn.Module):
    """DeepLabV3+ 解码器模块"""

    def __init__(self, low_level_channels, num_classes):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(low_level_channels, 48, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)

        self.output = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, low_level_features):
        low_level_features = self.relu(self.bn1(self.conv1(low_level_features)))
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((low_level_features, x), dim=1)
        return self.output(x)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetBackbone(nn.Module):


    def __init__(self, backbone='resnet50', pretrained=True, output_stride=16):
        super(ResNetBackbone, self).__init__()
        self.output_stride = output_stride

        try:

            if backbone == 'resnet50':
                self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
                self.out_channels = 2048
                self.low_level_channels = 256
            elif backbone == 'resnet101':
                self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
                self.out_channels = 2048
                self.low_level_channels = 256
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
                self.out_channels = 512
                self.low_level_channels = 64
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
        except AttributeError:

            if backbone == 'resnet50':
                self.backbone = models.resnet50(pretrained=pretrained)
                self.out_channels = 2048
                self.low_level_channels = 256
            elif backbone == 'resnet101':
                self.backbone = models.resnet101(pretrained=pretrained)
                self.out_channels = 2048
                self.low_level_channels = 256
            elif backbone == 'resnet34':
                self.backbone = models.resnet34(pretrained=pretrained)
                self.out_channels = 512
                self.low_level_channels = 64
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")

        self.conv1 = self.backbone.conv1
        self.bn1 = self.backbone.bn1
        self.relu = self.backbone.relu
        self.maxpool = self.backbone.maxpool
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        if output_stride == 8:

            self._make_dilated_conv(self.layer3, stride=1, dilation=2)
            self._make_dilated_conv(self.layer4, stride=1, dilation=4)
        elif output_stride == 16:

            self._make_dilated_conv(self.layer4, stride=1, dilation=2)

    def _make_dilated_conv(self, layer, stride=1, dilation=1):

        for m in layer.modules():
            if isinstance(m, nn.Conv2d):

                original_kernel_size = m.kernel_size[0]
                original_padding = m.padding[0]

                m.stride = (stride, stride)
                m.dilation = (dilation, dilation)

                new_padding = dilation * (original_kernel_size - 1) // 2
                m.padding = (new_padding, new_padding)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_features = x

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x, low_level_features

class DeepLabV3Plus(nn.Module):


    def __init__(self, num_classes=1, backbone='resnet50', output_stride=16, pretrained=True):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = ResNetBackbone(backbone=backbone, pretrained=pretrained, output_stride=output_stride)

        if backbone in ['resnet50', 'resnet101']:
            aspp_channels = 2048
            low_level_channels = 256
        elif backbone == 'resnet34':
            aspp_channels = 512
            low_level_channels = 64
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.aspp = ASPP(in_channels=aspp_channels, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels, num_classes)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        x, low_level_features = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_features)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x

if __name__ == "__main__":


    model_r50 = DeepLabV3Plus(num_classes=1, backbone='resnet50', pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    output = model_r50(x)
    print(f"DeepLabV3+ (ResNet-50): 输入 {x.shape} -> 输出 {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_r50.parameters()):,}")

    model_r101 = DeepLabV3Plus(num_classes=1, backbone='resnet101', pretrained=False)
    output = model_r101(x)
    print(f"DeepLabV3+ (ResNet-101): 输入 {x.shape} -> 输出 {output.shape}")
    print(f"模型参数量: {sum(p.numel() for p in model_r101.parameters()):,}")
