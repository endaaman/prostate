from torch import nn
from torch.nn import functional as F
from torchvision import models
import torch
import torchvision


class ConvRelu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                        mode=self.mode, align_corners=self.align_corners)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_size, mid_size, out_size, kernel_size=None):
        super().__init__()
        if kernel_size:
            modules = [
                ConvRelu(in_size, mid_size),
                nn.ConvTranspose2d(mid_size, out_size, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1),
            ]
        else:
            modules = [
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_size, mid_size),
                nn.Conv2d(mid_size, out_size, 3, padding=1),
            ]
        modules.append(nn.BatchNorm2d(out_size))
        modules.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes, upsample=False):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg11_bn(pretrained=True).features
        ks = 2 if not upsample else None
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu)
        self.conv2 = nn.Sequential(e[4], e[5], self.relu)
        self.conv3 = nn.Sequential(e[8], e[9], self.relu, e[11], e[12])
        self.conv4 = nn.Sequential(e[15], e[16], self.relu, e[18], e[19])
        self.conv5 = nn.Sequential(e[22], e[23], self.relu, e[25], e[26])
        self.center = DecoderBlock(512, 512, 256, ks)
        self.dec5 = DecoderBlock(768, 512, 256, ks)
        self.dec4 = DecoderBlock(768, 512, 128, ks)
        self.dec3 = DecoderBlock(384, 256, 64, ks)
        self.dec2 = DecoderBlock(192, 128, 32, ks)
        self.dec1 = ConvRelu(96, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        # return torch.sigmoid(x_out)
        return self.softmax(x_out)


class UNet16(nn.Module):
    def __init__(self, num_classes, upsample=True):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg16_bn(pretrained=True).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu, e[3], e[4], self.relu)
        self.conv2 = nn.Sequential(e[7], e[8], self.relu, e[10], e[11], self.relu)
        self.conv3 = nn.Sequential(e[14], e[15], self.relu, e[17], e[18], self.relu, e[20], e[21], self.relu)
        self.conv4 = nn.Sequential(e[24], e[25], self.relu, e[27], e[28], self.relu, e[30], e[31], self.relu)
        self.conv5 = nn.Sequential(e[34], e[35], self.relu, e[37], e[38], self.relu, e[40], e[41], self.relu)
        self.center = DecoderBlock(512, 512, 256, 4)
        self.dec5 = DecoderBlock(768, 512, 256, 2)
        self.dec4 = DecoderBlock(768, 512, 256, 2)
        self.dec3 = DecoderBlock(512, 256, 64, 2)
        self.dec2 = DecoderBlock(192, 128, 32, 2)
        self.dec1 = ConvRelu(96, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))
        center = self.center(self.pool(conv5))
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        x_out = self.final(dec1)
        # return torch.sigmoid(x_out)
        return self.softmax(x_out)


class DecoderBlock2(nn.Module):
    def __init__(self, in_size, out_size, mid_size=None):
        super().__init__()
        mid_size = mid_size or in_size
        self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_size, mid_size),
                ConvRelu(mid_size, out_size))

    def forward(self, x):
        return self.block(x)


class FeaturePyramidBlock(nn.Module):
    def __init__(self, in_size, scale):
        super().__init__()
        self.block = nn.Sequential(
                ConvRelu(in_size, 64),
                Interpolate(scale_factor=scale, mode='bilinear'))

    def forward(self, x):
        return self.block(x)


class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        base = models.resnet34(pretrained=True)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.dropout = nn.Dropout2d()
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4
        self.dec1 = DecoderBlock2(512, 256)
        self.dec2 = DecoderBlock2(512, 128, 256)
        self.dec3 = DecoderBlock2(256, 64, 128)
        self.dec4 = DecoderBlock2(128, 64)
        self.fpn1 = FeaturePyramidBlock(64, 1)
        self.fpn2 = FeaturePyramidBlock(128, 2)
        self.fpn3 = FeaturePyramidBlock(256, 4)
        self.fpn4 = FeaturePyramidBlock(512, 8)
        self.final = nn.Sequential(
                DecoderBlock2(256, 32, 128),
                ConvRelu(32, 32),
                ConvRelu(32, num_classes, 1),
                )
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        center = self.dec1(e4)
        d1 = torch.cat([center, e3], 1)
        d2 = torch.cat([self.dec2(d1), e2], 1)
        d3 = torch.cat([self.dec3(d2), e1], 1)
        d4 = self.dec4(d3)
        p1 = self.fpn1(d4)
        p2 = self.fpn2(d3)
        p3 = self.fpn3(d2)
        p4 = self.fpn4(d1)
        y = torch.cat([p1, p2, p3, p4], 1)
        y = F.dropout2d(y, 0.5, training=self.training)
        y = self.final(y)
        return self.softmax(y)


DefaultNet = UNet16
if __name__ == '__main__':
    input_tensor = torch.rand(1, 3, 224, 224)
    model = DefaultNet(num_classes=5)
    print('params count: {:,}'.format(sum(p.numel() for p in model.parameters())))
    with torch.no_grad():
        output_tensor = model(input_tensor)
        print(output_tensor.size())
