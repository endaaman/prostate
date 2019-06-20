from torch import nn
from torch.nn import functional as F
from torchvision import models
import torch
import torchvision

from utils import curry


def pack(arr):
    return [x for x in arr if x]


class ConvRelu(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_size, out_size, kernel_size, padding=kernel_size//2)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBnRelu(ConvRelu):
    def __init__(self, in_size, out_size, kernel_size=3):
        super().__init__(in_size, out_size, kernel_size)
        self.bn = nn.BatchNorm2d(out_size),

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = False if mode == 'bilinear' else align_corners

    def forward(self, x):
        x = self.interp(x, size=self.size, scale_factor=self.scale_factor,
                mode=self.mode, align_corners=self.align_corners)
        return x


class DeconvDecoder(nn.Module):
    def __init__(self, in_size, mid_size, out_size, kernel_size=4, bn=True):
        super().__init__()
        self.block = nn.Sequential(*pack([
            ConvRelu(in_size, mid_size),
            nn.BatchNorm2d(mid_size) if bn else None,
            nn.ConvTranspose2d(mid_size, out_size, stride=2,
                kernel_size=kernel_size, padding=kernel_size//2-1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_size) if bn else None,
            ]))

    def forward(self, x):
        return self.block(x)


class UpsampleDecoder(nn.Module):
    def __init__(self, in_size, mid_size, out_size, bn=True, mode='nearest'):
        super().__init__()
        self.block = nn.Sequential(*pack([
            Interpolate(scale_factor=2, mode=mode),
            ConvRelu(in_size, mid_size),
            nn.BatchNorm2d(mid_size) if bn else None,
            ConvRelu(mid_size, out_size),
            nn.BatchNorm2d(out_size) if bn else None]))

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes, upsample=False):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg11_bn(pretrained=True).features
        decoder = curry(UpsampleDecoder, mode=upsample) if upsample else DeconvDecoder
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu)
        self.conv2 = nn.Sequential(e[4], e[5], self.relu)
        self.conv3 = nn.Sequential(e[8], e[9], self.relu, e[11], e[12])
        self.conv4 = nn.Sequential(e[15], e[16], self.relu, e[18], e[19])
        self.conv5 = nn.Sequential(e[22], e[23], self.relu, e[25], e[26])
        self.dec1 = decoder(512, 512, 256)
        self.dec2 = decoder(768, 512, 256)
        self.dec3 = decoder(768, 512, 128)
        self.dec4 = decoder(384, 256, 64)
        self.dec5 = decoder(192, 128, 32)
        self.dec6 = ConvRelu(96, 32)
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(self.pool(e1))
        e3 = self.conv3(self.pool(e2))
        e4 = self.conv4(self.pool(e3))
        e5 = self.conv5(self.pool(e4))
        center = self.pool(e5)
        d5 = self.dec1(center)
        d4 = self.dec2(torch.cat([d5, e5], 1))
        d3 = self.dec3(torch.cat([d4, e4], 1))
        d2 = self.dec4(torch.cat([d3, e3], 1))
        d1 = self.dec5(torch.cat([d2, e2], 1))
        out = self.dec6(torch.cat([d1, e1], 1))
        out = self.final(out)
        # return torch.sigmoid(out)
        return self.softmax(out)


class UNet11b(UNet11):
    def __init__(self, num_classes):
        super().__init__(num_classes, upsample='bilinear')

class UNet11n(UNet11):
    def __init__(self, num_classes):
        super().__init__(num_classes, upsample='nearest')


class UNet16(nn.Module):
    def __init__(self, num_classes, upsample=False):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg16_bn(pretrained=True).features
        decoder = curry(UpsampleDecoder, mode=upsample) if upsample else DeconvDecoder
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu, e[3], e[4], self.relu)
        self.conv2 = nn.Sequential(e[7], e[8], self.relu, e[10], e[11], self.relu)
        self.conv3 = nn.Sequential(e[14], e[15], self.relu, e[17], e[18], self.relu, e[20], e[21], self.relu)
        self.conv4 = nn.Sequential(e[24], e[25], self.relu, e[27], e[28], self.relu, e[30], e[31], self.relu)
        self.conv5 = nn.Sequential(e[34], e[35], self.relu, e[37], e[38], self.relu, e[40], e[41], self.relu)
        self.center = decoder(512, 512, 256)
        self.dec5 = decoder(768, 512, 256)
        self.dec4 = decoder(768, 512, 256)
        self.dec3 = decoder(512, 256, 64)
        self.dec2 = decoder(192, 128, 32)
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
        out = self.final(dec1)
        # return torch.sigmoid(out)
        return self.softmax(out)


class UNet16b(UNet16):
    def __init__(self, num_classes):
        super().__init__(num_classes, upsample='bilinear')

class UNet16n(UNet16):
    def __init__(self, num_classes):
        super().__init__(num_classes, upsample='bilinear')


class AlbuNet(nn.Module):
    def __init__(self, num_classes, upsample=False):
        super().__init__()
        self.num_classes = num_classes
        base = models.resnet34(pretrained=True)
        decoder = curry(UpsampleDecoder, mode=upsample) if upsample else DeconvDecoder
        self.pool = base.maxpool
        self.first = nn.Sequential(base.conv1, base.bn1, base.relu, self.pool)
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4
        self.dec1 = decoder(512, 512, 256)
        self.dec2 = decoder(768, 512, 256)
        self.dec3 = decoder(512, 512, 256)
        self.dec4 = decoder(384, 256, 64)
        self.dec5 = decoder(128, 128, 128)
        self.final = nn.Sequential(
                decoder(128, 128, 32),
                ConvRelu(32, 32),
                nn.Conv2d(32, num_classes, kernel_size=1))
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        e1 = self.first(x)
        e2 = self.enc1(e1)
        e3 = self.enc2(e2)
        e4 = self.enc3(e3)
        e5 = self.enc4(e4)
        d1 = self.dec1(self.pool(e5))
        d2 = self.dec2(torch.cat([d1, e5], 1))
        d3 = self.dec3(torch.cat([d2, e4], 1))
        d4 = self.dec4(torch.cat([d3, e3], 1))
        d5 = self.dec5(torch.cat([d4, e2], 1))
        out = self.final(d5)
        return self.softmax(out)


class AlbuNet_b(AlbuNet):
    def __init__(self, num_classes):
        super().__init__(num_classes, upsample='bilinear')

class AlbuNet_n(AlbuNet):
    def __init__(self, num_classes):
        super().__init__(num_classes, upsample='nearest')


class FeaturePyramidBlock(nn.Module):
    def __init__(self, in_size, scale):
        super().__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_size, 64, 3, padding=1),
                Interpolate(scale_factor=scale))

    def forward(self, x):
        return self.block(x)


class UResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        base = models.resnet34(pretrained=True)
        self.first = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.enc1 = base.layer1
        self.enc2 = base.layer2
        self.enc3 = base.layer3
        self.enc4 = base.layer4
        self.dec1 = UpsampleDecoder(512, 256, 256)
        self.dec2 = UpsampleDecoder(512, 128, 128)
        self.dec3 = UpsampleDecoder(256, 64, 64)
        self.dec4 = UpsampleDecoder(128, 64, 64)
        self.fpn1 = FeaturePyramidBlock(64, 1)
        self.fpn2 = FeaturePyramidBlock(128, 2)
        self.fpn3 = FeaturePyramidBlock(256, 4)
        self.fpn4 = FeaturePyramidBlock(512, 8)
        self.final = nn.Sequential(
                UpsampleDecoder(256, 64, 32),
                ConvRelu(32, 32),
                ConvRelu(32, num_classes, kernel_size=1))
        self.softmax = nn.Softmax2d()

    def forward(self, x):
        x = self.first(x)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d1 = torch.cat([self.dec1(e4), e3], 1)
        d2 = torch.cat([self.dec2(d1), e2], 1)
        d3 = torch.cat([self.dec3(d2), e1], 1)
        d4 = self.dec4(d3)
        p1 = self.fpn1(d4)
        p2 = self.fpn2(d3)
        p3 = self.fpn3(d2)
        p4 = self.fpn4(d1)
        out = torch.cat([p1, p2, p3, p4], 1)
        out = F.dropout2d(out, 0.3, training=self.training)
        out = self.final(out)
        return self.softmax(out)
        # return torch.sigmoid(out)


DefaultNet = UNet11
if __name__ == '__main__':
    input_tensor = torch.rand(1, 3, 512, 512)
    model = DefaultNet(num_classes=5)
    print('params count: {:,}'.format(sum(p.numel() for p in model.parameters())))
    with torch.no_grad():
        output_tensor = model(input_tensor)
        print(output_tensor.size())
