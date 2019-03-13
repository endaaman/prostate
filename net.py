from torch import nn
from torch.nn import functional as F
from torchvision import models
import torch
import torchvision


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = nn.Conv2d(in_, out, 3, padding=1)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Interpolate, self).__init__()
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
    def __init__(self, in_, mid, out, kernel_size=None, bn=True):
        super(DecoderBlock, self).__init__()
        if kernel_size:
            modules = [
                ConvRelu(in_, mid),
                nn.ConvTranspose2d(mid, out, kernel_size=kernel_size, stride=2, padding=kernel_size//2-1),
            ]
        else:
            modules = [
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_, mid),
                nn.Conv2d(mid, out, 3, padding=1),
            ]
        if bn:
            modules.append(nn.BatchNorm2d(out))
        modules.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_filters=32):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg11_bn(pretrained=pretrained).features
        nf = num_filters
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu)
        self.conv2 = nn.Sequential(e[4], e[5], self.relu)
        self.conv3 = nn.Sequential(e[8], e[9], self.relu, e[11], e[12])
        self.conv4 = nn.Sequential(e[15], e[16], self.relu, e[18], e[19])
        self.conv5 = nn.Sequential(e[22], e[23], self.relu, e[25], e[26])
        self.center = DecoderBlock(nf * 8 * 2, nf * 8 * 2, nf * 8, 2)
        self.dec5 = DecoderBlock(nf * (16 + 8), nf * 8 * 2, nf * 8, 2)
        self.dec4 = DecoderBlock(nf * (16 + 8), nf * 8 * 2, nf * 4, 2)
        self.dec3 = DecoderBlock(nf * (8 + 4), nf * 4 * 2, nf * 2, 2)
        self.dec2 = DecoderBlock(nf * (4 + 2), nf * 2 * 2, nf, 2)
        self.dec1 = ConvRelu(nf * (2 + 1), nf)
        self.final = nn.Conv2d(nf, num_classes, kernel_size=1)

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
        return torch.sigmoid(x_out)


class UNet16(nn.Module):
    def __init__(self, num_classes, num_filters=32, pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        nf = num_filters
        e = models.vgg16_bn(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu, e[3], e[4], self.relu)
        self.conv2 = nn.Sequential(e[7], e[8], self.relu, e[10], e[11], self.relu)
        self.conv3 = nn.Sequential(e[14], e[15], self.relu, e[17], e[18], self.relu, e[20], e[21], self.relu)
        self.conv4 = nn.Sequential(e[24], e[25], self.relu, e[27], e[28], self.relu, e[30], e[31], self.relu)
        self.conv5 = nn.Sequential(e[34], e[35], self.relu, e[37], e[38], self.relu, e[40], e[41], self.relu)
        self.center = DecoderBlock(512, nf * 8 * 2, nf * 8, kernel_size=4)
        self.dec5 = DecoderBlock(512 + nf * 8, nf * 8 * 2, nf * 8, 2)
        self.dec4 = DecoderBlock(512 + nf * 8, nf * 8 * 2, nf * 8, 2)
        self.dec3 = DecoderBlock(256 + nf * 8, nf * 4 * 2, nf * 2, 2)
        self.dec2 = DecoderBlock(128 + nf * 2, nf * 2 * 2, nf, 2)
        self.dec1 = ConvRelu(64 + nf, nf)
        self.final = nn.Conv2d(nf, num_classes, kernel_size=1)

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
        return torch.sigmoid(x_out)


if __name__ == '__main__':
    input_tensor = torch.rand(1,  3, 224, 224)
    unet11 = UNet11(num_classes=4)
    unet16 = UNet16(num_classes=4)
    with torch.no_grad():
        output_tensor = unet11(input_tensor)
        print(output_tensor.size())
        output16output_tensor = unet16(input_tensor)
        print(output_tensor.size())
