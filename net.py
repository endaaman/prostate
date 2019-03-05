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


class DecoderBlock(nn.Module):
    def __init__(self, in_, mid, out, bn=False):
        super().__init__()
        modules = [
            ConvRelu(in_, mid),
            nn.ConvTranspose2d(mid, out, kernel_size=3, stride=2, padding=1, output_padding=1),
        ]
        if bn:
            modules.append(nn.BatchNorm2d(out))
        modules.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules)
    def forward(self, x):
        return self.block(x)


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


class DecoderBlockV2(nn.Module):
    def __init__(self, in_, mid, out, bn=False, is_deconv=True,):
        super(DecoderBlockV2, self).__init__()
        self.in_ = in_
        if is_deconv:
            modules = [
                ConvRelu(in_, mid),
                nn.ConvTranspose2d(mid, out, kernel_size=4, stride=2, padding=1),
            ]
            if bn:
                modules.append(nn.BatchNorm2d(out))
            modules.append(nn.ReLU(inplace=True))
            self.block = nn.Sequential(*modules)
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode='bilinear'),
                ConvRelu(in_, mid),
                ConvRelu(mid, out),
            )

    def forward(self, x):
        return self.block(x)


class UNet11(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_filters=32):
        super().__init__()
        self.num_classes = num_classes
        nf = num_filters
        e = models.vgg11(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], self.relu)
        self.conv2 = nn.Sequential(e[3], self.relu)
        self.conv3 = nn.Sequential(e[6], self.relu, e[8])
        self.conv4 = nn.Sequential(e[11], self.relu, e[13])
        self.conv5 = nn.Sequential(e[16], self.relu, e[18])
        self.center = DecoderBlock(nf * 8 * 2, nf * 8 * 2, nf * 8)
        self.dec5 = DecoderBlock(nf * (16 + 8), nf * 8 * 2, nf * 8)
        self.dec4 = DecoderBlock(nf * (16 + 8), nf * 8 * 2, nf * 4)
        self.dec3 = DecoderBlock(nf * (8 + 4), nf * 4 * 2, nf * 2)
        self.dec2 = DecoderBlock(nf * (4 + 2), nf * 2 * 2, nf)
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
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        return torch.sigmoid(x_out)


class UNet16(nn.Module):
    def __init__(self, num_classes, num_filters=32, pretrained=True, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        nf = num_filters
        e = models.vgg16(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], self.relu, e[2], self.relu)
        self.conv2 = nn.Sequential(e[5], self.relu, e[7], self.relu)
        self.conv3 = nn.Sequential(e[10], self.relu, e[12], self.relu, e[14], self.relu)
        self.conv4 = nn.Sequential(e[17], self.relu, e[19], self.relu, e[21], self.relu)
        self.conv5 = nn.Sequential(e[24], self.relu, e[26], self.relu, e[28], self.relu)
        self.center = DecoderBlockV2(512, nf * 8 * 2, nf * 8, is_deconv)
        self.dec5 = DecoderBlockV2(512 + nf * 8, nf * 8 * 2, nf * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + nf * 8, nf * 8 * 2, nf * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + nf * 8, nf * 4 * 2, nf * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + nf * 2, nf * 2 * 2, nf, is_deconv)
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
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        return torch.sigmoid(x_out)


class UNet11bn(nn.Module):
    def __init__(self, num_classes, pretrained=True, num_filters=32):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg11_bn(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu)
        self.conv2 = nn.Sequential(e[4], e[5], self.relu)
        self.conv3 = nn.Sequential(e[8], e[9], self.relu, e[11], e[12])
        self.conv4 = nn.Sequential(e[15], e[16], self.relu, e[18], e[19])
        self.conv5 = nn.Sequential(e[22], e[23], self.relu, e[25], e[26])
        self.center = DecoderBlock(num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8, bn=True)
        self.dec5 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8, bn=True)
        self.dec4 = DecoderBlock(num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4, bn=True)
        self.dec3 = DecoderBlock(num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2, bn=True)
        self.dec2 = DecoderBlock(num_filters * (4 + 2), num_filters * 2 * 2, num_filters, bn=True)
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

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
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        return torch.sigmoid(x_out)


class UNet16bn(nn.Module):
    def __init__(self, num_classes, num_filters=32, pretrained=True, is_deconv=False):
        super().__init__()
        self.num_classes = num_classes
        e = models.vgg16_bn(pretrained=pretrained).features
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Sequential(e[0], e[1], self.relu, e[3], e[4], self.relu)
        self.conv2 = nn.Sequential(e[7], e[8], self.relu, e[10], e[11], self.relu)
        self.conv3 = nn.Sequential(e[14], e[15], self.relu, e[17], e[18], self.relu, e[20], e[21], self.relu)
        self.conv4 = nn.Sequential(e[24], e[25], self.relu, e[27], e[28], self.relu, e[30], e[31], self.relu)
        self.conv5 = nn.Sequential(e[34], e[35], self.relu, e[37], e[38], self.relu, e[40], e[41], self.relu)
        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

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
        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)
        return torch.sigmoid(x_out)
