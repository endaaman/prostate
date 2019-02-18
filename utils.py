import datetime
import torch

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def dice_coef(a, b, smooth = 1.):
    a = a.view(-1)
    b = b.view(-1)
    intersection = (a * b).sum()
    return ((2. * intersection + smooth) / (a.sum() + b.sum() + smooth)).item()

def argmax_acc(a, b):
    num_classes = a.size(1)
    num_channels = a.nelement()
    _, a = torch.max(a.permute(0, 2, 3, 1).contiguous().view(-1, num_classes), 1)
    _, b = torch.max(b.permute(0, 2, 3, 1).contiguous().view(-1, num_classes), 1)
    return (a == b).sum().item() / (num_channels // num_classes)
