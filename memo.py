import sys
import time
import os
import datetime
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
from data import LaidDataset, RandomPatchDataset
from torchvision.transforms import ToTensor, Normalize, Compose
Image.MAX_IMAGE_PIXELS = 1000000000


tensor = torch.load('./mask.pt')


COLOR_MAP = np.array([
    [   0,   0,   0,   0], # 0 -> transparent
    [   0,   0,   0, 255], # 1 -> black
    [   0,   0, 255, 255], # 2 -> blue
])

def restore_mask(tensor, dims=None):
    arr = np.transpose(tensor.numpy(), (1, 2, 0))
    arr = np.argmax(arr, axis=2)
    h, w = arr.shape
    arr = COLOR_MAP[arr]
    if dims:
        arr = arr[dims[1]:dims[3],dims[0]:dims[2]]
    print(arr.shape)
    return Image.fromarray(np.uint8(arr))

i = restore_mask(tensor, [100,0,110,200])
i.save('p.png')
