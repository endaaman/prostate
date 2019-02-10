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

ds = RandomPatchDataset()
print('data loaded')
for i, (x_arr, y_arr) in enumerate(ds):
    print(i)
    Image.fromarray(x_arr).save(f'out/{i}_x.jpg')
    Image.fromarray(y_arr).save(f'out/{i}_y.png')
    if i > 5:
        break
