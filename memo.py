import sys
import time
import os
import datetime
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from data import LaidDataset, RandomPatchDataset
from torchvision.transforms import ToTensor, Normalize, Compose
Image.MAX_IMAGE_PIXELS = 1000000000


dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(dt)
exit(0)

ds = RandomPatchDataset(transform_x = ToTensor(), transform_y = ToTensor())
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4)
for i, (x, y) in enumerate(loader):
    pass
    # print(i)
    # y.save(f'./tmp/{i}.png')
    # x.save(f'./tmp/{i}.jpg')
    # if i > 10:
    #     exit(0)
print(i)
