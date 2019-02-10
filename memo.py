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


INDEX_MAP = np.array([
    0, # empty
    1, # 000: black
    1, # R00: red
    1, # 0G0: green
    1, # RG0: yellow
    2, # 00B: blue
    1, # R0B: purple
    1, # 0GB: cyan
    1, # RGB: white
])
NUM_CLASSES = len(np.unique(INDEX_MAP))
I = np.identity(NUM_CLASSES, dtype=np.float32)
def transform_y(arr):
    arr[arr > 0] = 1 # fill by 1
    arr = np.sum(np.multiply(arr, (1,2,4,8)), axis=2) # to 4bit
    arr = (arr - 7) # to 3bit + 1
    arr[arr < 0] = 0 # fill overrun
    return ToTensor()(I[INDEX_MAP[arr]])

ds = RandomPatchDataset(
        transform_x = Compose([
            ToTensor(),
            lambda x: x[[2,1,0]],
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transform_y = transform_y)
for i, (x, y) in enumerate(ds):
    print(x.size())
    break

