import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from data import LaidDataset, RandomPatchDataset
Image.MAX_IMAGE_PIXELS = 1000000000


ds = RandomPatchDataset()
for i, (x, y) in enumerate(ds):
    print(i)
    y.save(f'./tmp/{i}.png')
    x.save(f'./tmp/{i}.jpg')
    if i > 10:
        exit(0)
