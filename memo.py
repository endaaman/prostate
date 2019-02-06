import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = 1000000000



def func(x):
    if x[3] == 0:
         # transparent -> no label
        return (0,0,0,0)
    if x[2] == 255:
        return (0,0,255,255)
     # other = gleason 1
    return (0,0,0,255)

xx, yy, names = load_images('./train/224')
for i, y in enumerate(yy):
    name = names[i]
    if i > 10:
        exit(0)
    arr = np.asarray(y)
    arr2 = np.apply_along_axis(func, 2, arr)
    i = Image.fromarray(np.uint8(arr2))
    i.save(f'tmp/{name}.png')
    print(f'save {name}')
