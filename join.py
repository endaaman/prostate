import sys
import os
import re
import numpy as np
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = 1000000000

if len(sys.argv) < 2:
    print('Invalid arguments.')
    exit(1)

INPTUT_DIR = sys.argv[1]

file_names = os.listdir(INPTUT_DIR)
file_names.sort()

image = Image.new('RGBA', (10000, 10000), (0, 0, 0, 0))

w = 2000
h = 2000
base_name = '2_1_tile'
for file_name in file_names:
    m = re.match(f'^{base_name}_(\d)_(\d)\.png$', file_name)
    x = int(m[1])
    y = int(m[2])
    i = Image.open(f'{INPTUT_DIR}/{file_name}')
    image.paste(i, (w * x, h * y))
    print(f'paste {file_name}')

# def blue_to_black(x):
#     if x[3] == 0:
#         return (0, 0, 0, 0)
#     if x[0] == 0 and x[1] == 0 and x[2] == 255:
#         return (0, 0, 0, 255)
#     return x
# arr = np.asarray(image)
# arr = np.apply_along_axis(blue_to_black, 2, arr)
# image = Image.fromarray(arr)

image.save('out.png')
