import sys
import os
import gc
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = 1000000000

if len(sys.argv) < 3:
    print('Invalid arguments.')
    exit(1)

DIR_NAME = sys.argv[1]
SRC_PATH = sys.argv[2]
print(f'{DIR_NAME} {SRC_PATH}')
TILE_SIZE = 112

DST_DIR = f'./outputs/{DIR_NAME}'
os.makedirs(DST_DIR, exist_ok=True)

img = Image.open(SRC_PATH)
W = img.size[0]
H = img.size[1]
X = int(W / TILE_SIZE)
Y = int(H / TILE_SIZE)
for x in range(0, X):
    for y in range(0, Y):
        x_start = int(x * TILE_SIZE)
        y_start = int(y * TILE_SIZE)
        i = img.crop((x_start, y_start, x_start + TILE_SIZE, y_start + TILE_SIZE))
        ext = 'jpg' if img.mode == 'RGB' else 'png'
        i.save(f'{DST_DIR}/{x}_{y}.{ext}', quality=100, optimize=True)
        i.close()
    print(f'column {x}: done')
    gc.collect()

print('done')
