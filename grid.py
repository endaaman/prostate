import sys
import os
import gc
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = 1000000000

if len(sys.argv) < 5:
    print('Invalid arguments.')
    exit(1)

INPTUT_PATH = sys.argv[1]
OUTPUT_DIR = sys.argv[2]
X = int(sys.argv[3])
Y = int(sys.argv[4])


os.makedirs(OUTPUT_DIR, exist_ok=True)
base_name, _ = os.path.splitext(os.path.basename(INPTUT_PATH))

img = Image.open(INPTUT_PATH)
W = img.size[0]
H = img.size[1]
TILE_W = W // X
TILE_H = H // Y
for y in range(0, Y):
    y_last = y == Y - 1
    for x in range(0, X):
        x_last = x == X - 1
        w = TILE_W + ((W % X) if x_last else 0)
        h = TILE_H + ((H % Y) if y_last else 0)
        left = x * TILE_W
        top = y * TILE_H
        i = img.crop((left, top, left + w, top + h))
        ext = 'jpg' if img.mode == 'RGB' else 'png'
        p = f'{OUTPUT_DIR}/{base_name}_{x}_{y}.{ext}'
        i.save(p, quality=100, optimize=True)
        print(f'save {p}')
        i.close()
    gc.collect()

print('done')
