import os
import sys
import math
import argparse
import gc
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose

from models import get_model
from store import Store
from formula import *
from utils import pp, now_str, curry, to_heatmap, overlay_transparent, split_maxsize


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('-w', '--weight')
parser.add_argument('-m', '--model')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('-s', '--size', type=int, default=1000)
parser.add_argument('-s2', '--size2', type=int, default=None)
parser.add_argument('-d', '--dest')
args = parser.parse_args()

INPUT_PATH = args.input
WEIGHT_PATH = args.weight
MODEL_NAME = args.model
SIZE = args.size
SIZE2 = args.size2 or SIZE

BASE_NAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]
if args.dest:
    DEST_DIR = args.dest
else:
    DEST_DIR = f'./out/{MODEL_NAME}/{BASE_NAME}'

USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and torch.cuda.device_count() > 1

mode = ('multi' if USE_MULTI_GPU else 'single') if USE_GPU else 'cpu'
print(f'Preparing MODEL:{MODEL_NAME} MODE:{mode} SIZE:{SIZE} TARGET:{INPUT_PATH} ({now_str()})')

def add_padding(img):
    h, w = img.shape[0:2]
    new_w = math.ceil(w / 64) * 64
    new_h = math.ceil(h / 64) * 64
    left = (new_w - w) // 2
    right = (new_w - w) - left
    top = (new_h - h) // 2
    bottom = (new_h - h) - top
    new_arr = np.pad(img, ((top, bottom), (left, right), (0, 0)), 'constant', constant_values=0)
    return new_arr, (left, top, left + w, top + h)


def remove_padding(arr, dims=None):
    if dims:
        arr = arr[dims[1]:dims[3], dims[0]:dims[2]]
    row_sums = np.sum(arr, axis=2)
    return arr / row_sums[:, :, np.newaxis]

def arr_to_img(arr):
    COLOR_MAP = np.array([
        [   0,   0,   0,   0], # 0 -> transparent
        [   0,   0,   0, 255], # 1 -> black
        [ 255,   0,   0, 255], # 2 -> blue
        [   0, 255,   0, 255], # 3 -> green
        [   0,   0, 255, 255], # 4 -> red
    ], dtype='uint8')
    arr = np.argmax(arr, axis=2)
    return COLOR_MAP[arr]

device = 'cuda' if USE_GPU else 'cpu'
store = Store()
store.load(WEIGHT_PATH)

Model = get_model(MODEL_NAME)
model = Model(num_classes=NUM_CLASSES).to(device)
if store.weights:
    model.load_state_dict(store.weights)
else:
    raise Exception(f'Weights are needed.')
if USE_MULTI_GPU:
    model = torch.nn.DataParallel(model)

input_img = cv2.imread(INPUT_PATH)

print(f'Start inference')
grid = split_maxsize(input_img, (SIZE, SIZE2))
output_img_rows = []
for y, row in enumerate(grid):
    output_img_tiles = []
    for x, img in enumerate(row):
        padded_input_img, original_dims = add_padding(img)
        pp(f'Processing {x},{y}/{len(row)-1},{len(grid)-1} size:{padded_input_img.shape} ({now_str()})')
        pre_process = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = torch.unsqueeze(pre_process(padded_input_img).to(device), dim=0)
        with torch.no_grad():
            output_tensor = model(input_tensor)
        output_arr = output_tensor.data[0].cpu().numpy()
        output_arr = remove_padding(np.transpose(output_arr, (1, 2, 0)), original_dims)
        output_img_tiles.append(output_arr)
        gc.collect()
    output_img_rows.append(cv2.hconcat(output_img_tiles))

pp(f'Done process {INPUT_PATH}')
print('')
mask_arr = cv2.vconcat(output_img_rows)
os.makedirs(DEST_DIR, exist_ok=True)
np.save(f'{DEST_DIR}/out.npy', mask_arr)
mask_img = arr_to_img(mask_arr)
cv2.imwrite(f'{DEST_DIR}/org.png', input_img)
cv2.imwrite(f'{DEST_DIR}/overlay.png', mask_img)
masked_img = overlay_transparent(input_img, mask_img)
cv2.imwrite(f'{DEST_DIR}/masked.jpg', masked_img)

names = [
    'non-gland',
    'non-cancer',
    'GP3',
    'GP4',
    'GP5',
]
for i in range(NUM_CLASSES):
    img = to_heatmap(mask_arr[:, :, i], alpha=255)
    cv2.imwrite(f'{DEST_DIR}/overlay_{i}_{names[i]}.png', img)
    fused = overlay_transparent(input_img, img)
    cv2.imwrite(f'{DEST_DIR}/fused_{i}.jpg', fused)

print(f'Save images.  ({now_str()})')
