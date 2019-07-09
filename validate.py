import os
import sys
import math
import gc
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose

from models import get_model
from data import ValidationDataset
from store import Store
from metrics import Metrics, calc_coef
from utils import now_str
from formula import *


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight')
parser.add_argument('-m', '--model')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('-d', '--dest', default='out')
args = parser.parse_args()

WEIGHT_PATH = args.weight
MODEL_NAME = args.model
DEST_BASE_DIR = args.dest

USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and torch.cuda.device_count() > 1

DEST_DIR = os.path.join(DEST_BASE_DIR, MODEL_NAME)

mode = ('multi' if USE_MULTI_GPU else 'single') if USE_GPU else 'cpu'
print(f'Preparing MODEL:{MODEL_NAME} MODE:{mode} NUM_CLASSES:{NUM_CLASSES} ({now_str()})')


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

def img_to_label(arr):
    arr[arr > 0] = 1 # to 1bit each color
    arr = np.sum(np.multiply(arr, (1, 2, 4, 8)), axis=2) # to 4bit each pixel
    arr = arr - 7 # to 3bit + 1
    arr[arr < 0] = 0 # fill overrun
    return np.identity(NUM_CLASSES, dtype=np.float32)[INDEX_MAP[arr]]

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


print(f'Start validation')
dataset = ValidationDataset(max_size=1000, one=True)
for (x_data, y_data, name) in dataset:
    print(name)
    metrics = Metrics()
    for y, row in enumerate(x_data):
        for x, input_arr in enumerate(row):
            label_arr = img_to_label(y_data[y][x])
            input_arr, original_dims = add_padding(x_data[y][x])
            pre_process = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = torch.unsqueeze(pre_process(input_arr).to(device), dim=0)
            with torch.no_grad():
                output_tensor = model(input_tensor)
            output_arr = output_tensor.data[0].cpu().numpy()
            output_arr = np.transpose(output_arr, (1, 2, 0))
            output_arr = remove_padding(output_arr, original_dims)
            outputs_tensor = torch.unsqueeze(torch.from_numpy(output_arr).to(device), dim=0)
            labels_tensor = torch.unsqueeze(torch.from_numpy(label_arr).to(device), dim=0)
            coef = calc_coef(outputs_tensor, labels_tensor)
            metrics.append_coef(coef)
            gc.collect()
            print(x, y)
    print(name, metrics.avg_coef())
exit(0)

base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
output_dir = f'./{DEST_BASE_DIR}/{MODEL_NAME}/{base_name}'
os.makedirs(output_dir, exist_ok=True)
np.save(f'{output_dir}/out.npy', mask_arr)
mask_img = arr_to_img(mask_arr)
cv2.imwrite(f'{output_dir}/org.jpg', input_img)
cv2.imwrite(f'{output_dir}/out.png', mask_img)
masked_img = overlay_transparent(input_img, mask_img)
cv2.imwrite(f'{output_dir}/masked.jpg', masked_img)
for i in range(NUM_CLASSES):
    img = to_heatmap(mask_arr[:, :, i])
    cv2.imwrite(f'{output_dir}/heat_{i}.png', img)
    fused = overlay_transparent(input_img, img)
    cv2.imwrite(f'{output_dir}/fused_{i}.png', fused)

print(f'Save images.')
