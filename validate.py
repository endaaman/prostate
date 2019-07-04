import os
import sys
import math
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
from utils import now_str


NUM_CLASSES = 5

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


def remove_padding(tensor, dims=None):
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


print(f'Start validation')
dataset = ValidationDataset(max_size=3000)
for (x_data, y_data) in dataset:
    for y in x_data.shape[0]:
        for y in x_data.shape[1]:
            input_tensor = x_data[y][x]
            input_tensor, original_dims = add_padding(input_tensor)
            pre_process = Compose([
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = torch.unsqueeze(pre_process(input_tensor).to(device), dim=0)
            with torch.no_grad():
                output_tensor = model(input_tensor)
            output_tensor = np.transpose(output_tensor.numpy(), (1, 2, 0))
            output_tensor = remove_padding(output_tensor.data[0].cpu(), original_dims)
            print(output_tensor.size())
            break

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
