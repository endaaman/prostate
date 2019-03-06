
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
from net import UNet11, UNet11bn, UNet16, UNet16bn
from utils import now_str, dice_coef, overlay_transparent, to_heatmap


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('-n', '--net')
parser.add_argument('--single-gpu', action="store_true")
parser.add_argument('--cpu', action="store_true")
args = parser.parse_args()

USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and not args.single_gpu
NET_NAME = args.net
NUM_CLASSES = 3

input_path = args.input
base_name = os.path.splitext(os.path.basename(input_path))[0]

print(f'Checking NET:{NET_NAME} GPU:{USE_GPU} MULTI_GPU:{USE_MULTI_GPU} NUM_CLASSES:{NUM_CLASSES} ({now_str()})')

def add_padding(img):
    h, w = img.shape[0:2]
    new_w = math.ceil(w / 32) * 32
    new_h = math.ceil(h / 32) * 32
    left = (new_w - w) // 2
    right = (new_w - w) - left
    top = (new_h - h) // 2
    bottom = (new_h - h) - top
    new_arr = np.pad(img, ((top, bottom), (left, right), (0, 0)), 'constant', constant_values=0)
    return new_arr, (left, top, left + w, top + h)

def post_process(tensor, dims=None):
    arr = np.transpose(tensor.numpy(), (1, 2, 0))
    if dims:
        arr = arr[dims[1]:dims[3], dims[0]:dims[2]]
    row_sums = np.sum(arr, axis=2)
    return arr / row_sums[:, :, np.newaxis]

def to_standard(arr):
    COLOR_MAP = np.array([
        [   0,   0,   0,   0], # 0 -> transparent
        [   0,   0,   0, 255], # 1 -> black
        [ 255,   0,   0, 255], # 2 -> blue
    ], dtype='uint8')
    arr = np.argmax(arr, axis=2)
    return COLOR_MAP[arr]

device = 'cuda' if USE_GPU else 'cpu'

NET = {
    'unet11': UNet11,
    'unet16': UNet16,
    'unet11bn': UNet11bn,
    'unet16bn': UNet16bn,
}[NET_NAME.lower()]
model = NET(num_classes=NUM_CLASSES)
model = model.to(device)
if USE_MULTI_GPU:
    model = torch.nn.DataParallel(model)


input_img = cv2.imread(input_path)
padded_input_img, original_dims = add_padding(input_img)
pre_process = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f'Start checking')
input_tensor = torch.unsqueeze(pre_process(padded_input_img).to(device), dim=0)
with torch.no_grad():
    output_tensor = model(input_tensor)

mask_arr = post_process(output_tensor.data[0].cpu(), original_dims)
print('output dims: ', mask_arr.shape)
print(f'Finished inference.')
