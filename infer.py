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
from utils import now_str, dice_coef


parser = argparse.ArgumentParser()
parser.add_argument('weight')
parser.add_argument('input')
parser.add_argument('-n', '--net', default='UNet11bn')
parser.add_argument('--single-gpu', action="store_true")
parser.add_argument('--cpu', action="store_true")
args = parser.parse_args()

USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and not args.single_gpu
NET_NAME = args.net
NUM_CLASSES = 3

weight_path = args.weight
input_path = args.input
output_path = './out/'  + os.path.splitext(os.path.basename(input_path))[0] + '.png'

print(f'Preparing NET:{NET_NAME} GPU:{USE_GPU} MULTI_GPU:{USE_MULTI_GPU} NUM_CLASSES:{NUM_CLASSES} ({now_str()})')

def load_image_with_paddig(path):
    img = cv2.imread(path)
    h, w = img.shape[0:2]
    new_w = math.ceil(w / 32) * 32
    new_h = math.ceil(h / 32) * 32
    left = (new_w - w) // 2
    right = (new_w - w) - left
    top = (new_h - h) // 2
    bottom = (new_h - h) - top
    new_arr = np.pad(img, ((top, bottom), (left, right), (0, 0)), 'constant', constant_values=0)
    return new_arr, (left, top, left + w, top + h)

COLOR_MAP = np.array([
    [   0,   0,   0,   0], # 0 -> transparent
    [   0,   0,   0, 255], # 1 -> black
    [ 255,   0,   0, 255], # 2 -> blue
], dtype='uint8')

def restore_mask(tensor, dims=None):
    arr = np.transpose(tensor.numpy(), (1, 2, 0))
    if dims:
        arr = arr[dims[1]:dims[3], dims[0]:dims[2]]
    arr = np.argmax(arr, axis=2)
    arr = COLOR_MAP[arr]
    return arr

device = 'cuda' if USE_GPU else 'cpu'

NET = {
    'unet11': UNet11,
    'unet16': UNet16,
    'unet11bn': UNet11bn,
    'unet16bn': UNet16bn,
}[NET_NAME.lower()]
model = NET(num_classes=NUM_CLASSES)
model = model.to(device)
model.load_state_dict(torch.load(weight_path))
if USE_MULTI_GPU:
    model = torch.nn.DataParallel(model)

input_img, original_dims = load_image_with_paddig(input_path)
transform_img = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f'Start inference')
input_tensor = torch.unsqueeze(transform_img(input_img).to(device), dim=0)
with torch.no_grad():
    output_tensor = model(input_tensor)

mask_img = restore_mask(output_tensor.data[0].cpu(), original_dims)
cv2.imwrite(output_path, mask_img)
print(f'Done. saved to {output_path}')
