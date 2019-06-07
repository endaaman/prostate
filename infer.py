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
from net import UNet11, UNet16
from utils import now_str, overlay_transparent, to_heatmap


parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('-w', '--weight')
parser.add_argument('-n', '--net')
parser.add_argument('--single-gpu', action="store_true")
parser.add_argument('--cpu', action="store_true")
args = parser.parse_args()

INPUT_PATH = args.input
WEIGHT_PATH = args.weight
USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and not args.single_gpu
NET_NAME = args.net
NUM_CLASSES = 3
mode = ('multi' if USE_MULTI_GPU else 'single') if USE_GPU else 'cpu'
print(f'Preparing NET:{NET_NAME} GPU:{USE_GPU} MODE: {mode} NUM_CLASSES:{NUM_CLASSES} ({now_str()})')


base_name = os.path.splitext(os.path.basename(INPUT_PATH))[0]
output_dir = f'./out/{NET_NAME.lower()}/{base_name}'
os.makedirs(output_dir, exist_ok=True)
output_img_path = f'{output_dir}/out.png'
output_arr_path = f'{output_dir}/out.npy'



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
}[NET_NAME.lower()]
model = NET(num_classes=NUM_CLASSES)
model = model.to(device)
if WEIGHT_PATH:
    model.load_state_dict(torch.load(WEIGHT_PATH))
if USE_MULTI_GPU:
    model = torch.nn.DataParallel(model)


input_img = cv2.imread(INPUT_PATH)
padded_input_img, original_dims = add_padding(input_img)
pre_process = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f'Start inference')
input_tensor = torch.unsqueeze(pre_process(padded_input_img).to(device), dim=0)
with torch.no_grad():
    output_tensor = model(input_tensor)

mask_arr = post_process(output_tensor.data[0].cpu(), original_dims)
print(f'Finished inference.')

cv2.imwrite(f'{output_dir}/org.jpg', input_img)
mask_img = to_standard(mask_arr)
cv2.imwrite(output_img_path, mask_img)
np.save(output_arr_path, mask_arr)
for i in range(NUM_CLASSES):
    img = to_heatmap(mask_arr[:, :, i])
    cv2.imwrite(f'{output_dir}/heat_{i}.png', img)
    fused = overlay_transparent(input_img, img)
    cv2.imwrite(f'{output_dir}/fused_{i}.png', fused)

print(f'Save images.')
