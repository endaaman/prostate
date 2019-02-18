import os
import sys
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
from net import UNet11, UNet11bn, UNet16, UNet16bn
from utils import now_str, dice_coef


MULTI_GPU = True
NET_NAME = 'unet16'
NUM_CLASSES = 3

print(f'Preparing NET: {NET_NAME} MULTI_GPU: {MULTI_GPU} NUM_CLASSES: {NUM_CLASSES} ({now_str()})')

NETs = {
    'unet11': UNet11,
    'unet16': UNet16,
    'unet11bn': UNet11bn,
    'unet16bn': UNet16bn,
}
NET = NETs[NET_NAME]

def load_image_with_paddig(path):
    img = cv2.imread(path)
    h, w = img.shape[0:2]
    new_w = math.ceil(w / 32) * 32
    new_h = math.ceil(h / 32) * 32
    left = (new_w - w) // 2
    right = (new_w - w) - left
    top = (new_h - h) // 2
    bottom = (new_h - h) - top
    new_arr = np.pad(img, ((top, bottom), (left, right), (0,0,)), 'constant', constant_values=0)
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


if len(sys.argv) < 3:
    print(f'Invalid argument')
    exit(1)

weight_file = sys.argv[1]
input_file = sys.argv[2]
base_name, _ = os.path.splitext(os.path.basename(input_file))
output_file = f'./out/{base_name}.png'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None
if NET == 'unet11':
    model = UNet11(num_classes=NUM_CLASSES)
else:
    model = UNet16(num_classes=NUM_CLASSES)
model = model.to(device)
model.load_state_dict(torch.load(weight_file))
if MULTI_GPU and device == 'cuda':
    model = torch.nn.DataParallel(model)


input_img, original_dims = load_image_with_paddig(input_file)
transform_img = Compose([
    ToTensor(),
    lambda x: x[[2,1,0]],
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print(f'Start inference')
input_tensor = torch.unsqueeze(transform_img(input_img).to(device), dim=0)
with torch.no_grad():
    output_tensor = model(input_tensor)

mask_img = restore_mask(output_tensor.data[0].cpu(), original_dims)
cv2.imwrite(mask_img, output_file)
print(f'Done. saved to {output_file}')
