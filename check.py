
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
from utils import now_str, dice_coef, overlay_transparent, to_heatmap


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', default='UNet11')
parser.add_argument('--single-gpu', action="store_true")
parser.add_argument('--cpu', action="store_true")
args = parser.parse_args()

USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and not args.single_gpu
NET_NAME = args.net
NUM_CLASSES = 3


print(f'Checking NET:{NET_NAME} GPU:{USE_GPU} MULTI_GPU:{USE_MULTI_GPU} NUM_CLASSES:{NUM_CLASSES} ({now_str()})')

device = 'cuda' if USE_GPU else 'cpu'
NET = {
    'unet11': UNet11,
    'unet16': UNet16,
}[NET_NAME.lower()]
model = NET(num_classes=NUM_CLASSES)
print(model)
model = model.to(device)

input_img = np.random.randint(0, 256, [224, 224, 3], dtype=np.uint8)
pre_process = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f'Start checking')
input_tensor = torch.unsqueeze(pre_process(input_img).to(device), dim=0)
with torch.no_grad():
    output_tensor = model(input_tensor)

mask_arr = output_tensor.data[0].cpu()
print('output dims: ', mask_arr.shape)
print(mask_arr[:10])
print(f'Finished inference.')
