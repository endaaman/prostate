import os
import sys
import math
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
Image.MAX_IMAGE_PIXELS = 1000000000

from net import UNet11

def load_image_with_paddig(path):
    img = Image.open(path)
    w, h = img.size
    new_w = math.ceil(w / 32) * 32
    new_h = math.ceil(h / 32) * 32
    left = (new_w - w) // 2
    top = (new_h - h) // 2
    bg = Image.new('RGB', (new_w, new_h), (255, 255, 255))
    bg.paste(img, (left, top))
    return bg, (left, top, left + w, top + h)

COLOR_MAP = np.array([
    [   0,   0,   0,   0], # 0 -> transparent
    [   0,   0,   0, 255], # 1 -> black
    [   0,   0, 255, 255], # 2 -> blue
])

def restore_mask(tensor, dims=None):
    arr = np.transpose(tensor.numpy(), (1, 2, 0))
    arr = np.argmax(arr, axis=2)
    h, w = arr.shape
    arr = COLOR_MAP[arr]
    if dims:
        arr = arr[dims[1]:dims[3],dims[0]:dims[2]]
    return Image.fromarray(np.uint8(arr))


if len(sys.argv) < 3:
    print(f'Invalid argument')
    exit(1)

weight_file = sys.argv[1]
input_file = sys.argv[2]
base_name, _ = os.path.splitext(os.path.basename(input_file))
output_file = f'./out/{base_name}.png'

GPU = True
device = 'cuda' if GPU and torch.cuda.is_available() else 'cpu'


model = UNet11(num_classes = 3)
model.load_state_dict(torch.load(weight_file))
model = model.to(device)


input_img, original_dims = load_image_with_paddig(input_file)
transform_img = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


print(f'Start inference')
with torch.no_grad():
    input_img = torch.unsqueeze(transform_img(input_img).to(device), dim=0)
    mask = model(input_img)

tensor = mask.data[0].cpu()
mask_img = restore_mask(tensor, original_dims)
mask_img.save(output_file)
print(f'Done. saved to {output_file}')
