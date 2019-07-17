import os
import sys
import math
import json
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
from metrics import Metrics, calc_coef, coef_to_str
from utils import now_str, pp, overlay_transparent
from formula import *


parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight')
parser.add_argument('-m', '--model')
parser.add_argument('--cpu', action="store_true")
parser.add_argument('-d', '--dest', default='report')
parser.add_argument('-s', '--size', type=int, default=3000)
parser.add_argument('--one', action="store_true")
args = parser.parse_args()

WEIGHT_PATH = args.weight
MODEL_NAME = args.model
DEST_BASE_DIR = args.dest
ONE = args.one
SIZE = args.size

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

def label_to_img(arr, alpha=False):
    arr = np.argmax(arr, axis=2)
    return COLOR_MAP_ALPHA[arr] if alpha else COLOR_MAP[arr]


class Report:
    def __init__(self, meta):
        self.meta = meta
        self.items = []
        self.path = os.path.join(DEST_DIR, 'report.json')

    def to_entry(self, name, coef, t):
        return {'name': name, 'coef': coef._asdict(), 'type': t}

    def append(self, name, coef, is_train):
        self.items.append(self.to_entry(name, coef, is_train))

    def prepend(self, name, coef, is_train):
        self.items.insert(0, self.to_entry(name, coef, is_train))

    def save(self):
        data = {'meta': self.meta, 'items': self.items}
        with open(self.path, 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4, separators=(',', ': '))

report = Report({'model': MODEL_NAME, 'size': SIZE, 'mode': mode, 'weight': WEIGHT_PATH})

train_metrics = Metrics()
validation_metrics = Metrics()

dataset = ValidationDataset(max_size=SIZE, one=ONE)
print(f'Start validation')
for (name, x_raw, y_raw, x_data, y_data, is_train) in dataset:
    metrics = Metrics()
    output_img_rows = []
    for y, row in enumerate(x_data):
        output_img_tiles = []
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
            output_tensor = torch.unsqueeze(torch.from_numpy(output_arr).permute(2, 0, 1), dim=0).to(device)
            label_tensor = torch.unsqueeze(torch.from_numpy(label_arr).permute(2, 0, 1), dim=0).to(device)
            coef = calc_coef(output_tensor, label_tensor)
            output_img_tiles.append(output_arr)
            metrics.append_coef(coef)
            pp(f'Process {name} {x},{y}/{len(row)-1},{len(x_data)-1} {coef_to_str(coef)} ({now_str()})')
            gc.collect()
        output_img_rows.append(cv2.hconcat(output_img_tiles))
    output_img = label_to_img(cv2.vconcat(output_img_rows), alpha=True)
    masked_img = overlay_transparent(x_raw, output_img) # TODO: overlay transparented mask
    os.makedirs(DEST_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEST_DIR, f'{name}.jpg'), masked_img)
    m = train_metrics if is_train else validation_metrics
    m.append_nested_metrics(metrics)
    report.append(name, metrics.avg_coef(), 'train' if is_train else 'validation')
    report.save()
    pp(f'{name}: {coef_to_str(coef)} ({now_str()})')
    print('')

whole_metrics = Metrics()
whole_metrics.append_nested_metrics(train_metrics)
whole_metrics.append_nested_metrics(validation_metrics)

report.prepend('train', train_metrics.avg_coef(), 'mean train')
report.prepend('validation', validation_metrics.avg_coef(), 'mean validation')
report.prepend('all', whole_metrics.avg_coef(), 'mean all')
report.save()
print(f'Save report to. {report.path} ({now_str()})')
print()
