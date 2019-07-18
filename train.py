import os
import math
import re
import argparse
from enum import Enum, auto
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose

from models import get_model
from data import TrainingDataset
from store import Store
from metrics import Metrics, calc_coef
from formula import *
from utils import now_str, pp

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-t', '--tile', type=int, default=224)
parser.add_argument('-m', '--model', default='unet11')
parser.add_argument('-d', '--dest', default='weights')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--cpu', action="store_true")
args = parser.parse_args()

STARTING_WEIGHT = args.weight
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EPOCH_COUNT = args.epoch
TILE_SIZE = args.tile
MODEL_NAME = args.model
DEST_BASE_DIR = args.dest

USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and torch.cuda.device_count() > 1
DEST_DIR = os.path.join(DEST_BASE_DIR, MODEL_NAME)

os.makedirs(DEST_DIR, exist_ok=True)
if not os.path.isdir(DEST_DIR):
    print(f'Invalid dest dir: `{DEST_DIR}`')
    exit(1)

store = Store()
mode = ('multi' if USE_MULTI_GPU else 'single') if USE_GPU else 'cpu'
device = 'cuda' if USE_GPU else 'cpu'

# EPOCH
first_epoch = 1
if STARTING_WEIGHT:
    basename = os.path.splitext(os.path.basename(STARTING_WEIGHT))[0]
    nums = re.findall(r'\d+', basename)
    if len(nums) > 0 and not nums[-1].isdigit():
        print(f'Invalid pt file')
        exit(1)
    first_epoch = int(nums[-1]) + 1
    store.load(STARTING_WEIGHT)
epoch = first_epoch

print(f'Preparing MODEL:{MODEL_NAME} BATCH:{BATCH_SIZE} EPOCH:{EPOCH_COUNT} MODE:{mode} ({now_str()})')


# MDOEL
Model = get_model(MODEL_NAME)
model = Model(num_classes=NUM_CLASSES).to(device)
if store.weights:
    model.load_state_dict(store.weights)
if USE_MULTI_GPU:
    model = torch.nn.DataParallel(model)


# DATA
def transform_y(arr):
    arr[arr > 0] = 1 # to 1bit each color
    arr = np.sum(np.multiply(arr, (1, 2, 4, 8)), axis=2) # to 4bit each pixel
    arr = arr - 7 # to 3bit + 1
    arr[arr < 0] = 0 # fill overrun
    return ToTensor()(INDEX_MAP[arr])

data_set = TrainingDataset(
        transform_x = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        transform_y = transform_y,
        tile_size=TILE_SIZE)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


# TRAIN
def lr_func_exp(step):
    return 0.95 ** step

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
if store.optims:
    optimizer.load_state_dict(store.optims)
scheduler = LambdaLR(optimizer, lr_lambda=lr_func_exp, last_epoch=epoch if store.optims else -1)
# criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()
criterion = nn.CrossEntropyLoss()

metrics = Metrics()
if store.metrics:
    metrics.load_state_dict(store.metrics)

# LOOP
print(f'Starting ({now_str()})')
iter_count = len(data_set) // BATCH_SIZE
while epoch < first_epoch + EPOCH_COUNT:
    iter_metrics = Metrics()
    lr = scheduler.get_lr()[0]
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        coef = calc_coef(outputs, labels)
        iter_metrics.append_loss(loss.item())
        iter_metrics.append_coef(coef)
        pp('epoch[{ep}]:{i}/{I} iou:{c.pjac:.4f} acc:{c.pdice:.4f} lr:{lr:.4f} ({t})'.format(
            ep=epoch, i=i+1, I=iter_count, lr=lr, t=now_str(), loss=loss.item(), c=coef))
        loss.backward()
        optimizer.step()
    pp('epoch[{ep}]:Done. iou:{c.pjac:.4f} acc:{c.pdice:.4f} gsi:{c.gsensi:.4f} gsp:{c.gspec:.4f} tsi:{c.tsensi:.4f} tsp:{c.tspec:.4f} loss:{loss:.4f} lr:{lr:.4f} ({t})'.format(
        ep=epoch, t=now_str(), lr=lr, loss=iter_metrics.avg('losses'), c=iter_metrics.avg_coef()
        ))
    print()
    weight_path = os.path.join(DEST_DIR, f'{Model.__name__.lower()}_{epoch}.pt')
    weights = model.module.cpu().state_dict() if USE_MULTI_GPU else model.cpu().state_dict()
    metrics.append_nested_metrics(iter_metrics)
    store.set_states(weights, optimizer.state_dict(), metrics.state_dict())
    store.save(weight_path)
    print(f'save weights to {weight_path}')
    model = model.to(device)
    scheduler.step()
    epoch += 1

print(f'Finished training\n')
