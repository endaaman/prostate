import os
import math
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
from data import DefaultDataset
from store import Store
from metrics import Metrics
from utils import now_str, pp, revert_onehot, similarity_index, pixel_similarity_index, inspection_accuracy

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

# FORMULA
IDX_NONE, IDX_NORMAL, IDX_GLEASON_3, IDX_GLEASON_4, IDX_GLEASON_5 = range(5)
INDEX_MAP = np.array([
    IDX_NONE,      # empty
    IDX_NORMAL,    # 000: black
    IDX_GLEASON_3, # B00: blue
    IDX_GLEASON_4, # 0G0: green
    IDX_GLEASON_3, # BG0: cyan
    IDX_GLEASON_5, # 00R: red
    IDX_NORMAL,    # B0R: purple
    IDX_GLEASON_5, # 0GR: yellow
    IDX_NONE,      # BGR: white
])
NUM_CLASSES = len(np.unique(INDEX_MAP))


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
I = np.identity(NUM_CLASSES, dtype=np.float32)
def transform_y(arr):
    arr[arr > 0] = 1 # to 1bit each color
    arr = np.sum(np.multiply(arr, (1,2,4,8)), axis=2) # to 4bit each pixel
    arr = arr - 7 # to 3bit + 1
    arr[arr < 0] = 0 # fill overrun
    return ToTensor()(I[INDEX_MAP[arr]])

data_set = DefaultDataset(
        transform_x = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        transform_y = transform_y,
        tile_size=TILE_SIZE)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


# TRAIN
inflection = 100
def lr_func_linear(step):
    return max(1 - step * 0.9 / inflection, 0.1)

def lr_func_exp(step):
    return 0.95 ** step

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
if store.optims:
    optimizer.load_state_dict(store.optims)
scheduler = LambdaLR(optimizer, lr_lambda=lr_func_exp, last_epoch=epoch if store.optims else -1)
criterion = nn.BCELoss()
# criterion = nn.BCEWithLogitsLoss()

metrics = Metrics()
if store.metrics:
    metrics.load_state_dict(store.metrics)

def process_metrics(outputs, labels):
    dice, jac = similarity_index(outputs, labels)
    output_values = revert_onehot(outputs)
    label_values = revert_onehot(labels)
    pdice, pjac = pixel_similarity_index(output_values, label_values)
    output_glands = output_values >= IDX_NORMAL
    label_glands = label_values >= IDX_NORMAL
    output_tumors = output_values >= IDX_GLEASON_3
    label_tumors = label_values >= IDX_GLEASON_3
    gsensi, gspec = inspection_accuracy(output_glands, label_glands)
    tsensi, tspec = inspection_accuracy(output_tumors, label_tumors)
    return dice, jac, pdice, pjac, gsensi, gspec, tsensi, tspec


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
        values = process_metrics(outputs, labels)
        iter_metrics.append_values(loss, *values)
        pp(f'epoch[{epoch}]:{i}/{I} iou:{iou:.4f} acc:{acc:.4f} loss:{loss:.4f} lr:{lr:.4f} ({t})'.format(
            ep=epoch, i=i+1, I=iter_count,
            iou=iter_metrics.last('jac'),
            acc=iter_metrics.last('pdice'),
            loss=iter_metrics.last('loss'),
            t=now_str()))
        loss.backward()
        optimizer.step()
    print('')
    print('epoch[{ep}]: Done. iou:{iou:.4f} acc:{acc:.4f} loss:{loss:.4f} ({t})'.format(
        ep=epoch,
        iou=iter_metrics.avg('jac'),
        acc=iter_metrics.avg('pdice'),
        loss=iter_metrics.avg('loss'),
        t=now_str()))
    weight_path = os.path.join(DEST_DIR, f'{model.__class__.__name__.lower()}_{epoch}.pt')
    weights = model.module.cpu().state_dict() if USE_MULTI_GPU else model.cpu().state_dict()
    metrics.append_metrics(met)
    store.set_states(weights, optimizer.state_dict(), metrics.state_dict())
    store.save(weight_path)
    print(f'save weights to {weight_path}')
    model = model.to(device)
    scheduler.step()
    epoch += 1

print(f'Finished training\n')
