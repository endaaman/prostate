import os
import argparse
from enum import Enum, auto
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose

from net import UNet11, UNet16
from data import DefaultDataset
from utils import now_str, pp, dice_coef, argmax_acc, curry

IDX_NONE, IDX_NORMAL, IDX_GLEASON_3, IDX_GLEASON_4, IDX_GLEASON_5 = range(5)

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weight')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-e', '--epoch', type=int, default=100)
parser.add_argument('-t', '--tile', type=int, default=224)
parser.add_argument('-n', '--net', default='UNet11')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--single-gpu', action="store_true")
# parser.add_argument('--accurated', action="store_true")
parser.add_argument('--cpu', action="store_true")
args = parser.parse_args()

STARTING_WEIGHT = args.weight
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EPOCH_COUNT = args.epoch
TILE_SIZE = args.tile
# ACCURATED = args.accurated
USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and not args.single_gpu
NET_NAME = args.net
mode = ('multi' if USE_MULTI_GPU else 'single') if USE_GPU else 'cpu'

print(f'Preparing NET:{NET_NAME} BATCH SIZE:{BATCH_SIZE} EPOCH:{EPOCH_COUNT} MODE: {mode} ({now_str()})')

first_epoch = 1
if STARTING_WEIGHT:
    num = os.path.splitext(os.path.basename(STARTING_WEIGHT))[0]
    if not num.isdigit():
        print(f'Invalid pt file')
        exit(1)
    first_epoch = int(num) + 1

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

device = 'cuda' if USE_GPU else 'cpu'
NET = {
    'unet11': UNet11,
    'unet16': UNet16,
    'unet11v': curry(UNet11, pretrained=False),
    'unet16v': curry(UNet16, pretrained=False),
}[NET_NAME.lower()]
model = NET(num_classes=NUM_CLASSES)
model = model.to(device)
if STARTING_WEIGHT:
    model.load_state_dict(torch.load(STARTING_WEIGHT))
if USE_MULTI_GPU:
    model = torch.nn.DataParallel(model)


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

print(f'Starting ({now_str()})')
epoch = first_epoch
weight_dir = f'./weights/{NET_NAME}'
os.makedirs(weight_dir, exist_ok=True)
while epoch < first_epoch + EPOCH_COUNT:
    message = None
    dice_accs = []
    accs = []
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        dice_acc = dice_coef(outputs, labels)
        dice_accs.append(dice_acc)
        acc = argmax_acc(outputs, labels)
        accs.append(acc)
        loss.backward()
        optimizer.step()
        pp(f'epoch[{epoch}]: {i+1} / {len(data_set) // BATCH_SIZE} dice: {dice_acc:.5f} iou: {acc:.5f} loss: {loss:.5f} ({now_str()})')
    print('')
    print(f'epoch[{epoch}]: Done. dice:{np.average(dice_accs):.5f} iou:{np.average(accs):.5f} ({now_str()})')

    weight_path = f'./{weight_dir}/{epoch}.pt'
    state = model.module.cpu().state_dict() if USE_MULTI_GPU else model.cpu().state_dict()
    torch.save(state, weight_path)
    print(f'save weights to {weight_path}')
    model = model.to(device)
    epoch += 1

print(f'Finished training')
