import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose
from net import UNet11, UNet11bn, UNet16, UNet16bn
from data import DefaultDataset
from utils import now_str, dice_coef, argmax_acc, curry

parser = argparse.ArgumentParser()
parser.add_argument('weight', default=None, nargs='?')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-e', '--epoch', type=int, default=300)
parser.add_argument('-n', '--net', default='UNet11bn')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--single-gpu', action="store_true")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--vanilla', action="store_true")
args = parser.parse_args()

STARTING_WEIGHT = args.weight
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EPOCH_COUNT = args.epoch
USE_GPU = not args.cpu and torch.cuda.is_available()
USE_MULTI_GPU = USE_GPU and not args.single_gpu
NET_NAME = args.net
VANILLA = args.vanilla

print(f'Preparing NET:{NET_NAME} BATCH SIZE:{BATCH_SIZE} EPOCH:{EPOCH_COUNT} GPU:{USE_GPU} MULTI_GPU:{USE_MULTI_GPU} ({now_str()})')

first_epoch = 1
if STARTING_WEIGHT:
    num = os.path.splitext(os.path.basename(STARTING_WEIGHT))[0]
    if not num.isdigit():
        print(f'Invalid pt file')
        exit(1)
    first_epoch = int(num) + 1

INDEX_MAP = np.array([
    0, # empty
    1, # 000: black
    2, # B00: blue
    2, # 0G0: green
    1, # BG0: cyan
    2, # 00R: red
    1, # B0R: purple
    1, # 0GR: yellow
    1, # RGB: white
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
        transform_y = transform_y)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

device = 'cuda' if USE_GPU else 'cpu'
NET = {
    'unet11': UNet11,
    'unet16': UNet16,
    'unet11bn': UNet11bn,
    'unet16bn': UNet16bn,
    'unet11v': curry(UNet11, pretrained=False),
    'unet16v': curry(UNet16, pretrained=False),
    'unet11vbn': curry(UNet11bn, pretrained=False),
    'unet16vbn': curry(UNet16bn, pretrained=False),
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
while epoch <= EPOCH_COUNT:
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
        if message:
            sys.stdout.write('\r' * len(message))
        message = f'epoch[{epoch}]: {i+1} / {len(data_set) // BATCH_SIZE} dice acc: {dice_acc:.5f} acc: {acc:.5f} loss: {loss:.5f} ({now_str()})'
        sys.stdout.write(message)
        sys.stdout.flush()
    print('')
    print(f'epoch[{epoch}]: Done. dice acc:{np.average(dice_accs):.5f} acc:{np.average(accs):.5f} ({now_str()})')

    weight_path = f'./{weight_dir}/{epoch}.pt'
    state = model.module.cpu().state_dict() if USE_MULTI_GPU else model.cpu().state_dict()
    torch.save(state, weight_path)
    print(f'save weights to {weight_path}')
    model = model.to(device)
    epoch += 1

print(f'Finished training')
