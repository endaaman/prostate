import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose
from net import UNet11, UNet11bn, UNet16, UNet16bn
from data import LaidDataset, RandomPatchDataset
from utils import now_str, dice_coef


BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCH_COUNT = 500
MULTI_GPU = True
NET_NAME = 'unet11bn'
NETs = {
    'unet11': UNet11,
    'unet16': UNet16,
    'unet11bn': UNet11bn,
    'unet16bn': UNet16bn,
}
NET = NETs[NET_NAME]

print(f'Preparing NET: {NET_NAME} BATCH: {BATCH_SIZE} EPOCH: {EPOCH_COUNT} MULTI_GPU: {MULTI_GPU} ({now_str()})')

first_epoch = 1
weight_file = None
if len(sys.argv) > 1:
    weight_file = sys.argv[1]
    num = os.path.splitext(os.path.basename(weight_file))[0]
    if not num.isdigit():
        print(f'Invalid pt file')
        exit(1)
    first_epoch = int(num) + 1


INDEX_MAP = np.array([
    0, # empty
    1, # 000: black
    2, # B00: blue
    1, # 0G0: green
    1, # BG0: cyan
    1, # 00R: red
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

data_set = RandomPatchDataset(
        transform_x = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transform_y = transform_y)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = None
model = NET(num_classes=NUM_CLASSES)
model = model.to(device)
if weight_file:
    model.load_state_dict(torch.load(weight_file))
if MULTI_GPU and device == 'cuda':
    model = torch.nn.DataParallel(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

print(f'Starting ({now_str()})')
epoch = first_epoch
weight_dir = f'./weights/{NET_NAME}/'
os.makedirs(weight_dir, exist_ok=True)
while epoch <= EPOCH_COUNT:
    message = None
    accs = []
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        acc = dice_coef(outputs, labels)
        accs.append(acc.item())
        loss.backward()
        optimizer.step()
        if message:
            sys.stdout.write('\r' * len(message))
        message = f'epoch[{epoch}]: {i+1} / {len(data_set) // BATCH_SIZE} acc: {acc} loss: {loss} ({now_str()})'
        sys.stdout.write(message)
        sys.stdout.flush()
    print('')
    print(f'epoch[{epoch}]: Done. average acc:{np.average(accs)} ({now_str()})')

    weight_path = f'./{weight_dir}/{epoch}.pt'
    state = model.module.cpu().state_dict() if MULTI_GPU else model.cpu().state_dict()
    torch.save(state, weight_path)
    print(f'save weights to {weight_path}')
    model = model.to(device)
    epoch += 1

print(f'Finished training')
