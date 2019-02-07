import sys
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose

from net import UNet11
from data import LaidDataset, RandomPatchDataset

BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_CLASSES = 3
EPOCH_COUNT = 100

first_epoch = 1
weight_file = None
if len(sys.argv) > 1:
    weight_file = sys.argv[1]
    num = os.path.splitext(os.path.basename(weight_file))[0]
    if not num.isdigit():
        print(f'Invalid pt file')
        exit(1)
    first_epoch = int(num) + 1


I = np.identity(NUM_CLASSES, dtype=np.float32)

def now_str():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def one_hot(x):
    if x[3] == 0:
         # transparent -> no label
        return I[0]
    if x[0] == 0 and x[1] == 0 and x[2] == 255:
        # Blue -> Gleason 3
        return I[2]
    # Other -> Gleason 1
    return I[1]

def transform_y(img):
    arr = np.asarray(img)
    arr2 = np.apply_along_axis(one_hot, 2, arr)
    return ToTensor()(arr2)

data_set = RandomPatchDataset(
        transform_x = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transform_y = transform_y)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model = UNet11(num_classes=NUM_CLASSES)
if weight_file:
    model.load_state_dict(torch.load(weight_file))
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True


optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

print(f'Start training ({now_str()})')
epoch = first_epoch
weight_path = None
while epoch <= EPOCH_COUNT:
    train_loss = 0.0
    message = None
    for i, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if message:
            sys.stdout.write('\r' * len(message))
        message = f'epoch[{epoch}]: {i} / {len(data_set) // BATCH_SIZE} loss: {loss} ({now_str()})'
        sys.stdout.write(message)
        sys.stdout.flush()
    print('')
    print(f'epoch[{epoch}]: Done ({now_str()})')

    if weight_path and os.path.exists(weight_path):
      os.remove(weight_path)
    weight_path = f'./weights/{epoch}.pt'
    torch.save(model.cpu().state_dict(), weight_path)
    print(f'save weights to {weight_path}')
    model = model.to(device)
    epoch += 1

print(f'Finished training')
