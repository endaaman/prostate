import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import ToTensor, Normalize, Compose

from net import UNet11
from data import LaidDataset


BATCH_SIZE = 2
NUM_WORKERS = 1
NUM_CLASSES = 3


I = np.identity(NUM_CLASSES, dtype=np.float32)

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



data_set = LaidDataset(
        transform_x = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        transform_y = transform_y)
data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


model = UNet11(num_classes=NUM_CLASSES, pretrained=True)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCELoss()

for epoch in range(2):  # loop over the dataset multiple times
    total_loss = 0.0
    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = Variable(inputs), Variable(labels)
        print(inputs.type())
        print(labels.type())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()
        loss += loss.data[0]
        total_loss += loss.data[0]
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 2000))
            total_loss = 0.0

print('Finished Training')
