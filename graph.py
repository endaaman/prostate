import os
import sys
import time
import math
import cv2
import re
import json
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.transforms import ToTensor, Normalize, Compose, CenterCrop

from models import UNet11, UNet16
from store import Store
from metrics import Metrics, Coef
from datasets import ValidationDataset
from utils import now_str, pp

Image.MAX_IMAGE_PIXELS = 1000000000


def compare():
    f1 = ',/unet11.json'
    f2 = ',/unet16n.json'

    with open(f1, 'r') as f:
        data1 = json.load(f)

    with open(f2, 'r') as f:
        data2 = json.load(f)

    train_metrics = Metrics()
    val_metrics = Metrics()

    print(f1, f2)
    for i, item1 in enumerate(data1['items']):
        item2 = data2['items'][i]
        coef1 = Coef(**item1['coef'])
        coef2 = Coef(**item2['coef'])
        print(item1['name'], coef1.pjac, coef2.pjac)
        print()
        continue

        if re.match(r'mean', item['type']):
            continue
        if item['type'] == 'train':
            m = train_metrics
        if item['type'] == 'val':
            m = val_metrics
        m.append_coef(coef)

    exit()
    print(fp)
    print('train: ', train_metrics.avg_coef().to_str())
    print('val  : ', val_metrics.avg_coef().to_str())
    print()


def show(f1):
    print(f1)
    with open(f1, 'r') as f:
        data1 = json.load(f)

    train_metrics = Metrics()
    val_metrics = Metrics()
    for i, item in enumerate(data1['items']):
        coef = Coef(**item['coef'])
        if re.match(r'mean', item['type']):
            # if item['type'] == 'mean train':
            #     print('mean : ', coef.to_str())
            continue
        if item['type'] == 'train':
            m = train_metrics
        if item['type'] == 'val':
            m = val_metrics
        m.append_coef(coef)

    print('train: ', train_metrics.avg_coef().to_str())
    print('val  : ', val_metrics.avg_coef().to_str())
    print()


def get_avg_coef(fp):
    with open(fp, 'r') as f:
        data = json.load(f)
    tm, vm = Metrics(), Metrics()
    for i, item in enumerate(data['items']):
        coef = Coef(**item['coef'])
        if re.match(r'mean', item['type']):
            continue
        if item['type'] == 'train':
            tm.append_coef(coef)

        if item['type'] == 'val':
            vm.append_coef(coef)

    return tm.avg_coef(), vm.avg_coef()



LIST = [
        ['unet11', 'VGG11-deconv'],
        ['unet11b', 'VGG11-bilinear'],
        ['unet11n', 'VGG11-nearest'],
        ['unet16', 'VGG16-deconv'],
        ['unet16b', 'VGG16-bilinear'],
        ['unet16n', 'VGG16-nearest'],
        ['albunet', 'ResNet34-deconv'],
        ['albunet_b', 'ResNet34-bilinear'],
        ['albunet_n', 'ResNet34-nearest'],
    ]

def list1():
    label = 'IoU'
    target = 'pjac'
    # label = 'tumor sensitivity'
    # target = 'tsensi'

    items = []
    for n in LIST:
        train_coef, val_coef = get_avg_coef(f',/gen3/768/{n[0]}/report.json')
        items.append([n[1], getattr(train_coef, target)])

    # items = sorted(items, key=lambda x:x[1])
    x_position = np.arange(len(l))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(x_position, [v[1] for v in items], width=0.8, label=label)
    for x, (name, v) in zip(x_position, items):
        plt.text(x, 0.1, name, rotation=90, va='bottom', ha='left')
        plt.text(x - 0.4, v + 0.02, f'{v:.3f}')

    ax.legend()
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0.5, 1, 6))
    # ax.set_xticklabels(x)
    ax.grid(axis='y')
    ax.tick_params(axis='x',labelbottom=False)
    plt.show()


def list2():
    items = []
    labels = ['IoU', 'gland sensitivity',  'tumor sensitivity',  'gland specificity',  'tumor specificity', ]
    targets = ['pjac', 'gsensi', 'tsensi', 'gspec', 'tspec', ]
    for n in LIST:
        train_coef, val_coef = get_avg_coef(f',/gen3/768/{n[0]}/report.json')
        items.append([n[1]] + [getattr(train_coef, t) for t in targets])

    # items = sorted(items, key=lambda x:x[1])
    x_position = np.arange(len(LIST))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    w = 0.7 / len(targets)
    for i, (t, labe) in enumerate(zip(targets, labels)):
        ax.barh(x_position + i * w, [v[i+1] for v in items], height=w, label=labe)

    for x, item in zip(x_position, items):
        plt.text(0.1, x - 0.1, item[0], va='bottom', ha='left')
        for i, v in enumerate(item[1:]):
            plt.text(v, x + i * w + w / 2, f'{v:.3f}', va='bottom', ha='left')

    ax.legend(bbox_to_anchor=(0, -0.05), loc='upper left', borderaxespad=0)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0.5, 1.1, 7))
    ax.grid(axis='x')
    ax.tick_params(axis='y', labelbottom=False)
    plt.show()


list2()
