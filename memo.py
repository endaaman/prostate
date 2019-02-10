import sys
import time
import os
import datetime
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
from data import LaidDataset, RandomPatchDataset
from torchvision.transforms import ToTensor, Normalize, Compose
Image.MAX_IMAGE_PIXELS = 1000000000


def dice_coef(y_true, y_pred):
    y_true = y_true.view(-1, y_true.size(-1))
    y_pred = y_pred.view(-1, y_pred.size(-1))
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)

