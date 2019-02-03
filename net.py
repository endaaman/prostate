import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch
# from torchvision import models
# import torchvision


TILE_SIZE = 224

def load_images(base_dir):
    case_names = os.listdir(base_dir)
    yy = []
    xx = []
    names = []
    for case_name in case_names:
        case_dir = f'{base_dir}/{case_name}'
        file_names = os.listdir(f'{case_dir}/y/')
        i = 0
        count = 0
        for file_name in file_names:
            i += 1
            y_img = Image.open(f'{case_dir}/y/{file_name}')
            y_img.load()
            y_img_array = np.asarray(y_img)[:,:,:3]
            n = np.count_nonzero(y_img_array)
            if n < 2: # skip blank label image
                continue
            yy.append(y_img)
            base_name, ext_name = os.path.splitext(file_name)
            x_img = Image.open(f'{case_dir}/x/{base_name}.jpg')
            x_img.load()
            # x_img_array = np.asarray(x_img)[:,:,:3]
            xx.append(x_img)
            names.append(base_name)
    # yy = np.array(yy)
    # xx = np.array(xx)
    return xx, yy, names


class NormalDataset(Dataset):
    def augment_image(self, img, op):
        if op > 3:
            img = ImageOps.mirror(img)
        img = img.rotate(90 * (op % 4))
        return img

    def __init__(self):
        self.xx, self.yy, self.names = load_images('./train/224')

    def __len__(self):
        return len(self.yy)

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.yy))
        op = np.random.randint(0, 8)
        x_image = self.augment_image(self.xx[i], op)
        y_image = self.augment_image(self.yy[i], op)
        return (x_image, y_image)


class LaidDataset(Dataset):
    def combine_images(self, images, operations):
        size = images[0].size[0]
        image = Image.new('RGBA', (size * 2, size * 2), (255, 255, 255))
        for i, img in enumerate(images):
            op = operations[i]
            if op > 3:
                img = ImageOps.mirror(img)
            img = img.rotate(90 * (op % 4))
            pos = ((i // 2) * size, (i % 2) * size)
            image.paste(img, pos)
        return image

    def __init__(self):
        self.xx, self.yy, self.names = load_images('./train/112')

    def __len__(self):
        return len(self.yy)

    def __getitem__(self, idx):
        ii = np.random.choice(list(range(len(self.yy))), 4, replace=False)
        x_images = [self.xx[i] for i in ii]
        y_images = [self.yy[i] for i in ii]
        names = [self.names[i] for i in ii]
        operations = np.random.choice(list(range(8)), 4)
        x_image = self.combine_images(x_images, operations)
        y_image = self.combine_images(y_images, operations)
        return (x_image, y_image)


# dataset = NormalDataset()
dataset = LaidDataset()

i = 0
for i, pair in enumerate(dataset):
    i += 1
    x, y = pair
    x.save(f'./{i}_x.png')
    y.save(f'./{i}_y.png')
    if i > 5:
        break
