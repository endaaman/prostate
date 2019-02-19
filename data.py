import os
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
Image.MAX_IMAGE_PIXELS = 1000000000
TILE_SIZE = 224


class BaseDataset(Dataset):
    def flip_and_rotate_array(self, arr, op):
        if op > 3:
            arr = np.flip(arr, axis=0)
        return np.rot90(arr, op % 4)

    def __init__(self, transform_x=None, transform_y=None):
        self.transform_x = transform_x
        self.transform_y = transform_y

    def transform(self, x, y):
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        return x, y


class RandomPatchDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(RandomPatchDataset, self).__init__(*args, **kwargs)
        base_dir = './train/full'
        file_names = os.listdir(f'{base_dir}/y')
        self.names = []
        self.x_raws = []
        self.y_raws = []
        for file_name in file_names:
            base_name, ext_name = os.path.splitext(file_name)
            x_raw = cv2.imread(f'{base_dir}/x/{base_name}.jpg')
            y_raw = cv2.imread(f'{base_dir}/y/{base_name}.png', cv2.IMREAD_UNCHANGED)
            self.x_raws.append(x_raw)
            self.y_raws.append(y_raw)
            self.names.append(base_name)

    def __len__(self):
        return (10000 // 224) * (10000 // 224) * 8

    def __getitem__(self, _idx):
        i = np.random.randint(len(self.y_raws))
        y_raw = self.y_raws[i]
        x_raw = self.x_raws[i]
        use_patch = False
        while not use_patch:
            image_h, image_w, _ = y_raw.shape
            left = np.random.randint(image_w - TILE_SIZE)
            top = np.random.randint(image_h - TILE_SIZE)
            y_arr = y_raw[top:top + TILE_SIZE, left:left + TILE_SIZE]
            use_patch = np.any(y_arr != 0)
        x_arr = x_raw[top:top + TILE_SIZE, left:left + TILE_SIZE]
        op = np.random.randint(8)
        x_arr = self.flip_and_rotate_array(x_arr, op).copy()
        y_arr = self.flip_and_rotate_array(y_arr, op).copy()
        return self.transform(x_arr, y_arr)

DefaultDataset = RandomPatchDataset
