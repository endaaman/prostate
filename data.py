import os
import numpy as np
import cv2
import scipy.ndimage
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
TILE_SIZE = 224


class BaseDataset(Dataset):
    def __init__(self, transform_x=None, transform_y=None):
        self.transform_x = transform_x
        self.transform_y = transform_y
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

    def transform(self, x, y):
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        return x, y

    def select(self):
        p = []
        for i in self.y_raws:
            p.append((i.shape[0] - TILE_SIZE) * (i.shape[1] - TILE_SIZE))
        p = np.array(p / np.sum(p))
        use_patch = False
        while not use_patch:
            i = np.random.choice(len(self.y_raws), 1, p=p)[0]
            y_raw = self.y_raws[i]
            x_raw = self.x_raws[i]
            image_h, image_w, _ = y_raw.shape
            left = np.random.randint(image_w - TILE_SIZE)
            top = np.random.randint(image_h - TILE_SIZE)
            y_arr = y_raw[top:top + TILE_SIZE, left:left + TILE_SIZE]
            use_patch = np.any(y_arr != 0)
        x_arr = x_raw[top:top + TILE_SIZE, left:left + TILE_SIZE]
        return (x_arr, y_arr)


class DefaultDataset(BaseDataset):
    def __init__(self, p_rotation=-1, *args, **kwargs):
        super(DefaultDataset, self).__init__(*args, **kwargs)
        self.p_rotation = p_rotation

    def flip_and_rot90(self, arr, op):
        if op > 3:
            arr = np.flip(arr, axis=0)
        return np.rot90(arr, op % 4)

    def rotate(self, arr, degree):
        i = scipy.ndimage.rotate(arr, degree, mode='mirror')
        return i

    def __len__(self):
        l = 0
        for i in self.y_raws:
            l += (i.shape[0] // TILE_SIZE) * (i.shape[1] // TILE_SIZE)
        return l * 8

    def __getitem__(self, _idx):
        x_arr, y_arr = self.select()
        op = np.random.randint(8)
        x_arr = self.flip_and_rot90(x_arr, op)
        y_arr = self.flip_and_rot90(y_arr, op)
        if np.random.rand() < self.p_rotation:
            degree = np.random.randint(45)
            x_arr = self.rotate(x_arr, degree)
            y_arr = self.rotate(y_arr, degree)
            h, w = x_arr.shape[0:2]
            top = np.random.randint(h - TILE_SIZE) if h > TILE_SIZE else 0
            left = np.random.randint(w - TILE_SIZE) if w > TILE_SIZE else 0
            x_arr = x_arr[top:top + TILE_SIZE, left:left + TILE_SIZE]
            y_arr = y_arr[top:top + TILE_SIZE, left:left + TILE_SIZE]
        return self.transform(x_arr.copy(), y_arr.copy())
