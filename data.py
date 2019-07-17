import os
from errno import ENOENT
import numpy as np
import cv2
import scipy.ndimage
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def read_image(self, name):
        raw = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        if not type(raw) is np.ndarray:
            raise FileNotFoundError(ENOENT, os.strerror(ENOENT), name)
        return raw

    def read_images(self, target_dir, one=False):
        names = []
        x_raws = []
        y_raws = []
        file_names = sorted(os.listdir(os.path.join(target_dir, 'y')))
        for file_name in file_names:
            base_name, ext_name = os.path.splitext(file_name)
            x_raws.append(self.read_image(os.path.join(target_dir, 'x', f'{base_name}.jpg')))
            y_raws.append(self.read_image(os.path.join(target_dir, 'y', f'{base_name}.png')))
            names.append(base_name)
            if one:
                break
        return x_raws, y_raws, names

    def __init__(self, target_dir='./train', transform_x=None, transform_y=None, one=False):
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.x_raws, self.y_raws, self.names = self.read_images(target_dir, one)

    def transform(self, x, y):
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        return x, y


class TrainingDataset(BaseDataset):
    def __init__(self, tile_size, p_rotation=-1, stricted_roi=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size
        self.p_rotation = p_rotation
        self.stricted_roi = stricted_roi

    def flip_and_rot90(self, arr, op):
        if op > 3:
            arr = np.flip(arr, axis=0)
        return np.rot90(arr, op % 4)

    def rotate(self, arr, degree):
        i = scipy.ndimage.rotate(arr, degree, mode='mirror')
        return i

    def check_available(self, arr):
        return self.stricted_roi or np.any(arr != 0)

    def select(self):
        p = []
        for i in self.y_raws:
            p.append((i.shape[0] - self.tile_size) * (i.shape[1] - self.tile_size))
        p = np.array(p / np.sum(p))
        use_patch = False
        size = self.tile_size
        while not use_patch:
            i = np.random.choice(len(self.y_raws), 1, p=p)[0]
            y_raw = self.y_raws[i]
            x_raw = self.x_raws[i]
            image_h, image_w, _ = y_raw.shape
            left = np.random.randint(image_w - size)
            top = np.random.randint(image_h - size)
            y_arr = y_raw[top:top + size, left:left + size]
            use_patch = self.check_available(y_arr)
        x_arr = x_raw[top:top + size, left:left + size]
        return (x_arr, y_arr)

    def __len__(self):
        l = 0
        for i in self.y_raws:
            l += int((i.shape[0] / self.tile_size) * (i.shape[1] / self.tile_size))
        return l * 8

    def __getitem__(self, _idx):
        x_arr, y_arr = self.select()
        op = np.random.randint(8)
        x_arr = self.flip_and_rot90(x_arr, op)
        y_arr = self.flip_and_rot90(y_arr, op)
        size = self.tile_size
        if np.random.rand() < self.p_rotation:
            degree = np.random.randint(45)
            x_arr = self.rotate(x_arr, degree)
            y_arr = self.rotate(y_arr, degree)
            h, w = x_arr.shape[0:2]
            top = np.random.randint(h - size) if h > size else 0
            left = np.random.randint(w - size) if w > size else 0
            x_arr = x_arr[top:top + size, left:left + size]
            y_arr = y_arr[top:top + size, left:left + size]
        return self.transform(x_arr.copy(), y_arr.copy())


class ValidationDataset(BaseDataset):
    def __init__(self, max_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = max_size
        x_raws, y_raws, names = self.read_images('./validation', False)
        self.is_train_flags = [True] * len(self.names) + [False] * len(names)
        self.x_raws += x_raws
        self.y_raws += y_raws
        self.names += names

    def __len__(self):
        return len(self.y_raws)

    def __getitem__(self, i):
        x_raw, y_raw = self.x_raws[i], self.y_raws[i]
        H, W = x_raw.shape[:2]
        Y = -(-x_raw.shape[0] // self.max_size)
        X = -(-x_raw.shape[1] // self.max_size)
        ww = [(W + i) // X for i in range(X)]
        hh = [(H + i) // Y for i in range(Y)]
        x_data, y_data = [], []
        pos = [0, 0]
        for y, h in enumerate(hh):
            pos[0] = 0
            x_data.append([])
            y_data.append([])
            for x, w in enumerate(ww):
                x_data[-1].append(x_raw[pos[1]:pos[1]+h, pos[0]:pos[0]+w].copy())
                y_data[-1].append(y_raw[pos[1]:pos[1]+h, pos[0]:pos[0]+w].copy())
                pos[0] += w
            pos[1] += h
        return (self.names[i], x_raw, y_raw, x_data, y_data, self.is_train_flags[i])
