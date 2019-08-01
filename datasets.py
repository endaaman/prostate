import os
import math
from errno import ENOENT
import numpy as np
import cv2
import scipy.ndimage
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader



def read_image(name):
    raw = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    if not type(raw) is np.ndarray:
        raise FileNotFoundError(ENOENT, os.strerror(ENOENT), name)
    return raw

class Item():
    def __init__(self, target_dir, file_name, is_train=True):
        base_name, ext_name = os.path.splitext(file_name)
        self.name = base_name
        self.x_raw = read_image(os.path.join(target_dir, 'x', f'{base_name}.jpg'))
        self.y_raw = read_image(os.path.join(target_dir, 'y', f'{base_name}.png'))
        assert(self.x_raw.shape[:2] == self.y_raw.shape[:2])
        self.is_train = is_train

    def get_splitted(self, size):
        H, W = self.x_raw.shape[:2]
        Y = -(-self.x_raw.shape[0] // size)
        X = -(-self.x_raw.shape[1] // size)
        ww = [(W + i) // X for i in range(X)]
        hh = [(H + i) // Y for i in range(Y)]
        pairs = []
        pos = [0, 0]
        for y, h in enumerate(hh):
            pos[0] = 0
            pairs.append([])
            for x, w in enumerate(ww):
                x = self.x_raw[pos[1]:pos[1]+h, pos[0]:pos[0]+w].copy()
                y = self.y_raw[pos[1]:pos[1]+h, pos[0]:pos[0]+w].copy()
                pairs[-1].append((x, y))
                pos[0] += w
            pos[1] += h
        return pairs

def read_images(target_dir, one=False, is_train=True):
    items = []
    file_names = sorted(os.listdir(os.path.join(target_dir, 'y')))
    for file_name in file_names:
        items.append(Item(target_dir, file_name, is_train))
        if one:
            break
    return items

class BaseDataset(Dataset):
    def __init__(self, target_dir='./train', transform_x=None, transform_y=None, one=False, load_train=True):
        self.transform_x = transform_x
        self.transform_y = transform_y
        self.items = read_images(target_dir, one) if load_train else []

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
        return (not self.stricted_roi) and np.any(arr != 0)

    def select(self):
        p = []
        for i in self.items:
            # p.append((i.shape[0] - self.tile_size) * (i.shape[1] - self.tile_size))
            p.append(math.sqrt((i.x_raw.shape[0] - self.tile_size) * (i.x_raw.shape[1] - self.tile_size)))
        p = np.array(p / np.sum(p))
        use_patch = False
        size = self.tile_size
        while not use_patch:
            i = np.random.choice(len(self.items), 1, p=p)[0]
            y_raw = self.items[i].y_raw
            x_raw = self.items[i].x_raw
            image_h, image_w, _ = y_raw.shape
            left = np.random.randint(image_w - size)
            top = np.random.randint(image_h - size)
            y_arr = y_raw[top:top + size, left:left + size]
            use_patch = self.check_available(y_arr)
        x_arr = x_raw[top:top + size, left:left + size]
        return (x_arr, y_arr)

    def __len__(self):
        l = 0
        for i in self.items:
            l += int((i.x_raw.shape[0] / self.tile_size) * (i.x_raw.shape[1] / self.tile_size))
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
    def __init__(self, load_val=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        val_dir = './validation'
        if load_val and os.path.isdir(val_dir):
            self.items += read_images(val_dir, one=False, is_train=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

if __name__ == '__main__':
    ds = ValidationDataset(one=True)
    for item in ds:
        print(item.name, item.x_raw.shape)
        for y, row in enumerate(item.get_splitted(1000)):
            print(type(row))
            for x, (input_arr, label_arr) in enumerate(row):
                print(type(input_arr))
                break
