import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader
Image.MAX_IMAGE_PIXELS = 1000000000
TILE_SIZE = 224


def load_images(base_dir):
    case_names = os.listdir(base_dir)
    case_names.sort()
    yy = []
    xx = []
    names = []
    for case_name in case_names:
        case_dir = f'{base_dir}/{case_name}'
        file_names = os.listdir(f'{case_dir}/y/')
        file_names.sort()
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
    return xx, yy, names


class BaseDataset(Dataset):
    def flip_and_rotate_image(self, img, op):
        if op > 3:
            img = ImageOps.mirror(img)
        img = img.rotate(90 * (op % 4))
        return img

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


class NormalDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super(NormalDataset, self).__init__(*args, **kwargs)
        self.xx, self.yy, self.names = load_images('./train/224')

    def __len__(self):
        return len(self.yy) * 8

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.yy))
        op = idx // self.image_count
        x_image = self.flip_and_rotate_image(self.xx[i], op)
        y_image = self.flip_and_rotate_image(self.yy[i], op)
        return self.transform(x_image, y_image)


class LaidDataset(BaseDataset):
    def combine_images(self, images, mode, operations):
        size = images[0].size[0]
        image = Image.new(mode, (size * 2, size * 2), (255, 255, 255))
        for i, img in enumerate(images):
            op = operations[i]
            if op > 3:
                img = ImageOps.mirror(img)
            img = img.rotate(90 * (op % 4))
            pos = ((i // 2) * size, (i % 2) * size)
            image.paste(img, pos)
        return image

    def __init__(self, *args, **kwargs):
        super(LaidDataset, self).__init__(*args, **kwargs)
        self.xx, self.yy, self.names = load_images('./train/112')

    def __len__(self):
        # return int(1e10)
        return len(self.yy)

    def __getitem__(self, idx):
        ii = np.random.choice(list(range(len(self.yy))), 4, replace=False)
        x_images = [self.xx[i] for i in ii]
        y_images = [self.yy[i] for i in ii]
        # names = [self.names[i] for i in ii]
        operations = np.random.choice(list(range(8)), 4)
        x_image = self.combine_images(x_images, 'RGB', operations)
        y_image = self.combine_images(y_images, 'RGBA', operations)
        return self.transform(x_image, y_image)


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
            x_raw = Image.open(f'{base_dir}/x/{base_name}.jpg')
            y_raw = Image.open(f'{base_dir}/y/{base_name}.png')
            self.x_raws.append(np.asarray(x_raw))
            self.y_raws.append(np.asarray(y_raw))
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
        x_arr = self.flip_and_rotate_array(x_arr, op)
        y_arr = self.flip_and_rotate_array(y_arr, op)
        return self.transform(x_arr, y_arr)
