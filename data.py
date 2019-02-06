import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
from torchvision.transforms import ToTensor, Normalize, Compose
from torch.utils.data import Dataset, DataLoader


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
    # yy = np.array(yy)
    # xx = np.array(xx)
    return xx, yy, names


class NormalDataset(Dataset):
    def augment_image(self, img, op):
        if op > 3:
            img = ImageOps.mirror(img)
        img = img.rotate(90 * (op % 4))
        return img

    def __init__(self, transform_x=None, transform_y=None):
        self.xx, self.yy, self.names = load_images('./train/224')
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __len__(self):
        return len(self.yy) * 8

    def __getitem__(self, idx):
        i = np.random.randint(0, len(self.yy))
        op = idx // self.image_count
        x_image = self.augment_image(self.xx[i], op)
        y_image = self.augment_image(self.yy[i], op)
        if self.transform_x:
            x_image = self.transform_x(x_image)
        if self.transform_y:
            y_image = self.transform_y(y_image)
        return (x_image, y_image)


class LaidDataset(Dataset):
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

    def __init__(self, transform_x=None, transform_y=None):
        self.xx, self.yy, self.names = load_images('./train/112')
        self.transform_x = transform_x
        self.transform_y = transform_y

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
        if self.transform_x:
            x_image = self.transform_x(x_image)
        if self.transform_y:
            y_image = self.transform_y(y_image)
        return (x_image, y_image)
