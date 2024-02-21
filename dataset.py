# %%
import os
import torch
import torchvision
from torch.utils import data
import numpy as np
from PIL import Image
import glob
from typing import Any
try:
    import accimage
    torchvision.set_image_backend('accimage')
except:
    pass
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp',
                  '.pgm', '.tif', '.tiff', '.webp')


class CSdataset():
    def __init__(self, file_path='dataset', transform=None):
        if transform is not None:
            self.transform = transform
        else:
            self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        self.train_data = np.load(file_path)
        self.data_len = len(self.train_data)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img = Image.fromarray(self.train_data[index])
        img = self.transform(img)
        label = img

        return img, label


def pil_loader(path: str) -> Image.Image:
    return Image.open(path)


def accimage_loader(path: str) -> Any:
    # import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class CustomDataset(data.Dataset):
    def __init__(self, file_path, transforms=None):
        self.images = glob.glob(os.path.join(file_path, '*'))
        if transforms is None:
            self.transforms = torchvision.transforms.ToTensor()
        else:
            self.transforms = transforms

    def __getitem__(self, index):
        img = self.transforms(default_loader(self.images[index]))
        label = img
        return img, label

    def __len__(self):
        return len(self.images)
