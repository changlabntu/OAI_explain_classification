from os.path import join
import glob
import random

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
import os
from skimage import io
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import tifffile as tiff
import pandas as pd
import random


def to_8bit(x):
    if type(x) == torch.Tensor:
        x = (x / x.max() * 255).numpy().astype(np.uint8)
    else:
        x = (x / x.max() * 255).astype(np.uint8)

    if len(x.shape) == 2:
        x = np.concatenate([np.expand_dims(x, 2)]*3, 2)
    return x


def imagesc(x, show=True, save=None):
    # switch
    if (len(x.shape) == 3) & (x.shape[0] == 3):
        x = np.transpose(x, (1, 2, 0))

    if isinstance(x, list):
        x = [to_8bit(y) for y in x]
        x = np.concatenate(x, 1)
        x = Image.fromarray(x)
    else:
        x = x - x.min()
        x = Image.fromarray(to_8bit(x))
    if show:
        x.show()
    if save:
        x.save(save)


class PairedDataTif(data.Dataset):
    def __init__(self, root, path, labels=None, crop=None, mode='train'):
        self.directions = path.split('_')
        self.labels = labels
        self.root = root
        self.crop = crop
        self.mode = mode

        # Get the list of file names from the first folder
        folder_a = os.path.join(root, self.directions[0])
        self.file_names = sorted([f for f in os.listdir(folder_a) if f.endswith('.tif')])

        # Verify that matching files exist in other folders
        for direction in self.directions[1:]:
            folder = os.path.join(root, direction)
            for file_name in self.file_names:
                if not os.path.exists(os.path.join(folder, file_name)):
                    raise FileNotFoundError(f"File {file_name} not found in {folder}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        image_list = []

        for direction in self.directions:
            img_path = os.path.join(self.root, direction, file_name)
            image = tiff.imread(img_path)
            image = image / image.max()

            if self.crop:
                if self.mode == 'test':
                    dx = (image.shape[1] - self.crop) // 2
                    image = image[:, dx:-dx, dx:-dx]
                elif self.mode == 'train':
                    dx = np.random.randint(0, (image.shape[1] - self.crop) // 2)
                    dx2 = image.shape[1] - self.crop - dx
                    dy = np.random.randint(0, (image.shape[1] - self.crop) // 2)
                    dy2 = image.shape[2] - self.crop - dy
                    image = image[:, dx:-dx2, dy:-dy2]

            # permute (Z, X, Y) > (C, X, Y, Z)
            image = torch.from_numpy(image).permute(1, 2, 0).unsqueeze(0).float()

            image_list.append(image)

        # Stack images into a single tensor
        paired_images = image_list

        return {'img': paired_images, 'labels': self.labels[index]}


if __name__ == '__main__':
    from dotenv import load_dotenv
    import argparse
    load_dotenv('env/.t09')

    eval_set = PairedDataTif(root='/media/ghc/Ghc_data3/OAI_diffusion_final/diffusion_classification/',
                       path='a_b', labels=None, crop=False)
    x = eval_set.__getitem__(10)
    print(x[0].shape)
