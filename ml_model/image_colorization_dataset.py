import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageColorizationDataset(Dataset):
    def __init__(self, black_dir, color_dir, transform=None):
        """
        Args: black_dir (string): Directory with black and white images color_dir (string): Directory with colored
        images. Both directories should have the same number of images with the same names. transform (callable,
        optional): Optional transform to be applied on a sample.
        """
        self.black_dir = black_dir
        self.color_dir = color_dir
        self.transform = transform
        self.black_images = os.listdir(black_dir)
        self.color_images = os.listdir(color_dir)

    def __len__(self):
        return len(self.black_images)

    def __getitem__(self, idx):
        black_img_name = os.path.join(self.black_dir, self.black_images[idx])
        color_img_name = os.path.join(self.color_dir, self.color_images[idx])
        black_image = Image.open(black_img_name).convert('L')  # Convert to grayscale
        color_image = Image.open(color_img_name).convert('RGB')  # Ensure color

        if self.transform:
            black_image = self.transform(black_image)
            color_image = self.transform(color_image)

        sample = {'black_image': black_image, 'color_image': color_image}

        return sample
