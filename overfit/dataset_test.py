import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc
from PIL import Image

class TestDataset(Dataset):
    def __init__(self):
        """
        Args:
            path: Path to files
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
        """
        self.samples = ["./cat.png"]

        self.image = Image.open(self.samples[0])
        self.image = np.array(self.image, dtype=np.float32)  # Ensure data type is float for precision in normalization

        # Convert the image to a tensor
        self.image = torch.from_numpy(self.image)

        # Normalize the image to range [-1, 1]
        min_val = self.image.min()
        max_val = self.image.max()
        self.image = 2 * (self.image - min_val) / (max_val - min_val) - 1 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load png image 

        return self.image