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
from utils.visualization import show_image

class TestDataset(Dataset):
        

    def __init__(self, path: str, filter_func: Optional[Callable] = None, transform: Optional[Callable] = None, undersampled = True, cat = True):
        """
        Args:
            path: Path to files
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
        """
        self.cat = cat
        if cat: 
            self.samples = ["./cat_greyscale.png"]

            self.image = Image.open(self.samples[0])

            self.image = np.array(self.image, dtype=np.float32)  # Ensure data type is float for precision in normalization

            # Convert the image to a tensor
            self.image = torch.from_numpy(self.image)

            # Normalize the image to range [-1, 1]
            min_val = self.image.min()
            max_val = self.image.max()
            self.image = 2 * (self.image - min_val) / (max_val - min_val) - 1 

        else: 
            self.path = path
            self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')]
            self.transform = transform
            self.samples = []
            self.undersampled = undersampled

            self._prepare_dataset(filter_func)

    def _prepare_dataset(self, filter_func: Optional[Callable] = None):
        """ Prepare the dataset by listing all file paths and the number of slices per file. """
        for file_path in self.files:
            if (filter_func and filter_func(file_path)) or not filter_func:
                with h5py.File(file_path, 'r') as hf:
                    num_slices = hf['kspace'].shape[0]
                    for s in range(num_slices):
                        self.samples.append((file_path, s))
                        return


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # load png image 

        if self.cat: 
            return self.image
        else: 
            file_path, slice_idx = self.samples[idx]
            with h5py.File(file_path, 'r') as hf:
                kspace = np.asarray(hf['kspace'][slice_idx])
                kspace_tensor = T.to_tensor(kspace)      # Convert from numpy array to pytorch tensor

                if self.undersampled: 
                    mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
                    kspace_tensor, _, _ = T.apply_mask(kspace_tensor, mask_func)

                image = fastmri.ifft2c(kspace_tensor)           # Apply Inverse Fourier Transform to get the complex image
                image_abs = fastmri.complex_abs(image) 

            image_abs = image_abs.float()

            # apply transformations 
            if self.transform:
                image_abs = self.transform(image_abs)

            return image_abs
            
    
    def scale_mri_tensor_advanced(img_tensor):
        mean_val = img_tensor.mean()
        std_dev = img_tensor.std()

        # Scale based on standard deviation, e.g., mean ± 3*std_dev maps to [-1, 1]
        three_std_from_mean = 3 * std_dev
        img_tensor = (img_tensor - mean_val) / three_std_from_mean

        # Clipping the values to be within [-1, 1]
        img_tensor = torch.clamp(img_tensor, -1, 1)

        return img_tensor


    def __len__(self):
        return len(self.samples)