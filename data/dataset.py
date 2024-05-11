import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Callable
import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

class MRIDataset(Dataset):
    def __init__(self, path: str, filter_func: Optional[Callable] = None, transform: Optional[Callable] = None, undersampled = True):
        """
        Args:
            path: Path to files
            transform: Optional callable to apply to the data (e.g., normalization, augmentation).
        """
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
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
        return image_abs

# Example transform function
def normalize_and_to_tensor(kspace, mask):
    # Normalize k-space by scaling by the max value within the k-space (a simple normalization approach)
    kspace_abs = np.abs(kspace)
    max_val = np.max(kspace_abs)
    kspace_normalized = kspace / max_val if max_val > 0 else kspace

    # Convert to PyTorch tensor
    kspace_tensor = torch.from_numpy(kspace_normalized).float()
    mask_tensor = torch.from_numpy(mask).float() if mask is not None else None

    return kspace_tensor, mask_tensor
