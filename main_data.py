import os 
import h5py
import torch 
from typing import List, Optional, Callable

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

from data.dataset import TransformedMRIDataset

def create_h5py_dataset(path: str, filter_func: Optional[Callable] = None, undersampled = False, number_of_samples = -1, train: bool = True) -> None: 
    
    train_transformed_path = os.path.join(os.path.dirname(os.getcwd()), path)
    
    if train: 
        transformed_dataset = "transformed_dataset/brain/singlecoil_train" # Name and location of folder in which transformed train images are saved  
        transformed_dataset_h5 = "brain_train_transformed.h5" # Name of h5 file in which transformed images are saved
    else: 
        transformed_dataset = "transformed_dataset/brain/singlecoil_val" # Name and location of folder in which transformed validation images are saved  
        transformed_dataset_h5 = "brain_val_transformed.h5" # Name of h5 file in which transformed images are saved
    
    train_transformed_path = os.path.join(os.path.dirname(os.getcwd()), transformed_dataset)
    
    if not os.path.exists(train_transformed_path): 
        os.makedirs(train_transformed_path)
    
    try: 
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5') and filter_func(f)]
    except: 
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.h5')]
       
    dataset_list = []    
    for file_path in files: 
        with h5py.File(file_path, 'r') as hf_r:
            if 0 < number_of_samples < hf_r['kspace'].shape[0]: 
                 volume_kspace = hf_r['kspace'][:number_of_samples]
            else: 
                volume_kspace = hf_r['kspace'][()]
                                
            vol_kspace2 = T.to_tensor(volume_kspace) # Convert from numpy array to pytorch tensor
                      
            if undersampled: 
                mask_func = RandomMaskFunc(center_fractions=[0.1], accelerations=[8])
                vol_kspace2, _, _ = T.apply_mask(vol_kspace2, mask_func)
            
            vol_image = fastmri.ifft2c(vol_kspace2) # Apply Inverse Fourier Transform to get the complex image
            vol_image_abs = fastmri.complex_abs(vol_image)  # Compute absolute value to get a real imagep
                            
            vol_image_norm = 2 * (vol_image_abs - torch.min(vol_image_abs) / (torch.max(vol_image_abs) - torch.min(vol_image_abs))) - 1 # Normalize image to [-1, 1]
            dataset_list.append(vol_image_norm)
    
    with h5py.File(os.path.join(train_transformed_path, transformed_dataset_h5), "w") as hf_w: 
        dataset_tensor = torch.cat(tuple(dataset_list), dim=0)
        # print(dataset_tensor.shape)
        hf_w.create_dataset("MRI_tensor", data=dataset_tensor)  

def main(): 
    number_of_samples = 10
    filter_func = (lambda x: 'FLAIR' in x)
    
    train_path = "../dataset/brain/singlecoil_train/"
    val_path = "../dataset/brain/singlecoil_val/"
    
    create_h5py_dataset(path=train_path, filter_func=filter_func, number_of_samples=number_of_samples, train=True)
    create_h5py_dataset(path=val_path, filter_func=filter_func, number_of_samples=number_of_samples, train=False)
    
    train_dataset = TransformedMRIDataset(path="../transformed_dataset/brain/singlecoil_train/")
    # val_dataset = TransformedMRIDataset(path="../transformed_dataset/brain/singlecoil_train/")

    for i in range(len(train_dataset)): 
        print(train_dataset[i].shape)

if __name__ == '__main__':
    main()