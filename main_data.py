import os 
import h5py
import torch 

import fastmri
from fastmri.data import transforms as T

def main(): 
    number_of_samples = 10
    filter_func = (lambda x: 'FLAIR' in x)
    train_path = "../dataset/brain/singlecoil_train/"
    
    transformed_dataset = "transformed_dataset/brain/singlecoil_train" # Name and location of folder in which transformed images are saved  
    train_transformed_path = os.path.join(os.path.dirname(os.getcwd()), transformed_dataset)
    transformed_dataset_h5 = "file_brain_transformed.h5" # Name of h5 file in which transformed images are saved
    
    if not os.path.exists(train_transformed_path): 
        os.makedirs(train_transformed_path)
    
    try: 
        files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.h5') and filter_func(f)]
    except: 
        files = [os.path.join(train_path, f) for f in os.listdir(train_path) if f.endswith('.h5')]
        
    dataset_list = []
    
    for file_path in files: 
        with h5py.File(file_path, 'r') as hf_r:
            if 0 < number_of_samples < hf_r['kspace'].shape[0]: 
                 volume_kspace = hf_r['kspace'][:number_of_samples]
            else: 
                volume_kspace = hf_r['kspace'][()]
                                
            vol_kspace2 = T.to_tensor(volume_kspace) # Convert from numpy array to pytorch tensor
            vol_image = fastmri.ifft2c(vol_kspace2) # Apply Inverse Fourier Transform to get the complex image
            vol_image_abs = fastmri.complex_abs(vol_image)  # Compute absolute value to get a real imagep
                            
            vol_image_norm = 2 * (vol_image_abs - torch.min(vol_image_abs) / (torch.max(vol_image_abs) - torch.min(vol_image_abs))) - 1 # Normalize image to [-1, 1]
            dataset_list.append(vol_image_norm)
                
    with h5py.File(os.path.join(train_transformed_path, transformed_dataset_h5), "w") as hf_w: 
        dataset_tensor = torch.cat(tuple(dataset_list), dim=0)
        print(dataset_tensor[0].shape)
        hf_w.create_dataset("MRI_tensor", data=dataset_tensor)   
    
if __name__ == '__main__':
    main()