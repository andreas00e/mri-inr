import torch
from data.dataset import MRIDataset
from overfit.siren_test import SirenNet, SirenWrapper
from overfit.trainer_test import Trainer
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from overfit.dataset_test import TestDataset

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0)),  # Ensure there is a channel dimension
        transforms.Normalize(mean=[0.5], std=[0.5]), 
        transforms.Lambda(lambda x: x.squeeze(0)),    
        ])

    # Load dataset
    train_dataset = TestDataset()

    net = SirenNet(
        dim_in = 2,                        # input dimension, ex. 2d coor
        dim_hidden = 1024,                  # hidden dimension
        dim_out = 3,                       # output dimension, ex. rgb value
        num_layers = 5,                    # number of layers
        w0_initial = 30.,                   # different signals may require different omega_0 in the first layer - this is a hyperparameter, 
        dropout=0.1
    )

    wrapper = SirenWrapper(
        net,
        image_width = train_dataset[0].shape[1],
        image_height = train_dataset[0].shape[0]
    )

    # Create trainer instance
    trainer = Trainer(model=wrapper, device=device, train_dataset=train_dataset, val_dataset=train_dataset, batch_size=1)

    # Start training
    trainer.train(num_epochs=300)

if __name__ == '__main__':
    main()
