import torch
from data.dataset import MRIDataset
from networks.networks import ModulatedSiren, SirenNet
from trainer.trainer import Trainer
from torchvision import transforms, datasets

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.unsqueeze(0)),  # Ensure there is a channel dimension
        transforms.Normalize(mean=[0.5], std=[0.5]), 
        transforms.Lambda(lambda x: x.squeeze(0)),    
        ])

    # Load dataset
    train_dataset = MRIDataset(
        path='../../dataset/fastmri/brain/singlecoil_train', filter_func=(lambda x: 'FLAIR' in x), transform=transform, number_of_samples = 1
    )

    # Initialize the model
    model = ModulatedSiren(
        image_width=320,  # Adjust based on actual image dimensions
        image_height=640,  # Adjust based on actual image dimensions
        dim_in=1,
        dim_hidden=256,
        dim_out=1, 
        num_layers=5,
        latent_dim=256,
        dropout=0.1,
        modulate = False
    )

    # Create trainer instance
    trainer = Trainer(model=model, device=device, train_dataset=train_dataset, batch_size=1)

    # Start training
    trainer.train(num_epochs=15000)

if __name__ == '__main__':
    main()
