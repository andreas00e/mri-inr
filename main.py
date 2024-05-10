import torch
from data.dataset import MRIDataset
from networks.networks import ModulatedSiren
from trainer.trainer import Trainer

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    train_dataset = MRIDataset(
        path='../../dataset/brain/singlecoil_train',
    )

    # Initialize the model
    model = ModulatedSiren(
        image_width=128,  # Adjust based on actual image dimensions
        image_height=128,  # Adjust based on actual image dimensions
        dim_in=2,
        dim_hidden=256,
        dim_out=1, 
        num_layers=5,
        latent_dim=256,
        dropout=0.1,
    )

    # Create trainer instance
    trainer = Trainer(model=model, device=device, train_dataset=train_dataset)

    # Start training
    trainer.train(num_epochs=2)

if __name__ == '__main__':
    main()
