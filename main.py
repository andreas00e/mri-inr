import torch
from data.dataset import MRIDataset
from networks.networks import ModulatedSiren, SirenNet
from trainer.trainer import Trainer
from torchvision import transforms, datasets
import argparse
from utils.visualization import upscale_from_siren
from data.transformations import scale_mri_tensor_advanced

def main():
    parser = argparse.ArgumentParser(description="Train a SIREN network on MRI data")
    parser.add_argument('--visualize', type=str, choices=['siren', 'modulated'],
                        help='Choose the visualization mode: siren or modulated')

    args = parser.parse_args()

    if args.visualize:
        if args.visualize == 'siren':
            upscale_from_siren(model_path="model_checkpoints/siren_model.pth", upscale_factor=2, file_name="siren")
        elif args.visualize == 'modulated':
            print("Not implemented yet.")
    else: 
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            scale_mri_tensor_advanced
            ])

        # Load dataset
        train_dataset = MRIDataset(
            path='../../dataset/fastmri/brain/singlecoil_train', filter_func=(lambda x: 'FLAIR' in x), transform=transform, number_of_samples = 10
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
            modulate = True
        )

        # Create trainer instance
        trainer = Trainer(model=model, device=device, train_dataset=train_dataset, batch_size=10)

        # Start training
        trainer.train(num_epochs=15000)

if __name__ == '__main__':
    main()
