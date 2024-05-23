from utils.visualization import retrieve_from_siren
from data.transformations import scale_mri_tensor_advanced
from torchvision import transforms
from data.dataset import MRIDataset

def visualize(args):
    if args.visualize == 'siren':
            retrieve_from_siren(model_path="../outputmodel_checkpoints/siren_model.pth", file_name="siren")
    elif args.visualize == 'modulated':
        transform = transforms.Compose([
            scale_mri_tensor_advanced
        ])
        train_dataset = MRIDataset(
        path='../../../dataset/fastmri/brain/visualize', filter_func=(lambda x: 'FLAIR' in x), transform=transform, number_of_samples = 10
        )

        for i in range(10):
            img = train_dataset[i]
            img = img.squeeze().unsqueeze(0)
            retrieve_from_siren(model_path="../output/model_checkpoints/modulated_siren_model.pth", file_name=f"./output/modulated_{i}", img = img)