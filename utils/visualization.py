import matplotlib.pyplot as plt
from networks.networks import SirenNet, ModulatedSiren
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_batch(images, num_images=4, cmap='gray'):
    """
    Display a batch of images using matplotlib.

    Args:
        images (torch.Tensor): A batch of images as a 3D tensor (batch_size x height x width).
        num_images (int): Number of images to display from the batch. Default is 4.
        cmap (str): The colormap to use for displaying the images. Default is 'gray'.
    """
    # Ensure the number of images doesn't exceed the batch size
    num_images = min(num_images, images.shape[0])
    
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    if num_images == 1:
        axes = [axes]  # Make it iterable
    
    for i, ax in enumerate(axes):
        # Assuming image data is [height, width], if it has channels adjust indexing accordingly
        ax.imshow(images[i].squeeze(), cmap=cmap)  # Use squeeze() to handle single-channel (grayscale) images
        ax.axis('off')  # Turn off axis labels
    
    plt.show()


def show_image(image, cmap='gray'):
    """
    Display a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        cmap (str): The colormap to use for displaying the image. Default is 'gray'.
    """
    plt.imshow(image.squeeze(), cmap=cmap)
    plt.axis('off')
    plt.show()

def upscale_from_siren(model_path = 'model.pth', upscale_factor=4, file_name='output'): 

    model = ModulatedSiren(
            image_width=320,  # Adjust based on actual image dimensions
            image_height=640,  # Adjust based on actual image dimensions
            dim_in=1,
            dim_hidden=256,
            dim_out=1, 
            num_layers=5,
            latent_dim=256,
            dropout=0.1,
    )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    with torch.no_grad():
        model.eval()

        img = model.upscale(upscale_factor)
        original_image = model()

        if img.is_cuda:
            img = img.cpu()
            original_image = original_image.cpu()

        # display greyscale image
        plt.imshow(img.squeeze(),cmap='gray')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.savefig(f"{file_name}_upscaled.png", bbox_inches='tight', pad_inches=0, dpi=1200)
        plt.close()

        plt.imshow(original_image.squeeze(),cmap='gray')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.savefig(f"{file_name}_original.png", bbox_inches='tight', pad_inches=0, dpi=1200)
        plt.close()
