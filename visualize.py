from overfit.siren_test import SirenNet, SirenWrapper
import torch
import matplotlib.pyplot as plt


def main(): 

    net = SirenNet(
        dim_in = 2,                        # input dimension, ex. 2d coor
        dim_hidden = 256,                  # hidden dimension
        dim_out = 1,                       # output dimension, ex. rgb value
        num_layers = 5,                    # number of layers
        w0_initial = 30.,                   # different signals may require different omega_0 in the first layer - this is a hyperparameter, 
        dropout=0
    )

    wrapper = SirenWrapper(
        net,
        image_width = 320,
        image_height = 640
    )

    wrapper.load_state_dict(torch.load('model.pth'))

    with torch.no_grad():
        wrapper.eval()

        img = wrapper.upscale(2)
        if img.is_cuda:
            img = img.cpu()

        # display greyscale image
        print(img.shape)
        plt.imshow(img.squeeze(),cmap='gray')
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.savefig('upscaled_out.png', bbox_inches='tight', pad_inches=0)
        plt.close()


if __name__ == '__main__':
    main()