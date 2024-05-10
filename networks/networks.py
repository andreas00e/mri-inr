import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, repeat = 1):
    return val if isinstance(val, tuple) else ((val,) * repeat)

# sin activation
class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

# one siren layer
class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0 = 1.,
        c = 6.,
        is_first = False,
        use_bias = True,
        activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c = c, w0 = w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if exists(bias):
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out =  F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out

# siren network
class SirenNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0 = 1.,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0.
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden

        self.layers = nn.ModuleList([])
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layer = Siren(
                dim_in = layer_dim_in,
                dim_out = dim_hidden,
                w0 = layer_w0,
                use_bias = use_bias,
                is_first = is_first,
                dropout = dropout
            )

            self.layers.append(layer)

        final_activation = nn.Identity() if not exists(final_activation) else final_activation
        self.last_layer = Siren(dim_in = dim_hidden, dim_out = dim_out, w0 = w0, use_bias = use_bias, activation = final_activation)

    def forward(self, x, mods = None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if exists(mod):
                x *= rearrange(mod, 'd -> () d')

        return self.last_layer(x)

# modulatory feed forward network
class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(
                nn.Linear(dim, dim_hidden),
                nn.ReLU()
            ))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))

        return tuple(hiddens)

# encoder 
class Encoder(nn.Module):
    def __init__(self, feature_extract=True, use_pretrained=True, latent_dim=256):
        super(Encoder, self).__init__()
        self.resnet, num_features = self.load_pretrained_resnet(feature_extract, use_pretrained)
        
        # Add a fully connected layer to map to the desired latent vector size
        self.fc = nn.Linear(num_features, latent_dim)

    def load_pretrained_resnet(self, feature_extract=True, use_pretrained=True):
        # Load a pretrained ResNet model
        model = models.resnet18(pretrained=use_pretrained)

        # If we are using the model just for feature extraction, we freeze all parameters
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # Remove the original fully connected layer, the output will be the features from the penultimate layer
        num_features = model.fc.in_features
        model.fc = nn.Identity()  # Remove the final fully connected layer

        return model, num_features
    
    def forward(self, x):
        # Use the ResNet for feature extraction
        features = self.resnet(x)
        # Map features to the latent vector space
        latent_vector = self.fc(features)
        return latent_vector


# complete network
class ModulatedSiren(nn.Module):
    def __init__(
        self,
        image_width,
        image_height, 
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        latent_dim,
        w0 = 1.,
        w0_initial = 30.,
        use_bias = True,
        final_activation = None,
        dropout = 0., 
    ):        
        super().__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.net = SirenNet(
            dim_in = 2,
            dim_hidden = dim_hidden,
            dim_out = dim_out,
            num_layers = num_layers,
            w0 = w0,
            w0_initial = w0_initial,
            use_bias = use_bias,
            final_activation = final_activation,
            dropout = dropout
        )

        self.modulator = Modulator(
            dim_in = latent_dim,
            dim_hidden = dim_hidden,
            num_layers = num_layers
        )

        tensors = [torch.linspace(-1, 1, steps = image_height), torch.linspace(-1, 1, steps = image_width)]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing = 'ij'), dim=-1)
        mgrid = rearrange(mgrid, 'h w c -> (h w) c')
        self.register_buffer('grid', mgrid)

    def forward(self, img, coordinate):
        
        latent = self.encoder(img) 
        mods = self.modulator(latent) 
        coords = self.grid.clone().detach().requires_grad_()
        out = self.net(coords, mods)
        return out