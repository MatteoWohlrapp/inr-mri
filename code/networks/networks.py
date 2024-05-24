import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models
from torchvision.models import resnet18, ResNet18_Weights


def cast_tuple(val, repeat=1):
    return val if isinstance(val, tuple) else ((val,) * repeat)


# sin activation
class Sine(nn.Module):
    def __init__(self, w0=1.0):
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
        w0=1.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
        dropout=0.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)

    def init_(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        out = self.dropout(out)
        return out


# siren network
class SirenNet(nn.Module):
    def __init__(
        self, dim_in, dim_hidden, dim_out, num_layers, w0, w0_initial, use_bias, dropout
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
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                use_bias=use_bias,
                is_first=is_first,
                dropout=dropout,
            )

            self.layers.append(layer)

        self.last_layer = Siren(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, use_bias=use_bias
        )

    def forward(self, x, mods=None):
        mods = cast_tuple(mods, self.num_layers)

        for layer, mod in zip(self.layers, mods):
            x = layer(x)

            if mod is not None:
                x *= rearrange(mod, "b d -> b () d")

        return self.last_layer(x)


# modulatory feed forward network
class Modulator(nn.Module):
    def __init__(self, dim_in, dim_hidden, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([])

        for ind in range(num_layers):
            is_first = ind == 0
            dim = dim_in if is_first else (dim_hidden + dim_in)

            self.layers.append(nn.Sequential(nn.Linear(dim, dim_hidden), nn.ReLU()))

    def forward(self, z):
        x = z
        hiddens = []

        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z), dim=1)

        return tuple(hiddens)


# encoder
class Encoder(nn.Module):
    def __init__(self, feature_extract=True, use_pretrained=True, latent_dim=256):
        super(Encoder, self).__init__()
        self.resnet, num_features = self.load_pretrained_resnet(
            feature_extract, use_pretrained
        )

        # Add a fully connected layer to map to the desired latent vector size
        self.fc = nn.Linear(num_features, latent_dim)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    def load_pretrained_resnet(self, feature_extract=True, use_pretrained=True):
        # Load a pretrained ResNet model
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # Remove the original fully connected layer, the output will be the features from the penultimate layer
        num_features = model.fc.in_features
        model.fc = nn.Identity()  # Remove the final fully connected layer

        return model, num_features

    def forward(self, x):
        # Use the ResNet for feature extraction
        x = x.unsqueeze(1)
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
        w0,
        w0_initial,
        use_bias,
        dropout,
        modulate,
    ):
        super().__init__()

        self.image_width = image_width
        self.image_height = image_height
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.modulate = modulate

        self.net = SirenNet(
            dim_in=dim_in,
            dim_hidden=dim_hidden,
            dim_out=dim_out,
            num_layers=num_layers,
            w0=w0,
            w0_initial=w0_initial,
            use_bias=use_bias,
            dropout=dropout,
        )

        self.modulator = Modulator(
            dim_in=latent_dim, dim_hidden=dim_hidden, num_layers=num_layers
        )

        self.encoder = Encoder(latent_dim=latent_dim)

        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        mgrid = rearrange(mgrid, "h w b -> (h w) b")
        self.register_buffer("grid", mgrid)

    def forward(self, img=None):
        batch_size = img.shape[0] if img is not None else 1

        mods = (
            self.modulator(self.encoder(img))
            if self.modulate and img is not None
            else None
        )

        coords = self.grid.clone().detach().repeat(batch_size, 1, 1).requires_grad_()

        out = self.net(coords, mods)
        out = rearrange(
            out, "b (h w) c -> () b c h w", h=self.image_height, w=self.image_width
        )
        out = out.squeeze(0).squeeze(1)
        return out

    def upscale(self, scale_factor, img=None):
        mods = (
            self.modulator(self.encoder(img))
            if self.modulate and img is not None
            else None
        )

        tensors = [
            torch.linspace(-1, 1, steps=self.image_height * scale_factor),
            torch.linspace(-1, 1, steps=self.image_width * scale_factor),
        ]
        mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
        coords = rearrange(mgrid, "h w b -> (h w) b")

        out = self.net(coords, mods)
        out = rearrange(
            out,
            "(h w) c -> () c h w",
            h=self.image_height * scale_factor,
            w=self.image_width * scale_factor,
        )
        out = out.squeeze(0)
        return out
