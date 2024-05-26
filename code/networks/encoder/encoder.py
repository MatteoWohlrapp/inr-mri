"""Reimplementation of the autoencoder """
import torch
import torch.nn as nn
import yaml
import json
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
import pathlib
import tqdm
from networks.encoder.util import extract_patches, reconstruct_image

window_size = 32
latent_dim = 256

config = {
    'id': f'autoencoder_v1_{latent_dim}',
    'encoder': [
        {'type': 'Conv2d', 'params': {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1}},
        {'type': 'LeakyReLU', 'params': {'negative_slope': 0.2}},
        {'type': 'Conv2d', 'params': {'in_channels': 16, 'out_channels': 32, 'kernel_size': 3, 'stride': 2, 'padding': 1}},
        {'type': 'LeakyReLU', 'params': {'negative_slope': 0.2}},
        {'type': 'Conv2d', 'params': {'in_channels': 32, 'out_channels': 64, 'kernel_size': 8}},
        {'type': 'LeakyReLU', 'params': {'negative_slope': 0.2}},
        {'type': 'Flatten'},
        {'type': 'Linear', 'params': {'in_features': 64, 'out_features': latent_dim}},
    ],
    'decoder': [
        {'type': 'Linear', 'params': {'in_features': latent_dim, 'out_features': 64}},
        {'type': 'LeakyReLU', 'params': {'negative_slope': 0.2}},
        {'type': 'Unflatten'},
        {'type': 'ConvTranspose2d', 'params': {'in_channels': 64, 'out_channels': 32, 'kernel_size': 8}},
        {'type': 'LeakyReLU', 'params': {'negative_slope': 0.2}},
        {'type': 'ConvTranspose2d', 'params': {'in_channels': 32, 'out_channels': 16, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}},
        {'type': 'LeakyReLU', 'params': {'negative_slope': 0.2}},
        {'type': 'ConvTranspose2d', 'params': {'in_channels': 16, 'out_channels': 1, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}},
        {'type': 'Sigmoid'}
    ]
}


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class Unflatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 64, 1, 1)
    
class Inspector(nn.Module):
    def forward(self, input):
        print(f'Inspector input shape: {input.shape}')
        return input

class Autoencoder(nn.Module):

    def __init__(self, encoder, decoder, id):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id = id
        self.latent_dim = latent_dim

    def forward(self, x):
        #loop over batch
        batch_size = x.shape[0]
        height = x.shape[1]
        width = x.shape[2]
        x = extract_patches(x, window_size)
        x = x.view(-1, 1, window_size, window_size).cuda()
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(batch_size, -1, window_size, window_size)
        x = reconstruct_image(x, height, width, window_size)
        return x
    
    def encode(self, x):
        x = extract_patches(x, window_size)
        x = x.view(-1, 1, window_size, window_size).cuda()
        x = self.encoder(x)
        return x

class AutoencoderBuilder:
    def __init__(self, config):
        self.config = config

    def build_network(self):
        encoder_layers = self.build_layers(self.config['encoder'])
        decoder_layers = self.build_layers(self.config['decoder'])
        return Autoencoder(nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers), self.config['id'])

    def build_layers(self, layer_configs):
        layers = []
        for layer in layer_configs:
            layer_type = layer['type']
            if layer_type == 'Conv2d':
                layers.append(nn.Conv2d(**layer['params']))
            elif layer_type == 'ConvTranspose2d':
                layers.append(nn.ConvTranspose2d(**layer['params']))
            elif layer_type == 'LeakyReLU':
                layers.append(nn.LeakyReLU(**layer['params']))
            elif layer_type == 'Linear':
                layers.append(nn.Linear(**layer['params']))
            elif layer_type == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif layer_type == 'Flatten':
                layers.append(Flatten())
            elif layer_type == 'Unflatten':
                layers.append(Unflatten())
            elif layer_type == 'Inspector':
                layers.append(Inspector())
            else:
                raise ValueError(f"Unsupported layer type: {layer_type}")
        return layers

def load_config(file_path, file_type='yaml'):
    if file_type == 'yaml':
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    elif file_type == 'json':
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

class Trainer:
    def __init__(self, model: nn.Module, criterion, optimizer, device, train_dataset, val_dataset, batch_size):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.name = f'{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}_{self.model.id}'
        self.writer = SummaryWriter(log_dir=f'runs/tensorboard/{self.name}')
    
    def save_model(self):
        pathlib.Path(r'/vol/aimspace/projects/practical_SoSe24/mri_inr/runs/models/').mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, pathlib.Path(r'/vol/aimspace/projects/practical_SoSe24/mri_inr/runs/models/' + self.name + '.pth'))

    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def train(self, num_epochs):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True)
        pbar = tqdm.tqdm(range(num_epochs))
        for epoch in pbar:
            self.model.train()
            for i, batch in enumerate(train_loader):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch)
                loss = self.criterion(output, batch)
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + i)
                pbar.set_description(f'Epoch {epoch}, Loss: {loss.item()}')
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_loader):
                    batch = batch.to(self.device)
                    output = self.model(batch)
                    loss = self.criterion(output, batch)
                    self.writer.add_scalar('validation_loss', loss.item(), epoch * len(val_loader) + i)
        self.save_model()
        self.writer.close()

def get_encoder(train_dataset, val_dataset, config = config, num_epochs=2):
    # train the autoencoder
    autoencoder_builder = AutoencoderBuilder(config)
    model = autoencoder_builder.build_network()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model=model, criterion=torch.nn.MSELoss(), optimizer=torch.optim.Adam(model.parameters(), lr=1e-4), device=device, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=1)
    trainer.train(num_epochs)
    #return the model in evaluation mode
    model.eval()
    return model

def test_encoder(encoder, image):
    # plot the original image and the reconstructed image side by side
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')
    image = torch.tensor(image).float().unsqueeze(0).unsqueeze(0)
    image = image.to(torch.device('cuda'))
    image = image.squeeze(0)
    output = encoder(image)
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    axs[1].imshow(output, cmap='gray')
    axs[1].set_title('Reconstructed Image')
    plt.show()
    return


