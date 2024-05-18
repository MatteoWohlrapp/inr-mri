import torch
from data.dataset import MRIDataset
from overfit.siren_test import SirenNet, SirenWrapper
from overfit.trainer_test import Trainer
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from overfit.dataset_test import TestDataset
from data.transformations import scale_mri_tensor_advanced

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        scale_mri_tensor_advanced   
        ])
    
    # Load dataset
    train_dataset = TestDataset(
        path='../../dataset/brain/singlecoil_train', filter_func=(lambda x: 'FLAIR' in x), transform=transform, cat=False
    )

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

    # Create trainer instance
    trainer = Trainer(model=wrapper, device=device, train_dataset=train_dataset, val_dataset=train_dataset, batch_size=1)

    # Start training
    trainer.train(num_epochs=15000)

if __name__ == '__main__':
    main()
