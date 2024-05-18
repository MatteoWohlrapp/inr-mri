import torch
from data.dataset import MRIDataset
from networks.networks import ModulatedSiren, SirenNet
from trainer.trainer import Trainer
from torchvision import transforms, datasets
import argparse
from utils.visualization import upscale_from_siren

def main():
    parser = argparse.ArgumentParser(description="Train a SIREN network on MRI data")
    parser.add_argument('--visualize', type=str, choices=['siren', 'modulated'],
                        help='Choose the visualization mode: siren or modulated')

    args = parser.parse_args()

    if args.visualize:
        if args.visualize == 'siren':
            upscale_from_siren(model_path="model_checkpoints/siren_model.pth", upscale_factor=2, file_name="siren_upscaled_image.png")
        elif args.visualize == 'modulated':
            print("Not implemented yet.")
    else: 
        # Setup device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([
            transforms.Lambda(lambda x: x.unsqueeze(0)),  # Ensure there is a channel dimension
            transforms.Normalize(mean=[0.5], std=[0.5]), 
            transforms.Lambda(lambda x: x.squeeze(0)),    
            ])

        # Load dataset
        train_dataset = MRIDataset(
            path='../../dataset/brain/singlecoil_train', filter_func=(lambda x: 'FLAIR' in x), transform=transform
        )

        val_dataset = MRIDataset(
            path='../../dataset/brain/singlecoil_val', filter_func=(lambda x: 'FLAIR' in x)
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
        )

        # Create trainer instance
        trainer = Trainer(model=model, device=device, train_dataset=train_dataset, val_dataset=val_dataset, batch_size=1)

        # Start training
        trainer.train(num_epochs=1)

if __name__ == '__main__':
    main()
