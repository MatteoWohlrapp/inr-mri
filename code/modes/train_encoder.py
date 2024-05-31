import torch
from data.mri_dataset import MRIDataset, MRIDatasetTransformed
from networks.networks import ModulatedSiren
from trainer.trainer import Trainer
from networks.encoder.encoder import get_encoder, test_encoder
import pathlib

path_train = pathlib.Path(
    r"/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain/singlecoil_train_normalized"
)
path_val = pathlib.Path(
    r"/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain/singlecoil_val_normalized"
)


def train_encoder(args):
    print("Training the encoder...")

    # Load dataset
    train_dataset = MRIDatasetTransformed(path_train, number_of_samples = 20000)
    val_dataset = MRIDatasetTransformed(path_val, number_of_samples = 500)
    print(f'Size of train dataset: {len(train_dataset)}')
    print(f'Size of val dataset: {len(val_dataset)}')

    # Get encoder
    encoder = get_encoder(train_dataset, val_dataset, num_epochs=200, batch_size = 200)
