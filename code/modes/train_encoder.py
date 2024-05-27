import torch
from data.mri_dataset import MRIDataset, MRIDatasetTransformed
from networks.networks import ModulatedSiren
from trainer.trainer import Trainer
from networks.encoder.encoder import get_encoder, test_encoder
import pathlib

path_train = pathlib.Path(
    r"/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain_norm/singlecoil_train"
)
path_val = pathlib.Path(
    r"/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain_norm/singlecoil_val"
)


def train_encoder(args):
    print("Training the encoder...")

    # Load dataset
    train_dataset = MRIDatasetTransformed(path_train)
    val_dataset = MRIDatasetTransformed(path_val)

    # Get encoder
    encoder = get_encoder(train_dataset, val_dataset, num_epochs=10)
