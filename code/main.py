from utils.argparser import parse_cmd_args
from modes.train import train
from modes.test import test
from modes.train_encoder import train_encoder
from data.data_transform import process_files
from networks.encoder.encoder import config
from networks.encoder.util import plot_encoder_output
from data.mri_dataset import MRIDataset, MRIDatasetTransformed
import pathlib
import torch
from networks.encoder.parser import get_args
from modes.train_encoder import train_encoder
from networks.encoder.encoder import build_autoencoder, load_model


def main():

    args = parse_cmd_args()

    if args.mode == "test":
        test(args)
    else:
        train(args)


def main_encoder():
    args = get_args()
    print(args)
    #train_encoder(args)

def plot_examples():
    n_plots = 100
    path_train = pathlib.Path(
    r"/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain/singlecoil_train_normalized"
    )
    path_val = pathlib.Path(
    r"/vol/aimspace/projects/practical_SoSe24/mri_inr/dataset/fastmri/brain/singlecoil_val_normalized"
    )
    dataset = MRIDatasetTransformed(path_val, number_of_samples = 200, shuffle = True)
    model_path = pathlib.Path(r'/vol/aimspace/projects/practical_SoSe24/mri_inr/rogalka/mri-inr/models/20240530-170738_autoencoder_v1_256.pth')
    autoencoder = load_model(model_path)
    dest_dir_folder = pathlib.Path(r'/vol/aimspace/projects/practical_SoSe24/mri_inr/rogalka/mri-inr/models/plots') / model_path.stem
    dest_dir_folder.mkdir()
    for i in range(n_plots):
        plot_encoder_output(autoencoder,dataset[i], dest_dir_folder / f'{pathlib.Path(dataset.samples[i][0]).stem}.png', str(pathlib.Path(dataset.samples[i][0]).stem))



# #SBATCH --partition=course
if __name__ == "__main__":
    print('Start')
    main()
    print(('End'))
