import torch
from data.mri_dataset import MRIDatasetTransformed
from networks.encoder.encoder import Trainer, build_autoencoder, config, load_model, save_model
from networks.encoder.parser import get_args
import pathlib

def train_encoder(args):
    print("Training the encoder...")
    print(args)
    # Load dataset
    train_dataset = MRIDatasetTransformed(pathlib.Path(args.path_train_dataset), number_of_samples = args.num_samples_train)
    val_dataset = MRIDatasetTransformed(pathlib.Path(args.path_val_dataset), number_of_samples = args.num_samples_val)

    # Set the device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the model
    if args.model_path == "":
        autoencoder = build_autoencoder(config)
    else:
        autoencoder = load_model(pathlib.Path(args.model_path))

    # Define the criterion and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

    # Define the trainer
    trainer = Trainer(
        autoencoder,
        criterion,
        optimizer,
        device,
        train_dataset,
        val_dataset,
        args.batch_size,
    )

    # Train the model
    trainer.train(args.epochs)

    # Save the model TODO change path if not given
    save_model(autoencoder, pathlib.Path(args.model_path))

