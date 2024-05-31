import argparse
import datetime
import yaml


def add_args2parser_execution(parser):
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Choose the mode of the program: train, visualize or test",
    )
    parser.add_argument(
        "--name", type=str, default="modulated_siren", help="Name of the current run"
    )


def add_args2parser_dataset(parser):
    parser.add_argument(
        "--train_dataset",
        type=str,
        default="../../../dataset/fastmri/brain/singlecoil_train",
        help="Path to the MRI dataset.",
    )
    parser.add_argument(
        "--val_dataset", type=str, default=None, help="Path to the MRI dataset."
    )
    parser.add_argument(
        "--transformations",
        action="append",
        default=[],
        help="List of transformations to apply to MRI data. This argument can be specified multiple times for multiple transformations.",
    )
    parser.add_argument(
        "--undersampled",
        action="store_true",
        default=True,
        help="Flag to indicate if the images are undersampled or not.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to load from the MRI dataset.",
    )
    parser.add_argument(
        "--mri_type",
        type=str,
        default="FLAIR",
        help="Type of MRI image we are considering",
    )
    return parser


def add_args2parser_siren(parser):
    parser.add_argument(
        "--dim_in", type=int, default=2, help="Input dimension for SIREN network"
    )
    parser.add_argument(
        "--dim_hidden", type=int, default=256, help="Hidden dimension for SIREN network"
    )
    parser.add_argument(
        "--dim_out", type=int, default=1, help="Output dimension for SIREN network"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=5,
        help="Number of layers in the SIREN network",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Dimension of the latent vector for modulated SIREN network",
    )
    parser.add_argument(
        "--w0", type=float, default=1.0, help="Omega_0 for SIREN network"
    )
    parser.add_argument(
        "--w0_initial",
        type=float,
        default=30.0,
        help="Initial Omega_0 for the first layer in SIREN network",
    )
    parser.add_argument(
        "--use_bias",
        action="store_true",
        default=True,
        help="Whether to use bias in SIREN network layers",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Dropout rate in SIREN network"
    )
    parser.add_argument(
        "--image_width",
        type=int,
        default=320,
        help="Image width for modulated SIREN output",
    )
    parser.add_argument(
        "--image_height",
        type=int,
        default=640,
        help="Image height for modulated SIREN output",
    )
    parser.add_argument(
        "--modulate",
        action="store_true",
        default=True,
        help="Flag to enable modulation in Modulated SIREN",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="resnet18",
        choices=["resnet18", "autoencoder"],
        help="The enoder to derive the latent code from the input image",
    )
    return parser


def add_args2parser_trainer(parser):
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for the trainer"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=15000, help="Number of epochs to train the model"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer for training: SGD, Adam, or AdamW",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=[None, "StepLR", "ExponentialLR"],
        help="Learning rate scheduler to adjust the learning rate during training",
    )
    parser.add_argument(
        "--step_size", type=int, default=100, help="Step size for the StepLR scheduler"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma (decay factor) for learning rate reduction",
    )
    parser.add_argument(
        "--limit_io",
        default=False,
        help="Decide if we print after each batch or epoch",
    )
    return parser


def add_args2parser_test(parser):
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the model to be used for testing",
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        default="../../../dataset/fastmri/brain/singlecoil_val",
        help="Path to the MRI dataset for test.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for reproducibility during testing"
    )
    parser.add_argument(
        "--test_files",
        action="append",
        default=None,
        help="List of test files to apply the model on. This argument can be specified multiple times for multiple files. If not specified, random samples will be taken from the test dataset.",
    )
    return parser


def mk_parser_main():
    parser = argparse.ArgumentParser(description="Your Project Description Here")

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="load YAML configuration",
        dest="config_file",
        type=argparse.FileType(mode="r"),
    )

    add_args2parser_execution(parser)
    add_args2parser_dataset(parser)
    add_args2parser_siren(parser)
    add_args2parser_trainer(parser)

    return parser

def parse_cmd_args():
    parser = mk_parser_main()
    args = parser.parse_args()

    if args.config_file:
        config_data = yaml.safe_load(args.config_file)
        for key, value in config_data.items():
            setattr(args, key, value)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    args.name = f"{current_time}_{args.name}"

    # print complete configuration
    print("Configuration:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    return args
