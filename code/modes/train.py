import torch
from data.mri_dataset import MRIDataset
from networks.networks import ModulatedSiren
from trainer.trainer import Trainer
from data.transformations import scale_mri_tensor_advanced
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torchvision import transforms


def train(args):
    print("Running the model...")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup transformations
    transformations = []
    for transformation in args.transformations:
        if transformation == "normalize":
            transformations.append(scale_mri_tensor_advanced)

    # Load dataset
    train_dataset = MRIDataset(
        path=args.train_dataset,
        filter_func=(lambda x: args.mri_type in x),
        transform=transforms.Compose(transformations),
        number_of_samples=args.num_samples,
    )

    val_dataset = (
        MRIDataset(
            path=args.val_dataset,
            filter_func=(lambda x: args.mri_type in x),
            transform=transformations,
            number_of_samples=args.num_samples,
        )
        if args.val_dataset
        else None
    )

    # Initialize the model
    model = ModulatedSiren(
        image_width=args.image_width,
        image_height=args.image_height,
        dim_in=args.dim_in,
        dim_hidden=args.dim_hidden,
        dim_out=args.dim_out,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        w0=args.w0,
        w0_initial=args.w0_initial,
        use_bias=args.use_bias,
        dropout=args.dropout,
        modulate=args.modulate,
    )

    # Select the optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    else:
        raise ValueError("Unsupported optimizer")

    # Optional: Initialize the learning rate scheduler
    scheduler = None
    if args.scheduler == "StepLR":
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)


    # Create trainer instance
    trainer = Trainer(
        model=model,
        device=device,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        output_name=args.name,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # Start training
    trainer.train(num_epochs=args.epochs)
