from utils.visualization import reconstruct_from_model
from data.transformations import scale_mri_tensor_advanced
import torch
from data.mri_sampler import MRIRandomSampler
from networks.networks import ModulatedSiren
from torchvision import transforms
import os

def save_args_to_file(args, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    args_path = os.path.join(output_dir, "config.txt")
    
    with open(args_path, 'w') as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

def test(args):

    output_dir = f"../output/results/{args.name}"
    save_args_to_file(args, output_dir)

    # Setup transformations
    transformations = []
    for transformation in args.transformations:
        if transformation == "normalize":
            transformations.append(scale_mri_tensor_advanced)

    # Load the dataset
    sampler = MRIRandomSampler(
        path=args.test_dataset,
        filter_func=(lambda x: args.mri_type in x),
        transform=transforms.Compose(transformations),
        test_files=args.test_files,
    )

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

    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    with torch.no_grad():
        model.eval()

        for i in range(args.num_samples):
            fully_sampled_img, undersampled_img, filename = sampler.get_random_sample()
            fully_sampled_img = fully_sampled_img.squeeze().unsqueeze(0)
            undersampled_img = undersampled_img.squeeze().unsqueeze(0)

            reconstruct_from_model(
                model=model,
                model_path=args.model_path,
                output_dir=output_dir,
                filename=f"{filename}_fully_sampled",
                img=fully_sampled_img,
            )

            reconstruct_from_model(
                model=model,
                model_path=args.model_path,
                output_dir=output_dir,
                filename=f"{filename}_undersampled",
                img=undersampled_img,
            )
