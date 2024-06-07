from data.transformations import scale_mri_tensor_advanced
import torch
from data.mri_sampler import MRIRandomSampler, MRIRandomSamplerTransformed
from networks.networks import ModulatedSiren, ModulatedSirenTiling
from torchvision import transforms
import os
from utils.error import inference_error
from utils.tiling import extract_with_inner_patches, reconstruct_image_from_inner_patches
import matplotlib.pyplot as plt


def save_args_to_file(args, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    config_path = os.path.join(output_dir, "config.txt")

    with open(config_path, "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def test(args):
    print("Testing the model...")

    output_dir = f"../output/results/{args.name}"
    save_args_to_file(args, output_dir)

    # Setup transformations
    transformations = []
    for transformation in args.transformations:
        if transformation == "normalize":
            transformations.append(scale_mri_tensor_advanced)

    # Load the dataset
    sampler = MRIRandomSamplerTransformed(
        path=args.test_dataset,
        filter_func=(lambda x: args.mri_type in x),
        transform=transforms.Compose(transformations),
        test_files=args.test_files,
        target_height = args.image_height, 
        target_width = args.image_width
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
        encoder_type=args.encoder_type,
    )

    model.load_state_dict(torch.load(args.model_path, map_location=torch.device("cpu")))

    with torch.no_grad():
        model.eval()

        for i in range(args.num_samples):
            print(f"Processing sample {i + 1}/{args.num_samples}...")
            fully_sampled_img, undersampled_img, filename = sampler.get_random_sample()

             # unsqueeze image to add batch dimension
            fully_sampled_img = fully_sampled_img.unsqueeze(0)
            undersampled_img = undersampled_img.unsqueeze(0)

            fully_sampled_patch, fully_sampled_inormation = extract_with_inner_patches(fully_sampled_img, 32, 16)
            undersampled_patch, undersampled_information = extract_with_inner_patches(undersampled_img, 32, 16)

            output_dir_temp = os.path.join(output_dir, filename)
            if not os.path.exists(output_dir_temp):
                os.makedirs(output_dir_temp)

            inference_error(
                model=model,
                model_path=args.model_path,
                output_dir=output_dir_temp,
                filename=f"{filename}_fully_sampled",
                img=fully_sampled_patch,
                img_information=fully_sampled_inormation
            )

            inference_error(
                model=model,
                model_path=args.model_path,
                output_dir=output_dir_temp,
                filename=f"{filename}_undersampled",
                img=undersampled_patch,
                img_information=undersampled_information
            )
