import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from utils.visualization import save_image
import os
from utils.tiling import extract_with_inner_patches, reconstruct_image_from_inner_patches
from utils.overlapping_tiling import tiles


def calculate_data_range(original, predicted):
    data_min = min(np.min(original), np.min(predicted))
    data_max = max(np.max(original), np.max(predicted))
    data_range = data_max - data_min

    return data_range


def calculate_psnr(original, predicted):
    """
    Calculate the PSNR between two images.
    """
    return psnr(
        original, predicted, data_range=calculate_data_range(original, predicted)
    )


def calculate_ssim(original, predicted):
    """
    Calculate the SSIM between two images.
    """
    return ssim(
        original, predicted, data_range=calculate_data_range(original, predicted)
    )


def calculate_nrmse(original, predicted):
    """
    Calculate the NRMSE between two images.
    """
    return nrmse(original, predicted)


def calculate_difference(original, predicted):
    """
    Create and show an image visualizing the difference between the original and predicted images.
    """
    # Compute the absolute difference image
    difference = np.abs(original - predicted)

    return difference


def inference_error(model, model_path, output_dir, filename, img, img_information):

    reconstructed_patches = model(img)

    if reconstructed_patches.is_cuda:
        reconstructed_patches = reconstructed_patches.cpu()

    reconstructed_image = reconstruct_image_from_inner_patches(reconstructed_patches, img_information, 16)
    original_image = reconstruct_image_from_inner_patches(img, img_information, 16)

    save_image(reconstructed_image, f"{filename}_reconstructed", output_dir)
    save_image(original_image, f"{filename}_gt", output_dir)
    save_image(
        calculate_difference(
            original_image.squeeze().numpy(), reconstructed_image.squeeze().numpy()
        ),
        f"{filename}_difference",
        output_dir,
        cmap="viridis",
    )

    # Calculate the error metrics
    psnr_value = calculate_psnr(
        original_image.squeeze().numpy(), reconstructed_image.squeeze().numpy()
    )
    ssim_value = calculate_ssim(
        original_image.squeeze().numpy(), reconstructed_image.squeeze().numpy()
    )
    nrmse_value = calculate_nrmse(
        original_image.squeeze().numpy(), reconstructed_image.squeeze().numpy()
    )

    # Write them to a file
    with open(os.path.join(output_dir, f"{filename}_error.txt"), "w") as f:
        f.write(f"PSNR: {psnr_value}\n")
        f.write(f"SSIM: {ssim_value}\n")
        f.write(f"NRMSE: {nrmse_value}\n")
        
def inference_error_overlapping(model, model_path, output_dir, filename, img):
    tiles = tiles()
    patches = tiles.create_tiles(img)
    patches_shape = list(patches.shape)

    patches = patches.reshape((patches_shape[0]*patches_shape[1]*patches_shape[2], 
                                                           patches_shape[3], patches_shape[4])) # -> [patches, k_h, k_w]
    patches_model = model(patches)

    if patches_model.is_cuda:
        patches_model = patches_model.cpu()
    
    patches_model = patches_model.reshape((patches_shape[0], patches_shape[1], patches_shape[2], 
                                           patches_shape[3], patches_shape[4])) # -> [batch, #h, #w, k_h, k_w]
    reconstructed_image, counter = tiles.recreate_image(patches_model)

    save_image(reconstructed_image, f"{filename}_reconstructed", output_dir)
    save_image(reconstructed_image/counter, f"{filename}_reconstructed_normalized", output_dir)
    save_image(img.squeeze(), f"{filename}_gt", output_dir)
    save_image(
        calculate_difference(
            img.squeeze().numpy(), reconstructed_image.numpy()
        ),
        f"{filename}_difference",
        output_dir,
        cmap="viridis",
    )

    # Calculate the error metrics
    psnr_value = calculate_psnr(
        img.squeeze().numpy(), reconstructed_image.numpy()
    )
    ssim_value = calculate_ssim(
        img.squeeze().numpy(), reconstructed_image.numpy()
    )
    nrmse_value = calculate_nrmse(
        img.squeeze().numpy(), reconstructed_image.numpy()
    )

    # Write them to a file
    with open(os.path.join(output_dir, f"{filename}_error.txt"), "w") as f:
        f.write(f"PSNR: {psnr_value}\n")
        f.write(f"SSIM: {ssim_value}\n")
        f.write(f"NRMSE: {nrmse_value}\n")
