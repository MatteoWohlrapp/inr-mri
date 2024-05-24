import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import normalized_root_mse as nrmse
from utils.visualization import save_image
import os


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


def inference_error(model, model_path, output_dir, filename, img):

    reconstructed_image = model(img)

    if reconstructed_image.is_cuda:
        reconstructed_image = reconstructed_image.cpu()

    save_image(reconstructed_image, f"{filename}_reconstructed", output_dir)
    save_image(img, f"{filename}_gt", output_dir)
    save_image(
        calculate_difference(
            img.squeeze().numpy(), reconstructed_image.squeeze().numpy()
        ),
        f"{filename}_difference",
        output_dir,
        cmap="viridis",
    )

    # Calculate the error metrics
    psnr_value = calculate_psnr(
        img.squeeze().numpy(), reconstructed_image.squeeze().numpy()
    )
    ssim_value = calculate_ssim(
        img.squeeze().numpy(), reconstructed_image.squeeze().numpy()
    )
    nrmse_value = calculate_nrmse(
        img.squeeze().numpy(), reconstructed_image.squeeze().numpy()
    )

    # Write them to a file
    with open(os.path.join(output_dir, f"{filename}_error.txt"), "w") as f:
        f.write(f"PSNR: {psnr_value}\n")
        f.write(f"SSIM: {ssim_value}\n")
        f.write(f"NRMSE: {nrmse_value}\n")
