import torch


def scale_mri_tensor_advanced(img_tensor):
    mean_val = img_tensor.mean()
    std_dev = img_tensor.std()

    # Scale based on standard deviation, e.g., mean ± 3*std_dev maps to [-1, 1]
    three_std_from_mean = 3 * std_dev
    img_tensor = (img_tensor - mean_val) / three_std_from_mean

    # Clipping the values to be within [-1, 1]
    img_tensor = torch.clamp(img_tensor, -1, 1)

    return img_tensor
