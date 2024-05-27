import torch
import torch.nn.functional as F


def extract_patches(tensor, patch_size):
    """
    Extract non-overlapping patches from a batch of grayscale images.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).
        patch_size (int): Size of each patch (patch_size x patch_size).

    Returns:
        torch.Tensor: Extracted patches of shape (batch_size, num_patches, patch_size, patch_size).
    """
    tensor = tensor.unsqueeze(1)
    batch_size, channels, height, width = tensor.shape
    # Ensure that the tensor is a 4D tensor with only one channel (grayscale)
    assert channels == 1, "Input tensor must have only one channel (grayscale images)."
    # Extract patches using F.unfold
    patches = F.unfold(tensor, kernel_size=(patch_size, patch_size), stride=patch_size)
    # Reshape to (batch_size, num_patches, patch_size, patch_size)
    patches = (
        patches.transpose(1, 2)
        .contiguous()
        .view(batch_size, -1, patch_size, patch_size)
    )

    return patches


def reconstruct_image(patches, image_height, image_width, patch_size):
    """
    Reconstruct the original images from the patches.

    Args:
        patches (torch.Tensor): Patches tensor of shape (batch_size, num_patches, patch_size, patch_size).
        image_height (int): The original image height.
        image_width (int): The original image width.
        patch_size (int): Size of each patch (patch_size x patch_size).

    Returns:
        torch.Tensor: Reconstructed images of shape (batch_size, 1, image_height, image_width).
    """
    batch_size, num_patches, patch_h, patch_w = patches.shape
    # Ensure the patch dimensions are correct
    assert (
        patch_h == patch_size and patch_w == patch_size
    ), "Patch size does not match expected dimensions."
    # Calculate the number of patches per row and column
    patches_per_row = image_width // patch_size
    patches_per_col = image_height // patch_size
    # Reshape the patches to (batch_size, patch_size*patch_size, num_patches)
    patches = patches.view(batch_size, num_patches, -1).transpose(1, 2)
    # Use fold to reconstruct the image
    recon_images = F.fold(
        patches,
        output_size=(image_height, image_width),
        kernel_size=(patch_size, patch_size),
        stride=patch_size,
    )
    # remove the extra dimension
    recon_images = recon_images.squeeze(1)
    return recon_images
