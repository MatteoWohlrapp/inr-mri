
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def extract_patches_non_overlapping(tensor, patch_size):
    """
    Extract non-overlapping patches from a batch of grayscale images.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).
        patch_size (int): Size of each patch (patch_size x patch_size).

    Returns:
        torch.Tensor: Extracted patches of shape (batch_size, num_patches, patch_size, patch_size).
    """
    if len(tensor.shape) == 3:
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


def reconstruct_image_patches_non_overlapping(patches, image_height, image_width, patch_size):
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


def extract_with_inner_patches(tensor, outer_patch_size, inner_patch_size):
    """
    Extract overlapping patches from a batch of grayscale images, handling variable image sizes in a batch.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).
        outer_patch_size (int): Total size of each patch including the border.
        inner_patch_size (int): Size of the core area of each patch.
        
    Returns:
        Tuple[torch.Tensor, List[Tuple[int, int]]]: Tuple of extracted patches and list of tuples with number of patches (num_patches_vertical, num_patches_horizontal) for each image.
    """
    tensor = tensor.unsqueeze(1)
    batch_size, _, _, _ = tensor.shape

    stride = inner_patch_size
    padding = (outer_patch_size - inner_patch_size) // 2
    image_information = []

    all_patches = []

    for i in range(batch_size):
        height, width = tensor[i].shape[1:3]
        vertical_pad = (inner_patch_size - (height % inner_patch_size)) % inner_patch_size
        horizontal_pad = (inner_patch_size - (width % inner_patch_size)) % inner_patch_size

        padded_tensor = F.pad(tensor[i].unsqueeze(0), (padding, padding + horizontal_pad, padding, padding + vertical_pad), mode='reflect')
        
        patches = F.unfold(padded_tensor, kernel_size=(outer_patch_size, outer_patch_size), stride=stride)
        num_patches_vertical = (height + vertical_pad) // inner_patch_size
        num_patches_horizontal = (width + horizontal_pad) // inner_patch_size
        image_information.append((num_patches_vertical, num_patches_horizontal))
        
        patches = patches.transpose(1, 2).contiguous().view(1, -1, outer_patch_size, outer_patch_size)
        all_patches.append(patches.squeeze(0))

    
    cat_patches = torch.cat(all_patches, dim=0)

    return torch.cat(all_patches, dim=0), image_information

def reconstruct_image_from_inner_patches(patches, image_information, inner_patch_size):
    """
    Reconstruct an image from the inner parts of patches, using metadata for patch placement.
    
    Args:
        patches (torch.Tensor): Tensor of patches of shape (batch_size, num_patches, outer_patch_size, outer_patch_size).
        image_information (List[Tuple[int, int]]): List of tuples containing the number of vertical and horizontal patches for each image.
        inner_patch_size (int): Size of the inner part of each patch used for reconstruction.
        
    Returns:
        torch.Tensor: Reconstructed images of shape (batch_size, 1, image_height, image_width).
    """
    batch_size = len(image_information)
    reconstructed_images = []

    start_index = 0
    for i in range(batch_size):
        num_patches_vertical, num_patches_horizontal = image_information[i]
        num_patches = num_patches_vertical * num_patches_horizontal
        single_image_patches = patches[start_index:start_index + num_patches]
        start_index += num_patches

        # Extract the 16x16 center of each 32x32 patch
        center_start = (32 - inner_patch_size) // 2
        inner_patches = single_image_patches[:, center_start:center_start + inner_patch_size, center_start:center_start + inner_patch_size]

        # Flatten the patches for reconstruction
        patches_flattened = inner_patches.reshape(num_patches, inner_patch_size * inner_patch_size).transpose(0, 1)
        image_height = num_patches_vertical * inner_patch_size
        image_width = num_patches_horizontal * inner_patch_size

        # Reconstruct the image
        recon_image = F.fold(patches_flattened.unsqueeze(0), (image_height, image_width), kernel_size=(inner_patch_size, inner_patch_size), stride=inner_patch_size)
        reconstructed_images.append(recon_image.squeeze(0))

    return torch.cat(reconstructed_images, dim=0).unsqueeze(1)
