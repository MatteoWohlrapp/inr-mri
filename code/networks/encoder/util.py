import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch


def extract_patches(tensor, patch_size):
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

def process_image(image, encoder):
    '''Process the image through the encoder and return the original image, the output of the encoder and the difference between the two.
    Args:
        image: torch.Tensor: The image to be processed.
        encoder: torch.nn.Module: The encoder model.
    Returns:
        image: torch.Tensor: The original image.
        output: torch.Tensor: The output of the encoder.
        diff: torch.Tensor: The difference between the original image and the output of the encoder.
    '''
    image = image.clone().detach().float().unsqueeze(0).unsqueeze(0)
    image = image.squeeze(0)
    image = image.cpu()
    output = encoder(image)
    output = output.cpu().detach().numpy()
    output = np.squeeze(output)
    diff = np.abs(image.cpu().detach().numpy() - output).squeeze(0)
    return image, output, diff

def plot_images(image, output, diff, title):
    '''Plot the original image, the output of the encoder and the difference between the two.
    Args:
        image: torch.Tensor: The original image.
        output: torch.Tensor: The output of the encoder.
        diff: torch.Tensor: The difference between the original image and the output of the encoder.
        title: str: The title of the plot.
    '''
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(title)
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(output, cmap="gray")
    axs[1].set_title("Reconstructed Image")
    axs[2].imshow(diff, cmap='gray')
    axs[2].set_title('Difference')

def save_plot(dest_dir):
    '''Save the plot to the destination directory.
    Args:
        dest_dir: str: The destination directory.
    '''
    print('saving image')
    plt.savefig(dest_dir)
    print('Done saving image')

def plot_encoder_output(encoder, image, dest_dir, title = 'Image title'):
    '''Test the encoder model on an image and plot the results.
    Args:
        encoder: torch.nn.Module: The encoder model.
        image: torch.Tensor: The image to be processed.
        dest_dir: str: The destination directory to save the plot.
        title: str: The title of the plot.
    '''
    image, output, diff = process_image(image, encoder)
    plot_images(image, output, diff, title)
    save_plot(dest_dir)