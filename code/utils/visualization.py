import matplotlib.pyplot as plt
import os


def show_batch(images, num_images=4, cmap="gray"):
    """
    Display a batch of images using matplotlib.

    Args:
        images (torch.Tensor): A batch of images as a 3D tensor (batch_size x height x width).
        num_images (int): Number of images to display from the batch. Default is 4.
        cmap (str): The colormap to use for displaying the images. Default is 'gray'.
    """
    num_images = min(num_images, images.shape[0])

    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 5))
    if num_images == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(images[i].squeeze(), cmap=cmap)
        ax.axis("off")

    plt.show()


def show_image(image, cmap="gray"):
    """
    Display a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        cmap (str): The colormap to use for displaying the image. Default is 'gray'.
    """
    plt.imshow(image.squeeze(), cmap=cmap)
    plt.axis("off")
    plt.show()


def save_image(image, filename, output_dir):
    """
    Save a single image using matplotlib.

    Args:
        image (torch.Tensor): A single image as a 2D tensor (height x width).
        filename (str): The filename to save the image as.
        output_dir (str): The directory to save the image in.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.imshow(image.squeeze(), cmap="gray")
    plt.axis("off")
    plt.savefig(
        f"{output_dir}/{filename}.png",
        bbox_inches="tight",
        pad_inches=0,
        dpi=1200,
    )
    plt.close()


def reconstruct_from_model(model, model_path, output_dir, filename, img):

    reconstructed_image = model(img)

    if reconstructed_image.is_cuda:
        reconstructed_image = reconstructed_image.cpu()

    save_image(reconstructed_image, f"{filename}_reconstructed", output_dir)
    save_image(img, f"{filename}_gt", output_dir)
