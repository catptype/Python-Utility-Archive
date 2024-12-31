import os
import random
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple

class Visualize:
    """A utility class for visualizing datasets with random image sampling."""

    @staticmethod
    def show_random_datasets(
        dataset_root: str,
        num_images: int = 5,
        rows: int = 1,
        cols: int = 5,
        figsize: Tuple[int, int] = (16, 7),
    ) -> None:
        """
        Displays random images from a dataset directory.

        Args:
            dataset_root (str): The root directory of the dataset.
            num_images (int): The number of random images to display.
            rows (int): The number of rows in the plot grid.
            cols (int): The number of columns in the plot grid.
            figsize (Tuple[int, int]): The figure size for the plot.

        Raises:
            ValueError: If `num_images` exceeds the number of available images.
            FileNotFoundError: If the `dataset_root` does not exist or contains no images.
        """
        # Validate dataset directory
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")

        # Gather all image paths
        image_list = [os.path.join(root, image) for root, _, files in os.walk(dataset_root) for image in files]
        if not image_list:
            raise FileNotFoundError("No images found in the specified dataset directory.")

        # Validate the number of images to sample
        if num_images > len(image_list):
            raise ValueError(f"Requested {num_images} images, but only {len(image_list)} are available.")

        # Randomly sample images
        random_images = random.sample(image_list, num_images)

        # Set up the plot
        plt.figure(figsize=figsize)
        for idx, image_path in enumerate(random_images):
            # Extract class name from directory structure
            info_list = image_path.split(os.path.sep)
            main_class = info_list[-2] if len(info_list) > 1 else "Unknown"

            try:
                # Load image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Plot image
                plt.subplot(rows, cols, idx + 1)
                plt.imshow(image)
                plt.axis("off")
                plt.title(f"Class: {main_class}\nShape: {image.width}×{image.height}")

            except Exception as e:
                # Handle image loading errors gracefully
                print(f"Error loading image {image_path}: {e}")
                break

        plt.tight_layout()
        plt.show()
