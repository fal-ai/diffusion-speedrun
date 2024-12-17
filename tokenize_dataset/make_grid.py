import os
import glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import torch as th
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from functools import lru_cache
import requests
from io import BytesIO


@lru_cache
def get_font(bold=True, size=32):
    if bold:
        font_url = "https://github.com/openmaptiles/fonts/raw/master/noto-sans/NotoSans-Bold.ttf"
    else:
        font_url = "https://github.com/openmaptiles/fonts/raw/master/noto-sans/NotoSans-Regular.ttf"
    font_response = requests.get(font_url)
    font_data = BytesIO(font_response.content)
    font = ImageFont.truetype(font_data, size)
    return font


def create_grid_visualization(
    image_dir="reconstructed_images", grid_size=4, output_path="comparison_grid.png"
):
    """Create a side-by-side grid comparison of discrete and continuous reconstructions"""

    # Find all images
    discrete_images = sorted(glob.glob(os.path.join(image_dir, "*discrete.png")))
    continuous_images = sorted(glob.glob(os.path.join(image_dir, "*continuous.png")))

    # Take only the first grid_size^2 images
    n_images = grid_size * grid_size
    discrete_images = discrete_images[:n_images]
    continuous_images = continuous_images[:n_images]

    # Load all images
    d_imgs = [Image.open(f) for f in discrete_images]
    c_imgs = [Image.open(f) for f in continuous_images]

    # Get image size
    img_width, img_height = d_imgs[0].size

    # Create canvas for the full comparison
    margin = 30  # Margin between grids
    title_height = 60  # Height for titles
    canvas_width = (img_width * grid_size * 2) + (3 * margin)
    canvas_height = (img_height * grid_size) + (2 * margin) + title_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    # Add titles
    font_size = 40
    try:
        font = get_font(bold=True, size=font_size)
    except:
        font = ImageFont.load_default()
        print("Failed to load font, using default")

    # Draw titles
    draw.text(
        (margin + 50, 10),
        "Discrete",
        fill="black",
        font=font,
    )
    draw.text(
        (2 * margin + img_width * grid_size + 50, 10),
        "Continuous",
        fill="black",
        font=font,
    )

    # Paste discrete images
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j < len(d_imgs):
                x = margin + j * img_width
                y = margin + title_height + i * img_height
                canvas.paste(d_imgs[i * grid_size + j], (x, y))

    # Paste continuous images
    for i in range(grid_size):
        for j in range(grid_size):
            if i * grid_size + j < len(c_imgs):
                x = 2 * margin + grid_size * img_width + j * img_width
                y = margin + title_height + i * img_height
                canvas.paste(c_imgs[i * grid_size + j], (x, y))

    # Save the result
    canvas.save(output_path)
    print(f"Saved comparison grid to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create grid visualization of discrete and continuous reconstructions"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="reconstructed_images",
        help="Directory containing the reconstructed images",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=3,
        help="Size of each grid (grid_size x grid_size)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="comparison_grid.png",
        help="Output path for the comparison grid",
    )

    args = parser.parse_args()
    create_grid_visualization(args.image_dir, args.grid_size, args.output)
