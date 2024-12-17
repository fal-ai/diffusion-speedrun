import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from safetensors import safe_open
from cosmos_tokenizer.image_lib import ImageTokenizer
import json
from imagenet_labels import IMGNET_LABELS
import click


class SafeTensorDataset(Dataset):
    def __init__(self, safetensor_path="./imagenet_di8x8.safetensors"):
        """
        Dataset that loads tokenized images from safetensors format
        Supports both discrete and continuous tokenizers
        """
        self.safetensor_path = safetensor_path

        # Load metadata
        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                self.total_samples = int(self.metadata["total_samples"])
                self.tokenizer_type = self.metadata.get("tokenizer_type", "discrete")

        with safe_open(self.safetensor_path, framework="pt") as f:
            if self.tokenizer_type == "continuous":
                self.data = f.get_tensor("latents").to(torch.bfloat16) * 16.0 / 255.0
            else:
                self.data = f.get_tensor("indices").to(torch.uint16)
            self.labels = f.get_tensor("labels")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]

        return {"data": data, "label": label}


def visualize_samples(input_file, output_dir, device, num_samples, stride):
    # Create dataset
    dataset = SafeTensorDataset(input_file)
    is_continuous = dataset.tokenizer_type == "continuous"
    click.echo(f"Using {'continuous' if is_continuous else 'discrete'} tokenizer")

    # Load appropriate Cosmos tokenizer
    model_name = "Cosmos-Tokenizer-CI8x8" if is_continuous else "Cosmos-Tokenizer-DI8x8"
    decoder = ImageTokenizer(
        checkpoint_dec=f"pretrained_ckpts/{model_name}/decoder.jit"
    ).to(device)

    os.makedirs(output_dir, exist_ok=True)

    # Process specified samples
    for i in range(0, min(num_samples, len(dataset)), stride):
        sample = dataset[i]
        data = sample["data"]

        if is_continuous:
            data = data.reshape(1, 16, 32, 32).to(device).to(torch.bfloat16)
            click.echo("\nLatents Stats:")
            click.echo(f"Mean: {data.float().mean():.2f}")
            click.echo(f"Std: {data.float().std():.2f}")
            click.echo(f"Min: {data.min():.2f}")
            click.echo(f"Max: {data.max():.2f}")
        else:
            # For discrete tokenizer, reshape to [1, 32, 32]
            data = data.reshape(1, 32, 32).to(device).long()
            click.echo("\nToken Indices Stats:")
            click.echo(f"Mean: {data.float().mean():.2f}")
            click.echo(f"Std: {data.float().std():.2f}")
            click.echo(f"Min: {data.min()}")
            click.echo(f"Max: {data.max()}")

        # Decode the image
        with torch.no_grad():
            reconstructed = decoder.decode(data)

        # Convert to PIL image
        img = (
            ((reconstructed[0].cpu().float() + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        )
        img = img.permute(1, 2, 0).numpy()
        img = Image.fromarray(img)

        # Get label information
        label_idx = sample["label"].item()
        label_str = IMGNET_LABELS[label_idx]

        # Save the image
        save_path = os.path.join(
            output_dir,
            f"reconstructed_{i:07d}_{label_str}_{dataset.tokenizer_type}.png",
        )
        img.save(save_path)
        click.echo(f"Saved image: {save_path}")
        click.echo(f"Label: {label_str} (index: {label_idx})")


@click.command()
@click.option(
    "--input-file",
    default="preprocessed_dataset/imagenet_di8x8.safetensors",
    help="Input safetensors file",
)
@click.option(
    "--output-dir",
    default="reconstructed_images",
    help="Output directory for reconstructed images",
)
@click.option("--device", default="cuda:0", help="Device to use for decoding")
@click.option("--num-samples", default=10, help="Number of samples to process")
@click.option("--stride", default=1, help="Stride between samples")
def main(input_file, output_dir, device, num_samples, stride):
    """Visualize tokenized images from safetensors"""
    visualize_samples(input_file, output_dir, device, num_samples, stride)


if __name__ == "__main__":
    # python visualize.py --input-file imagenet_ci8x8.safetensors
    main()
