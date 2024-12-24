import torch
import click
from train_gpt import ImageGPT, GPTConfig
import numpy as np
from PIL import Image
import os
from cosmos.image_lib import ImageTokenizer


@click.command()
@click.option("--checkpoint", required=True, help="Path to model checkpoint")
@click.option("--num_samples", default=5, help="Number of samples to generate")
@click.option("--temperature", default=0.9, help="Sampling temperature")
@click.option("--top_k", default=100, help="Top-k sampling parameter")
@click.option(
    "--class_ids",
    default="1,130,933,833",
    help="Comma-separated list of class IDs to generate",
)
@click.option(
    "--output_dir",
    default="generated_images",
    help="Output directory for generated images",
)
def generate(checkpoint, num_samples, temperature, top_k, class_ids, output_dir):
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location="cpu")
    config = ckpt["config"]

    # Initialize model
    model = ImageGPT(config)
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Initialize decoder
    decoder = ImageTokenizer(
        checkpoint_dec="tokenize_dataset/pretrained_ckpts/Cosmos-Tokenizer-DI8x8/decoder.jit"
    ).to(device)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Parse class IDs
    class_ids = [int(x) for x in class_ids.split(",")]

    print(
        f"Generating {len(class_ids)} samples with temperature={temperature}, top_k={top_k}"
    )

    with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.bfloat16):
        # Generate samples
        test_classes = torch.tensor(class_ids, device=device)

        samples = model.generate(
            test_classes,
            max_tokens=1024,  # 32x32 image
            temperature=temperature,
            top_k=None,
        )

        # Decode and save each sample
        for i, (sample, class_id) in enumerate(zip(samples, class_ids)):
            # Reshape to [1, 32, 32] for decoder
            tokens = sample.reshape(1, 32, 32)

            # Decode the image
            reconstructed = decoder.decode(tokens)

            # Convert to PIL image
            img = (
                ((reconstructed[0].cpu().float() + 1) * 127.5)
                .clamp(0, 255)
                .to(torch.uint8)
            )
            img = img.permute(1, 2, 0).numpy()
            img = Image.fromarray(img)

            # Save the image
            save_path = os.path.join(
                output_dir, f"generated_{i:03d}_class{class_id}.png"
            )
            img.save(save_path)
            print(f"Saved image: {save_path}")


if __name__ == "__main__":
    generate()
