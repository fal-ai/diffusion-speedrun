import os
import glob
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from tqdm import tqdm
import json
import click


@click.command()
@click.option(
    "--continuous",
    is_flag=True,
    help="Process continuous tokenizer output instead of discrete",
)
@click.option("--is-validation", is_flag=True, help="Use validation mode")
def concat_shards(continuous=False, is_validation=False):
    """
    Concatenate all numpy shards into a single tensor
    Support both discrete indices and continuous latents
    """
    base_dir = "./cosmos_continuous" if continuous else "./cosmos_discrete"
    base_dir = f"{base_dir}_val" if is_validation else base_dir
    # Get all shard directories
    shard_dirs = sorted(glob.glob(os.path.join(base_dir, "*")))
    print(f"Found {len(shard_dirs)} shard directories")

    all_data = []  # either indices or latents
    all_labels = []
    total_samples = 0

    # Process each shard directory
    for shard_dir in tqdm(shard_dirs, desc="Processing shard directories"):
        # Get metadata
        metadata_path = os.path.join(shard_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                total_samples += metadata["total_samples"]

                # Verify tokenizer type matches
                if continuous and metadata.get("tokenizer_type") != "continuous":
                    raise ValueError(
                        f"Expected continuous tokenizer but found {metadata.get('tokenizer_type')} in {shard_dir}"
                    )
                elif not continuous and metadata.get("tokenizer_type") != "discrete":
                    raise ValueError(
                        f"Expected discrete tokenizer but found {metadata.get('tokenizer_type')} in {shard_dir}"
                    )

        # Process data (either indices or latents)
        data_dir = os.path.join(shard_dir, "latents" if continuous else "indices")
        data_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))

        for data_file in data_files:
            data = np.load(data_file)
            all_data.append(data)

        # Process labels
        labels_dir = os.path.join(shard_dir, "labels")
        labels_files = sorted(glob.glob(os.path.join(labels_dir, "*.npy")))

        for label_file in labels_files:
            labels = np.load(label_file)
            all_labels.append(labels)

    # Concatenate all arrays
    print("Concatenating arrays...")
    all_data = np.concatenate(all_data, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(
        f"Final shapes - {'Latents' if continuous else 'Indices'}: {all_data.shape}, Labels: {all_labels.shape}"
    )

    # Convert to torch tensors with appropriate dtype
    if continuous:
        data_tensor = torch.from_numpy(all_data).to(torch.int8)
        out_name = "imagenet_ci8x8_val" if is_validation else "imagenet_ci8x8"
        tensor_key = "latents"
    else:
        data_tensor = torch.from_numpy(all_data).to(torch.int16)
        out_name = "imagenet_di8x8_val" if is_validation else "imagenet_di8x8"
        tensor_key = "indices"

    labels_tensor = torch.from_numpy(all_labels).to(torch.int16)

    # Create metadata
    metadata = {
        "total_samples": str(int(total_samples)),
        "data_shape": str(list(all_data.shape)),
        "labels_shape": str(list(all_labels.shape)),
        "tokenizer_type": "continuous" if continuous else "discrete",
        "is_validation": str(is_validation),
    }

    # Save as safetensors
    print("Saving to safetensors format...")
    tensors = {tensor_key: data_tensor, "labels": labels_tensor}

    save_file(tensors, f"./{out_name}.safetensors", metadata)

    # Save metadata separately for easy access
    with open(f"./{out_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done! Files saved as:")
    print(f"- ./{out_name}.safetensors")
    print(f"- ./{out_name}_metadata.json")

    # Verify the saved file
    print("\nVerifying saved file...")
    with safe_open(f"./{out_name}.safetensors", framework="pt") as f:
        saved_data = f.get_tensor(tensor_key)
        saved_labels = f.get_tensor("labels")
        print(
            f"Verified shapes - {tensor_key.capitalize()}: {saved_data.shape}, Labels: {saved_labels.shape}"
        )
        if continuous:
            print(f"Latents range: {saved_data.min():.3f} to {saved_data.max():.3f}")
        else:
            print(f"Indices range: {saved_data.min()} to {saved_data.max()}")


if __name__ == "__main__":
    concat_shards()
