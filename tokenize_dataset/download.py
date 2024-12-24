"""
Example usage:

# Download Cosmos tokenizer models
python download_cosmos.py download-models

# Download raw ImageNet dataset
python download_cosmos.py download-imagenet

# Download preprocessed ImageNet dataset (recommended)
python download_cosmos.py download-preprocessed

# For faster downloads, first:
# pip install hf-transfer
# export HF_HUB_ENABLE_HF_TRANSFER=1
"""

import click
from huggingface_hub import login, snapshot_download
from datasets import load_dataset
import os


@click.group()
def cli():
    """Download Cosmos models and datasets"""
    pass


@cli.command()
def download_models():
    """Download Cosmos tokenizer models from HuggingFace"""
    model_names = [
        "Cosmos-Tokenizer-CI8x8",
        "Cosmos-Tokenizer-DI8x8",
    ]
    for model_name in model_names:
        hf_repo = "nvidia/" + model_name
        local_dir = "pretrained_ckpts/" + model_name
        os.makedirs(local_dir, exist_ok=True)
        print(f"Downloading {model_name}...")
        snapshot_download(repo_id=hf_repo, local_dir=local_dir)
    print("Finished downloading models")


@cli.command()
def download_imagenet():
    """Download ImageNet dataset from HuggingFace"""
    print("Downloading ImageNet training split...")
    # train_dataset = load_dataset("imagenet-1k", split="train")
    print("Sample training images:")
    # for i in range(5):
    #     print(train_dataset[i]["image"])

    print("\nDownloading ImageNet validation split...")
    val_dataset = load_dataset("imagenet-1k", split="validation")
    print("Finished downloading ImageNet")


@cli.command()
def download_preprocessed():
    """Download preprocessed dataset from HuggingFace"""
    print("Downloading preprocessed ImageNet dataset...")
    print(
        "Note: Install hf-transfer and set HF_HUB_ENABLE_HF_TRANSFER=1 for faster downloads"
    )
    hf_repo = "fal/cosmos-imagenet"
    os.makedirs("preprocessed_dataset", exist_ok=True)
    snapshot_download(
        repo_id=hf_repo, local_dir="preprocessed_dataset", repo_type="dataset"
    )
    print("Finished downloading preprocessed dataset")


if __name__ == "__main__":
    cli()
