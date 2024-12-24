import os
import sys
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from cosmos_tokenizer.image_lib import ImageTokenizer
import logging
import time
from typing import Any
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp
import click

SIZE = 256

# Initialize logging
logging.basicConfig(level=logging.INFO)


def crop_to_center(image, new_size=SIZE):
    width, height = image.size
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def prepare_image(pil_image, w=SIZE, h=SIZE):
    pil_image = pil_image.resize((w, h), resample=Image.BICUBIC, reducing_gap=1)
    arr = np.array(pil_image.convert("RGB"))
    arr = arr.astype(np.float32) / 127.5 - 1
    arr = np.transpose(arr, [2, 0, 1])
    image = torch.from_numpy(arr)
    return image


class ImageDataset(Dataset):
    def __init__(self, idx_range=(0, 1000), is_validation=False):
        self.dataset = load_dataset(
            "imagenet-1k", split="train" if not is_validation else "validation"
        )
        self.idx_range = idx_range
        if self.idx_range[1] > len(self.dataset):
            self.idx_range = (self.idx_range[0], len(self.dataset))

    def __len__(self):
        return self.idx_range[1] - self.idx_range[0]

    def __getitem__(self, idx):
        idx = idx + self.idx_range[0]
        image, label = self.dataset[idx]["image"], self.dataset[idx]["label"]

        w, h = image.size
        if w > h:
            image = image.resize((int(w * SIZE / h), SIZE), resample=Image.BICUBIC)
        else:
            image = image.resize((SIZE, int(h * SIZE / w)), resample=Image.BICUBIC)

        image = crop_to_center(image, SIZE)
        image = prepare_image(image, SIZE, SIZE)

        return image, label


@torch.no_grad()
def convert_to_numpy(
    idx_range,
    out_root,
    device,
    continuous=False,
    batch_size=8,
    num_workers=4,
    is_validation=False,
):
    logging.info(
        f"Processing on {device} with {'continuous' if continuous else 'discrete'} tokenizer"
    )

    # Load the appropriate Cosmos Tokenizer model
    model_name = "Cosmos-Tokenizer-CI8x8" if continuous else "Cosmos-Tokenizer-DI8x8"

    encoder = ImageTokenizer(
        checkpoint_enc=f"pretrained_ckpts/{model_name}/encoder.jit"
    ).to(device)

    dataset = ImageDataset(idx_range, is_validation=is_validation)

    if dataset.__len__() < 1:
        logging.info("No images to process.")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    # Create output directories
    os.makedirs(out_root, exist_ok=True)
    latents_dir = os.path.join(out_root, "latents" if continuous else "indices")
    labels_dir = os.path.join(out_root, "labels")
    os.makedirs(latents_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    total_samples = len(dataset)
    shard_size = 10000
    current_shard = 0
    current_index = 0

    current_latents = []
    current_labels = []

    inference_latencies = []

    for batch in tqdm(dataloader):
        start_time = time.time()

        processed_images, labels = batch
        processed_images = processed_images.to(device).to(torch.bfloat16)

        # Get encoded representation
        if continuous:
            (latent,) = encoder.encode(processed_images)
            latent = latent.float().cpu().numpy()  # Shape: [B, 16, 64, 64]
            # quantize this with clip range of -8 ~ 8 as int8
            # average is 0.0
            latent = np.clip(latent, -8, 8) / 16.0
            latent = latent * 255
            latent = latent.astype(np.int8)

        else:
            indices, _ = encoder.encode(processed_images)
            latent = indices.to(torch.uint16).cpu().numpy()

        batch_size = len(labels)

        current_latents.extend(latent)
        current_labels.extend(labels.numpy())

        current_index += batch_size

        if current_index >= shard_size or current_index >= total_samples:
            shard_name = f"shard_{str(current_shard).zfill(5)}"

            # Convert lists to numpy arrays and save
            if continuous:
                latents_array = np.array(current_latents, dtype=np.int8)
            else:
                latents_array = np.array(current_latents, dtype=np.uint16)
            labels_array = np.array(current_labels, dtype=np.int64)

            save_name = "latents" if continuous else "indices"
            np.save(
                os.path.join(latents_dir, f"{shard_name}_{save_name}.npy"),
                latents_array,
            )
            np.save(os.path.join(labels_dir, f"{shard_name}_labels.npy"), labels_array)

            current_latents = []
            current_labels = []
            current_shard += 1
            current_index = 0

        inference_latencies.append(time.time() - start_time)

    # Save any remaining data
    if current_latents:
        shard_name = f"shard_{str(current_shard).zfill(5)}"
        if continuous:
            latents_array = np.array(current_latents, dtype=np.int8)
        else:
            latents_array = np.array(current_latents, dtype=np.uint16)
        labels_array = np.array(current_labels, dtype=np.int64)

        save_name = "latents" if continuous else "indices"
        np.save(
            os.path.join(latents_dir, f"{shard_name}_{save_name}.npy"), latents_array
        )
        np.save(os.path.join(labels_dir, f"{shard_name}_labels.npy"), labels_array)

    metadata = {
        "total_samples": total_samples,
        "shard_size": shard_size,
        "num_shards": current_shard + 1,
        "average_inference_latency": np.mean(inference_latencies),
        "tokenizer_type": "continuous" if continuous else "discrete",
    }

    with open(os.path.join(out_root, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    logging.info(
        f"Average Inference Latency on {device}: {np.mean(inference_latencies)} seconds"
    )


def process_single_job(
    file_index,
    gpu_id,
    continuous=False,
    batch_size=64,
    num_workers=8,
    is_validation=False,
):
    print(f"Single Job: Is validation: {is_validation}")
    print(f"GPU ID: {gpu_id}")
    """Process a single file on specified GPU"""
    device = f"cuda:{gpu_id}"
    name = "continuous" if continuous else "discrete"
    name = f"{name}_val" if is_validation else name
    print(f"Processing {name} on GPU {gpu_id}, file {file_index}")
    out_root = f"./cosmos_{name}/{str(file_index).zfill(5)}"

    idx_range = (file_index * 100000, (file_index + 1) * 100000)
    convert_to_numpy(
        idx_range, out_root, device, continuous, batch_size, num_workers, is_validation
    )


def launch_multi_gpu_jobs(num_gpus, total_files, continuous, batch_size, is_validation):
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        click.echo("No GPUs found!")
        sys.exit(1)

    num_gpus = num_gpus or available_gpus
    num_gpus = min(num_gpus, available_gpus)

    click.echo(f"Found {available_gpus} GPUs, using {num_gpus}")
    click.echo(f"Using {'continuous' if continuous else 'discrete'} tokenizer")

    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        click.echo(
            f"GPU {i}: {props.name}, {props.total_memory / 1024**2:.0f}MB memory"
        )

    files_per_gpu = (total_files + num_gpus - 1) // num_gpus

    ctx = mp.get_context("spawn")
    processes = []
    print(f"Launching {num_gpus} processes, Is validation: {is_validation}")
    for gpu_id in range(num_gpus):
        start_file = gpu_id * files_per_gpu
        end_file = min((gpu_id + 1) * files_per_gpu, total_files)

        for file_idx in range(start_file, end_file):
            p = ctx.Process(
                target=process_single_job,
                args=(file_idx, gpu_id, continuous, batch_size, 8, is_validation),
            )
            processes.append((p, gpu_id, file_idx))
            p.start()
            click.echo(f"Started process for GPU {gpu_id}, file {file_idx}")

    for p, gpu_id, file_idx in processes:
        p.join()
        if p.exitcode == 0:
            click.echo(f"Successfully completed GPU {gpu_id}, file {file_idx}")
        else:
            click.echo(f"Failed GPU {gpu_id}, file {file_idx}")


@click.command()
@click.option(
    "--num-gpus",
    type=int,
    default=None,
    help="Number of GPUs to use. Defaults to all available GPUs.",
)
@click.option(
    "--total-files",
    type=int,
    default=13,
    help="Total number of ImageNet shards to process",
)
@click.option("--batch-size", type=int, default=64, help="Batch size per GPU")
@click.option(
    "--continuous", is_flag=True, help="Use continuous tokenizer instead of discrete"
)
@click.option("--single-job", is_flag=True, help="Run only a single job (for testing)")
@click.option(
    "--file-index", type=int, default=0, help="File index for single job mode"
)
@click.option("--gpu-id", type=int, default=0, help="GPU ID for single job mode")
@click.option("--is-validation", is_flag=True, help="Use validation mode")
def main(
    num_gpus,
    total_files,
    batch_size,
    continuous,
    single_job,
    file_index,
    gpu_id,
    is_validation,
):
    print(f"Processing {is_validation} on GPU {gpu_id}, file {file_index}")
    """Multi-GPU ImageNet Processing with Cosmos Tokenizer"""
    if single_job:
        process_single_job(
            file_index,
            gpu_id,
            continuous,
            batch_size,
            is_validation=is_validation,
        )
    else:
        launch_multi_gpu_jobs(
            num_gpus, total_files, continuous, batch_size, is_validation=is_validation
        )


if __name__ == "__main__":
    # example usage:
    # python tokenize_imagenet_numpy.py --num-gpus 8 --total-files 8 --is-validation
    # python tokenize_imagenet_numpy.py --num-gpus 8 --total-files 8 --continuous --is-validation
    # python tokenize_imagenet_numpy.py --num-gpus 1 --total-files 1 --batch-size 64 --continuous --single-job --file-index 0 --gpu-id 0 --is-validation
    main()
