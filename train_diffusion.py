import os
from torch.nn.utils import clip_grad_norm_

from transport import create_transport
from transport.transport import Sampler
from dit import SiT_models
from eval import Eval
import click
import torch
import torch.nn.functional as F
import numpy as np
import random
import torch.distributed as dist
from torch.backends.cuda import (
    enable_cudnn_sdp,
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
)
from torchvision.utils import make_grid
import copy
from torch.nn.attention import SDPBackend, sdpa_kernel
from datetime import datetime
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import wandb
from collections import OrderedDict
from tqdm import tqdm, trange
import cosmos.image_lib
import cosmos
from safetensors.torch import safe_open


SCALING_FACTOR = 0.4
NUM_CHANNELS = 16


class IMAGENET(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        dpath = "./inet/imagenet_ci8x8.safetensors" if is_train else "./inet/imagenet_ci8x8_val.safetensors"
        with safe_open(dpath, framework="pt") as f:
            self.labels, self.latents = f.get_tensor("labels"), (f.get_tensor("latents") ) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.latents[idx]
        label = self.labels[idx]
        image = image.type(torch.float32) * 16.0 / 255.0
        return image, int(label)



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def cleanup():
    dist.destroy_process_group()


def cosmos_vae(device="cuda:0"):

    vae = cosmos.image_lib.ImageTokenizer(
        checkpoint_dec= "./cosmos_ckpt/decoder.jit",
    ).to(device)


    def decode(z):
        latent = z
        latent = latent.type(torch.bfloat16)
        with torch.no_grad():
            return vae.decode(latent)

    return None, decode


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


@click.command()
@click.option("--run_name", default="run_1", help="Name of the run")
@click.option("--global_batch_size", default=256, help="Global batch size across all GPUs")
@click.option("--global_seed", default=4, help="Global seed")
@click.option("--per_gpu_batch_size", default=32, help="Per GPU batch size")
@click.option("--num_iterations", default=500_000, help="Number of training iterations")
@click.option("--learning_rate", default=1e-4, help="Learning rate")
@click.option("--sample_every", default=10_000, help="Sample frequency")
@click.option("--val_every", default=2_000, help="Validation frequency")
@click.option("--kdd_every", default=2_000, help="KDD evaluation frequency")
@click.option("--save_every", default=2_000, help="Checkpoint save frequency")
@click.option("--init_ckpt", default=None, help="Path to initial checkpoint")
@click.option("--cfg_scale", default=1.5, help="CFG scale during KDD evaluation")
@click.option("--uncond_prob", default=0.1, help="Probability of dropping label for unconditional training")
def main(run_name, global_batch_size, global_seed, per_gpu_batch_size, num_iterations,
         learning_rate, sample_every, val_every, kdd_every, save_every, init_ckpt, cfg_scale, uncond_prob):

    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    torch.manual_seed(global_seed + ddp_rank)
    np.random.seed(global_seed + ddp_rank)
    random.seed(global_seed + ddp_rank)

    ##########################################################################
    #                   DDP Initialization and Basic Setup                   #
    ##########################################################################
    val_per_gpu_batch_size = per_gpu_batch_size * 2
    dist.init_process_group(backend="nccl")
    
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)


    grad_accum_steps = int(global_batch_size // (per_gpu_batch_size * ddp_world_size))
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{date_time}_{run_name}"

    if master_process:
        print(f"Global batch size: {global_batch_size}")
        print(f"Per GPU batch size: {per_gpu_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch size per step: {per_gpu_batch_size * ddp_world_size}")

        wandb.init(
            project="imagegpt",
            name=run_name,
            config={
                "global_batch_size": global_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "sample_every": sample_every,
                "val_every": val_every,
                "kdd_every": kdd_every,
                "save_every": save_every,
                "cfg_scale": cfg_scale,
                "uncond_prob": uncond_prob
            },
        )
        wandb.run.log_code(".")

    # Allow tf32 for speed
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = SiT_models['SiT-XL/2'](in_channels=NUM_CHANNELS).to(memory_format=torch.channels_last)  # From your code
    model = torch.compile(model)
    # ema model
    ema = copy.deepcopy(model)
    model = model.to(device)
    ema = ema.to(device)    
    requires_grad(ema, False)
    # ema 
    if init_ckpt is not None and master_process:
        print(f"Loading checkpoint from {init_ckpt}")
    if init_ckpt is not None:
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])

    random_tensor = torch.ones(1000, 1000, device=device) * ddp_rank
    dist.all_reduce(random_tensor, op=dist.ReduceOp.SUM)
    if master_process:
        print(f"Rank {ddp_rank} has value {random_tensor[0, 0].item()}\n")

    # Wrap in DDP
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False)
    transport = create_transport(
        "Linear",
        "velocity",
        "velocity",
        1e-3,
        1e-3
    )  # default: velocity; 
    transport_sampler = Sampler(transport)

    # make x_embedder & final_layer lr / 4 , rest lr, optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0,
    )



    enable_cudnn_sdp(True)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)


    train_dataset = IMAGENET(is_train=True)
    val_dataset = IMAGENET(is_train=False)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True, seed=global_seed
    )
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )



    evaluator = Eval()  # Uses DINOv2 for MMD
    _, decode_fn = cosmos_vae(device=device)  # for latents decode


    seed_for_rank = global_seed + ddp_rank
    # find appropriate # of samples that's more than 2000 and divisible by world size
    num_kdd_samples = ((2000 // ddp_world_size + 1)) * ddp_world_size
    num_kdd_samples_per_rank = num_kdd_samples // ddp_world_size
    fixed_class_ids = torch.randint(0, 1000, (num_kdd_samples_per_rank,), generator=torch.Generator().manual_seed(seed_for_rank)).to(device)
    ##########################################################################
    #                          Helper Functions                              #
    ##########################################################################

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)


    @torch.no_grad()
    def do_validation():
        """
        Compute a simple validation loss across the entire val_loader.
        """
        model.eval()
        val_losses = []

        for val_latents, val_labels in val_loader:
            val_latents = val_latents.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            val_labels = val_labels.to(device, non_blocking=True)

            # Scale latents by SCALING_FACTOR for stable diffusion training
            data_val = val_latents * SCALING_FACTOR

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                model_kwargs = dict(y=val_labels)
                loss_dict = transport.training_losses(model, data_val, model_kwargs)
                loss = loss_dict["loss"].mean()
            val_losses.append(loss.item())

        val_loss = np.mean(val_losses)
        model.train()
        return val_loss

    @torch.no_grad()
    def do_ema_sample(num_samples):
        """
        Sample images using the EMA model with CFG.
        Returns decoded images in uint8 format [0, 255].
        """
        ema.eval()
        z = torch.randn(num_samples, 16, 32, 32, device=device, 
                        generator=torch.Generator(device=device).manual_seed(seed_for_rank))
        y = fixed_class_ids[:num_samples]

        all_imgs = []

        range_fn_imgs = (lambda *args, **kwargs: trange(*args, **kwargs, position=1)) if master_process else range
        sample_fn = transport_sampler.sample_ode()
        

        for i in range_fn_imgs(0, num_samples, val_per_gpu_batch_size):
            z_i = z[i:i+val_per_gpu_batch_size]
            y_i = y[i:i+val_per_gpu_batch_size]
            b_i = z_i.size(0)
            
            z_i = torch.cat([z_i, z_i], dim=0).to(memory_format=torch.channels_last)
            ynull = torch.zeros_like(y_i) + 1000
            y_i = torch.cat([y_i, ynull], dim=0)
            model_kwargs = dict(y=y_i, cfg_scale=cfg_scale)
            model_fn = ema.forward_with_cfg
            # with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                
      
            
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, 
                             SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]): 
                z_i = sample_fn(z_i, model_fn, **model_kwargs)[-1]
                z_i, _ = z_i.chunk(2, dim=0)  # Remove null class samples  
                z_i = z_i / SCALING_FACTOR             
                imgs = decode_fn(z_i)
            imgs = (imgs.clamp(-1,1) + 1) * 127.5
            imgs = imgs.type(torch.uint8)
            all_imgs.append(imgs)
        
        return torch.cat(all_imgs, dim=0)

    @torch.no_grad()
    def do_sample_grid(step):
        samples = do_ema_sample(per_gpu_batch_size) # (PGPU, 3, 256, 256)
        all_samples = torch.zeros((global_batch_size, 3, 256, 256), device=device, dtype=samples.dtype)
        dist.all_gather_into_tensor(all_samples, samples)
        # all_samples = all_samples.permute(0, 2, 3, 1)
        x = make_grid(all_samples, nrow=int(np.sqrt(global_batch_size)))
        x = x.permute(1, 2, 0)
        if master_process:
            sample = Image.fromarray(x.cpu().numpy())
            sample.save("sample.jpg", quality=50)
            sample.save("sample_hq.jpg", quality=95)
            wandb.log({f"samples": wandb.Image("./sample.jpg"),f"samples_hq": wandb.Image("./sample_hq.jpg")}, step=step)
        dist.barrier()

    @torch.no_grad()
    def do_kdd_evaluation():
        imgs = do_ema_sample(num_kdd_samples_per_rank)
        mmd = evaluator.eval(imgs)
        return mmd

    ##########################################################################
    #                             Training Loop                              #
    ##########################################################################

    update_ema(ema, model.module, 0.0)
    model.train()
    ema.eval()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(range(num_iterations), desc="Training", position=0) if master_process else range(num_iterations)
    running_loss = []
    for step in pbar:
        epoch = step // len(train_loader)
        train_sampler.set_epoch(epoch)
        
        # Gradient Accumulation
        for micro_step in range(grad_accum_steps):
            try:
                latents, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                latents, labels = next(train_iter)

            latents = latents.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                latents = latents * SCALING_FACTOR

            with ctx:
                model_kwargs = dict(y=labels)
                loss_dict = transport.training_losses(model, latents, model_kwargs)
                loss = loss_dict["loss"].mean()
                running_loss.append(loss.item())


            loss.backward()
        
        clip_grad_norm_(model.parameters(), max_norm=1.0)  # or any suitable value
        optimizer.step()
        # ema beta should be 0 for first 10k steps and then annealed to 0.999 within 100k steps
        ema_beta = 0.0 if (step < 10000) else 0.999
        update_ema(ema, model.module, ema_beta)
        optimizer.zero_grad(set_to_none=True)

        # Logging
        if master_process and step % 10 == 0:
            wandb.log({
                "train/loss": np.mean(running_loss),
            }, step=step)
            running_loss = []
        # Validation
        if  step % val_every == 0:
            val_loss = do_validation()
            if master_process:
                wandb.log({"val/loss": val_loss}, step=step)



        # KDD Evaluation
        if  step % kdd_every == 0:
            kdd = do_kdd_evaluation()
            if master_process:
                # print(f"step: {step}, kdd: {kdd:.4f}")
                wandb.log({"kdd/mmd": kdd}, step=step)

        # Sample
        if step % sample_every == 0:
            do_sample_grid(step)

        # Save Checkpoints
        if master_process and step > 0 and step % save_every == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "ema": ema.state_dict(),
            }
            os.makedirs(f"logs/ckpts_{run_id}", exist_ok=True)
            ckpt_path = f"logs/ckpts_{run_id}/step_{step}.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)



    if master_process:
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()