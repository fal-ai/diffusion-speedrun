import os
import sys
import uuid
import glob
import time
import click
import wandb
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from safetensors import safe_open
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime

from torch.backends.cuda import (
    enable_cudnn_sdp,
    enable_flash_sdp,
    enable_math_sdp,
    enable_mem_efficient_sdp,
)


@dataclass
class GPTConfig:
    vocab_size: int = 64 * (68000 // 64)
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    num_classes: int = 1000
    wte_init_std: float = 0.02
    v_residual: bool = False


class ImageTokenDataset(Dataset):
    def __init__(self, safetensor_path="./imagenet_di8x8.safetensors", debug=False):
        self.safetensor_path = safetensor_path

        metadata_path = safetensor_path.replace(".safetensors", "_metadata.json")
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)
            self.total_samples = self.metadata["total_samples"]

        if debug:
            self.total_samples = 10

        with safe_open(safetensor_path, framework="pt") as f:
            self.indices = f.get_tensor("indices").to(torch.uint16).long()
            self.labels = f.get_tensor("labels").long()

        if debug:
            self.indices = self.indices[:10]
            self.labels = self.labels[:10]

    def __len__(self):
        return int(self.total_samples)

    def __getitem__(self, idx):
        indices = self.indices[idx].reshape(-1)
        # replace randomly with 1000
        indices = indices.masked_fill_(torch.rand((indices.shape)) < 0.05, 1000)
        class_label = self.labels[idx]

        return {"input_ids": indices, "class_label": class_label}


class Rotary(torch.nn.Module):
    def __init__(self, dim, base=100, h=128, w=128, var_like_order=False):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / (dim)))
        self.h = h
        self.w = w

        t_h = torch.arange(h).type_as(self.inv_freq)
        t_w = torch.arange(w).type_as(self.inv_freq)
        freqs_h = torch.outer(t_h, self.inv_freq).unsqueeze(1)
        freqs_w = torch.outer(t_w, self.inv_freq).unsqueeze(0)
        freqs_h = freqs_h.repeat(1, w, 1)
        freqs_w = freqs_w.repeat(h, 1, 1)
        freqs_hw = torch.cat([freqs_h, freqs_w], 2)

        self.register_buffer("freqs_hw_cos", freqs_hw.cos())
        self.register_buffer("freqs_hw_sin", freqs_hw.sin())
        self.cache_cos = None
        self.cache_sin = None

    def forward(
        self, x, height_width=None, extend_with_register_tokens=0, augment=False
    ):
        if self.cache_cos is not None and self.cache_sin is not None:
            return self.cache_cos, self.cache_sin

        if height_width is not None:
            this_h, this_w = height_width
        else:
            this_hw = x.shape[1]
            this_h, this_w = int(this_hw**0.5), int(this_hw**0.5)

        if augment:
            start_h = torch.randint(0, self.h - this_h + 1, (1,)).item()
            start_w = torch.randint(0, self.w - this_w + 1, (1,)).item()
        else:
            start_h = 0
            start_w = 0

        cos = self.freqs_hw_cos[start_h : start_h + this_h, start_w : start_w + this_w]
        sin = self.freqs_hw_sin[start_h : start_h + this_h, start_w : start_w + this_w]

        cos = cos.clone().reshape(this_h * this_w, -1)
        sin = sin.clone().reshape(this_h * this_w, -1)

        if extend_with_register_tokens > 0:
            cos = torch.cat(
                [
                    torch.ones(extend_with_register_tokens, cos.shape[1]).to(
                        cos.device
                    ),
                    cos,
                ],
                0,
            )
            sin = torch.cat(
                [
                    torch.zeros(extend_with_register_tokens, sin.shape[1]).to(
                        sin.device
                    ),
                    sin,
                ],
                0,
            )

        self.cache_cos = cos[None, :, None, :]
        self.cache_sin = sin[None, :, None, :]

        return self.cache_cos, self.cache_sin  # 1, T, 1, D


def apply_rotary_emb(x, cos, sin):
    cos, sin = cos[:, : x.shape[1]], sin[:, : x.shape[1]]
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

        if config.v_residual:
            self.lamb1 = nn.Parameter(torch.tensor(0.5))
            self.lamb2 = nn.Parameter(torch.tensor(0.5))
        else:
            self.lamb1 = 1.0
            self.lamb2 = 0.0

    def forward(self, x, kv_cache=None, freq=None, v1=None):
        B, T, C = x.size()  # if this is sampling, T would be 1.

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)  # B, T, n_head, D
        cos, sin = freq

        if v1 is None:
            v1 = v

        v = self.lamb1 * v + self.lamb2 * v1.view_as(v)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache

            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))

            if k_cache is not None:
                if isinstance(k_cache, int):
                    k_cache = k
                    v_cache = v
                else:
                    k = torch.cat([k_cache, k], dim=1)
                    v = torch.cat([v_cache, v], dim=1)  # it cats in T dim.

                new_kv_cache = (k, v)

            # do classic attention.
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
            )

        else:

            q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
            new_kv_cache = None
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
            )

        y = y.transpose(1, 2).contiguous().view_as(x)
        y = self.c_proj(y)
        return (y, v1), new_kv_cache


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, freq=None, v1=None):
        (attn_out, v1), new_kv_cache = self.attn(
            F.rms_norm(x, (x.size(-1),)), kv_cache, freq, v1=v1
        )
        x = x + attn_out
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return (x, v1), new_kv_cache


class ImageGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size + config.num_classes, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_()
        self.rotary = Rotary(config.n_embd // (2 * config.n_head))
        self.transformer.wte.weight.data.normal_(mean=0.0, std=config.wte_init_std)

    def forward(self, token_indices, class_labels):
        b, t = token_indices.size()
        class_tokens = class_labels + 65536
        targets = token_indices
        token_indices = token_indices[:, :-1]
        token_sequence = torch.cat([class_tokens.unsqueeze(1), token_indices], dim=1)

        freq = self.rotary(None, height_width=(32, 32))

        x = self.transformer.wte(token_sequence)

        # add first element to all of the embedding
        x = x + x[:, 0:1, :]
        x = F.rms_norm(x, (x.size(-1),))

        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, freq=freq, v1=v1)[0]

        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = logits.float()

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )

        return logits, loss

    @torch.no_grad()
    def generate(self, class_labels, max_tokens=1024, temperature=1.0, top_k=None):
        b = class_labels.size(0)
        device = class_labels.device

        # do unconditional generation as well
        class_labels = class_labels.repeat(2)
        class_labels[b:] = 1000
        x = (class_labels + 65536).unsqueeze(1)

        kv_caches = [(0, 0)] * len(self.transformer.h)
        x_init_embed = self.transformer.wte(x)

        freq = self.rotary(None, height_width=(32, 32))
        cos, sin = freq
        x_all = x

        for i in range(max_tokens):
            x_emb = self.transformer.wte(x_all[:, -1:])

            x_emb = x_emb + x_init_embed
            x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))

            cos_local = cos[:, i : i + 1, :, :]
            sin_local = sin[:, i : i + 1, :, :]
            freq_local = (cos_local, sin_local)
            v1 = None
            for j, block in enumerate(self.transformer.h):
                (x_emb, v1), new_kv_cache = block(
                    x_emb, kv_caches[j], freq=freq_local, v1=v1
                )
                kv_caches[j] = new_kv_cache

            x_emb = F.rms_norm(x_emb, (x_emb.size(-1),))
            logits = self.lm_head(x_emb)

            # do uncond
            logits_cond = logits[:b, :]
            logits_uncond = logits[b:, :]

            logits = logits_uncond + 5.0 * (logits_cond - logits_uncond)
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits.squeeze(1), dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx_next = idx_next.repeat(2, 1)
            x_all = torch.cat((x_all, idx_next), dim=1)

        return x_all[:, 1:]


@click.command()
@click.option("--run_name", default="run_1", help="Name of the run")
@click.option(
    "--train_data",
    default="./tokenize_dataset/preprocessed_dataset/imagenet_di8x8.safetensors",
    help="Path to training data",
)
@click.option(
    "--val_data",
    default="./tokenize_dataset/preprocessed_dataset/imagenet_di8x8_val.safetensors",
    help="Path to validation data",
)
@click.option(
    "--global_batch_size", default=16 * 8, help="Global batch size across all GPUs"
)
@click.option("--per_gpu_batch_size", default=16, help="Per GPU batch size")
@click.option("--num_iterations", default=6004, help="Number of training iterations")
@click.option("--learning_rate", default=1e-3, help="Learning rate")
@click.option("--weight_decay", default=0.1, help="Weight decay")
@click.option("--warmup_iters", default=10, help="Warmup iterations")
@click.option("--warmdown_iters", default=2000, help="Warmdown iterations")
@click.option("--val_every", default=500, help="Validation frequency")
@click.option("--save_every", default=1000, help="Checkpoint save frequency")
@click.option("--n_embed", default=768, help="Embedding dimension")
@click.option("--init_ckpt", default=None, help="Path to initial checkpoint")
@click.option("--vres", default=False, help="Use vres")
@click.option("--n_layer", default=24, help="Number of layers")
def train(
    run_name,
    train_data,
    val_data,
    global_batch_size,
    per_gpu_batch_size,
    num_iterations,
    learning_rate,
    weight_decay,
    warmup_iters,
    warmdown_iters,
    val_every,
    save_every,
    n_embed,
    init_ckpt,
    vres,
    n_layer,
):
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0

    # fix all the seed
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    grad_accum_steps = int(global_batch_size // (per_gpu_batch_size * ddp_world_size))
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"{date_time}_{run_name}"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model = ImageGPT(
        GPTConfig(n_layer=n_layer, n_head=6, n_embd=n_embed, v_residual=vres)
    )
    model = model.to(device)

    if init_ckpt is not None:
        print(f"Loading checkpoint from {init_ckpt}")
        checkpoint = torch.load(init_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    random_tensor = torch.ones(1000, 1000).to(device) * ddp_rank
    dist.all_reduce(random_tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {ddp_rank} has value {random_tensor[0, 0].item()}")

    model = DDP(model, device_ids=[ddp_local_rank])
    # if hasattr(config, "coordinate_descent_tuning"):
    #     config.coordinate_descent_tuning = True # suggested by @Chillee

    # model = torch.compile(model, mode="reduce-overhead")

    num_params = sum(p.numel() for p in model.parameters())

    if master_process:
        print(f"Global batch size: {global_batch_size}")
        print(f"Per GPU batch size: {per_gpu_batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch size per step: {per_gpu_batch_size * ddp_world_size}")

        wandb.init(
            project="imagegpt",
            name=run_name,
            config={
                "train_data": train_data,
                "val_data": val_data,
                "global_batch_size": global_batch_size,
                "per_gpu_batch_size": per_gpu_batch_size,
                "grad_accum_steps": grad_accum_steps,
                "num_iterations": num_iterations,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "warmup_iters": warmup_iters,
                "warmdown_iters": warmdown_iters,
                "n_embed": n_embed,
                "num_params": num_params,
                "vres": vres,
            },
        )

        wandb.run.log_code(".")

    ### CONFIGURE OPTIMIZER muP way.
    # wte have higher lr.
    # lm_head slightly higher lr.

    optimizer_grouped_parameters = [
        {"params": model.module.transformer.wte.parameters(), "lr": 0.01},
        {
            "params": model.module.lm_head.parameters(),
            "lr": learning_rate * 2 * 768 / n_embed,
        },
    ]

    for name, param in model.module.transformer.h.named_parameters():
        if "lamb" in name:
            optimizer_grouped_parameters.append(
                {"params": param, "lr": 0.01, "weight_decay": 0.0}
            )
        else:
            optimizer_grouped_parameters.append(
                {"params": param, "lr": learning_rate * 768 / n_embed}
            )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
        fused=True,
    )

    # enable_cudnn_sdp(True)
    enable_flash_sdp(True)
    # enable_mem_efficient_sdp(False)
    # enable_math_sdp(False)

    print(
        f"warmup_iters: {warmup_iters}, num_iterations: {num_iterations}, warmdown_iters: {warmdown_iters}, learning_rate: {learning_rate}"
    )

    def get_lr(it):
        if it < warmup_iters:
            return it / warmup_iters
        if it > num_iterations - warmdown_iters:
            return (num_iterations - it) / warmdown_iters

        return 1

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    train_dataset = ImageTokenDataset(train_data)
    val_dataset = ImageTokenDataset(train_data)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
    )

    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    model.train()
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    for step in range(num_iterations):
        epoch = step // len(train_loader)
        train_sampler.set_epoch(epoch)

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # batch = {
            #     k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            #     for k, v in batch.items()
            # }

            with ctx:
                logits, loss = model(batch["input_ids"], batch["class_label"])

            loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if master_process and step % 1 == 0:
            lr = scheduler.get_last_lr()[0]
            print(f"step: {step}, loss: {loss.item():.2f}, lr: {lr:.2e}")
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "train/step": step})

        if step > 0 and step % val_every == 0:
            model.eval()
            val_losses = []

            with torch.no_grad():
                for val_batch in val_loader:
                    # val_batch = {
                    #     k: (
                    #         v.to(device, non_blocking=True)
                    #         if isinstance(v, torch.Tensor)
                    #         else v
                    #     )
                    #     for k, v in val_batch.items()
                    # }

                    with ctx:
                        _, val_loss = model(
                            val_batch["input_ids"], val_batch["class_label"]
                        )

                    val_losses.append(val_loss.clone().item())

                    break

            val_loss = torch.tensor(np.mean(val_losses)).to(device)
            dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)

            if master_process:
                wandb.log({"val/loss": val_loss.item(), "val/step": step})

            model.train()

        if master_process and step % save_every == 0:
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": step,
                "config": model.module.config,
            }
            os.makedirs(f"logs/ckpts_{run_id}", exist_ok=True)
            ckpt_path = f"logs/ckpts_{run_id}/step_{step}.pt"
            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint, ckpt_path)

    if master_process:
        wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    train()
