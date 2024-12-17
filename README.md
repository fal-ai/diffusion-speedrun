# Nano Diffusion Speedrun

A minimal implementation for tokenizing ImageNet and training GPT/Diffusion models, optimized for performance and simplicity. This codebase provides an efficient path from initialization to image generation.

## Features

- NVIDIA's Cosmos tokenizer for image tokenization
- Implementation of both GPT and Diffusion architectures
- Multi-GPU training optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/fal-ai/diffusion-speedrun
cd diffusion-speedrun

# Execute setup script
bash oneclick_setup.sh
```

## Training Pipeline

### 1. Data Preparation
The repository utilizes pre-tokenized ImageNet data for optimal performance. For custom tokenization:

```bash
cd tokenize_dataset
python download.py download-models     # Acquire Cosmos tokenizer
python download.py download-preprocessed  # Acquire preprocessed ImageNet
```

### 2. Model Training

```bash
# Single GPU Configuration
python train_gpt.py

# Multi-GPU Configuration (Recommended)
torchrun --nproc_per_node=8 train_gpt.py \
    --run_name="experiment_1" \
    --global_batch_size=128 \
    --per_gpu_batch_size=16 \
    --num_iterations=51000 \
    --learning_rate=3e-3
```

### 3. Image Generation

```bash
python generate_gpt.py \
    --checkpoint="path/to/checkpoint.pt" \
    --num_samples=5 \
    --temperature=1.0 \
    --top_k=100 \
    --class_ids="1,130" \
    --output_dir="generated_images"
```

## Acknowledgments

- NVIDIA's [Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer)
- [@karpathy](https://github.com/karpathy) for foundational GPT implementation
- [@kellerjordan](https://github.com/kellerjordan) for nano-gpt speedrun efforts, which I took many ideas from

## License

MIT
