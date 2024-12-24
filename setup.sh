mkdir inet
mkdir cosmos_ckpt
wget https://huggingface.co/nvidia/Cosmos-Tokenizer-CI8x8/resolve/main/decoder.jit -O cosmos_ckpt/decoder.jit
wget https://huggingface.co/datasets/fal/cosmos-imagenet/resolve/main/imagenet_ci8x8.safetensors -O inet/imagenet_ci8x8.safetensors
wget https://huggingface.co/datasets/fal/cosmos-imagenet/resolve/main/imagenet_ci8x8_val.safetensors -O inet/imagenet_ci8x8_val.safetensors
wget https://huggingface.co/datasets/ramimmo/dinov2.inet/resolve/main/dinov2_inet_feats.pt -O inet/dinov2_inet_feats.pt