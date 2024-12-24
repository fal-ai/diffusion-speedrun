import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
import click

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class SiTBlock(nn.Module):
    """
    A single Transformer block that optionally accepts a skip tensor.
    If skip=True, we learn a linear projection over the concatenation of the current features x and skip.
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        skip=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_size, eps=1e-6)
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.norm2 = norm_layer(hidden_size, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=0.0,
        )

        # For injecting time or label embeddings (AdaLayerNorm style)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        # Skip connection logic
        self.skip_linear = nn.Linear(2 * hidden_size, hidden_size) if skip else None
        self.use_checkpoint = use_checkpoint

    def forward(self, x, c, skip=None):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, c, skip)
        else:
            return self._forward(x, c, skip)

    def _forward(self, x, c, skip=None):
        # If skip_linear exists, we do "concat + linear" just like the paper
        if self.skip_linear is not None and skip is not None:
            x = self.skip_linear(torch.cat([x, skip], dim=-1))

        # AdaLayerNorm modulations from c
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # --- Attention path ---
        x_attn_normed = modulate(self.norm1(x), shift_msa, scale_msa)
        x_attn = self.attn(x_attn_normed)
        x = x + gate_msa.unsqueeze(1) * x_attn

        # --- MLP path ---
        x_mlp_normed = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x_mlp = self.mlp(x_mlp_normed)
        x = x + gate_mlp.unsqueeze(1) * x_mlp

        return x
    

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class SiT(nn.Module):
    """
    A UViT-like refactor of your SiT model:
      - Split 'depth' into in-blocks, a single mid-block, and out-blocks
      - Skip-connections from in-block outputs to out-blocks
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels

        # Number of 'in blocks' and 'out blocks' (like U-Net's encoder/decoder)
        # We'll reserve 1 block for the 'mid_block' in the center.
        in_depth = depth // 2
        out_depth = depth - in_depth - 1

        # Patch embedding
        self.x_embedder = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=hidden_size
        )
        # Timestep + label
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        # Precompute positional embeddings
        self.num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_size),
            requires_grad=False
        )

        # In-blocks (encoder)
        self.in_blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=False)
            for _ in range(in_depth)
        ])
        # Mid-block
        self.mid_block = SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=False)
        # Out-blocks (decoder), each with skip=True
        self.out_blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, skip=True)
            for _ in range(out_depth)
        ])

        # Final prediction layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # Initialize
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize label embedding table
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize TimestepEmbedder MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in all blocks
        for block in list(self.in_blocks) + [self.mid_block] + list(self.out_blocks):
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.view(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # N, c, h, p, w, p
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def forward(self, x, t, y):
        # 1. Patch Embedding + Pos Embedding
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D)
        # 2. Timestep Embedding
        t = self.t_embedder(t)                   # (N, D)
        # 3. Label Embedding (with optional dropout for classifier-free guidance)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # Combined condition embedding

        # ============ Encoder (in_blocks) ============
        skips = []
        for blk in self.in_blocks:
            x = blk(x, c)  # no skip yet
            skips.append(x)

        # ============ Mid-block ============
        x = self.mid_block(x, c)

        # ============ Decoder (out_blocks) ============
        # pop from the 'skips' list to feed into out-blocks
        for blk in self.out_blocks:
            skip_x = skips.pop()  # last in-block output
            x = blk(x, c, skip=skip_x)

        # ============ Final Prediction + Unpatchify ============
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Classifier-free guidance pass. Similar to your original logic.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb = np.concatenate([get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]),  get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) ], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = 1. / (10000 ** ((np.arange(embed_dim // 2, dtype=np.float64)) / (embed_dim / 2.)))
    out = np.einsum('m,d->md', pos.reshape(-1), omega)
    emb = np.concatenate([np.sin(out), np.cos(out) ], axis=1)  
    return emb

#################################################################################
#                                   SiT Configs                                  #
#################################################################################

SiT_models = {
    'SiT-XL/2': lambda **kwargs:SiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs) ,  'SiT-XL/4': lambda **kwargs:SiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs) ,  'SiT-XL/8': lambda **kwargs:SiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs) ,
    'SiT-L/2':  lambda **kwargs:SiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs) ,   'SiT-L/4':  lambda **kwargs:SiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs) ,   'SiT-L/8':  lambda **kwargs:SiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs) ,
    'SiT-B/2':  lambda **kwargs:SiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs) ,   'SiT-B/4':  lambda **kwargs:SiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs) ,   'SiT-B/8':  lambda **kwargs:SiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs) ,
    'SiT-S/2':  lambda **kwargs:SiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs) ,   'SiT-S/4':  lambda **kwargs:SiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs) ,   'SiT-S/8':  lambda **kwargs:SiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs) , 
    'SiT-T/8':  lambda **kwargs:SiT(depth=6, hidden_size=192, patch_size=8, num_heads=6, **kwargs)
}
