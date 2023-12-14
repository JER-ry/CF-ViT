from functools import partial
from torchinfo import summary
import torch
from torch import nn
from deit.models_deit import CFVisionTransformer


model = CFVisionTransformer(
    patch_size=16,
    embed_dim=384,
    depth=12,
    num_heads=6,
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
).to("cuda")


summary(
    model,
    input_data=[
        [
            torch.zeros(1, 3, 96, 96).to("cuda"),
            torch.zeros(1, 3, 192, 192).to("cuda"),
            torch.zeros(1, 3, 384, 384).to("cuda"),
        ],
    ],
    # depth=0,
)
