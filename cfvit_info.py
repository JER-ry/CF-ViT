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
            torch.zeros(512, 3, 112, 112).to("cuda"),
            torch.zeros(512, 3, 224, 224).to("cuda"),
        ],
    ],
    depth=0,
)

# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# CFVisionTransformer                      [512, 1000]               95,232
# ├─MultiResoPatchEmbed: 1-1               [512, 49, 384]            --
# │    └─Conv2d: 2-1                       [512, 384, 7, 7]          295,296
# ├─Dropout: 1-2                           [512, 50, 384]            --
# ├─ModuleList: 1-9                        --                        (recursive)
# │    └─Block: 2-2                        [512, 50, 384]            --
# │    │    └─LayerNorm: 3-1               [512, 50, 384]            768
# │    │    └─Attention: 3-2               [512, 50, 384]            591,360
# │    │    └─Identity: 3-3                [512, 50, 384]            --
# │    │    └─LayerNorm: 3-4               [512, 50, 384]            768
# │    │    └─Mlp: 3-5                     [512, 50, 384]            1,181,568
# │    │    └─Identity: 3-6                [512, 50, 384]            --
# │    └─Block: 2-3                        [512, 50, 384]            1,774,464
# │    └─Block: 2-4                        [512, 50, 384]            1,774,464
# │    └─Block: 2-5                        [512, 50, 384]            1,774,464
# │    └─Block: 2-6                        [512, 50, 384]            1,774,464
# │    └─Block: 2-7                        [512, 50, 384]            1,774,464
# │    └─Block: 2-8                        [512, 50, 384]            1,774,464
# │    └─Block: 2-9                        [512, 50, 384]            1,774,464
# │    └─Block: 2-10                       [512, 50, 384]            1,774,464
# │    └─Block: 2-11                       [512, 50, 384]            1,774,464
# │    └─Block: 2-12                       [512, 50, 384]            1,774,464
# │    └─Block: 2-13                       [512, 50, 384]            1,774,464
# ├─LayerNorm: 1-4                         [512, 50, 384]            768
# ├─Linear: 1-5                            [512, 1000]               385,000
# ├─MultiResoPatchEmbed: 1-6               [512, 196, 384]           (recursive)
# │    └─Conv2d: 2-14                      [512, 384, 14, 14]        (recursive)
# ├─Sequential: 1-7                        [512, 49, 384]            --
# │    └─LayerNorm: 2-15                   [512, 49, 384]            768
# │    └─Mlp: 2-16                         [512, 49, 384]            1,181,568
# ├─Dropout: 1-8                           [512, 197, 384]           --
# ├─ModuleList: 1-9                        --                        (recursive)
# │    └─Block: 2-17                       [512, 197, 384]           (recursive)
# │    └─Block: 2-18                       [512, 197, 384]           (recursive)
# │    └─Block: 2-19                       [512, 197, 384]           (recursive)
# │    └─Block: 2-20                       [512, 197, 384]           (recursive)
# │    └─Block: 2-21                       [512, 197, 384]           (recursive)
# │    └─Block: 2-22                       [512, 197, 384]           (recursive)
# │    └─Block: 2-23                       [512, 197, 384]           (recursive)
# │    └─Block: 2-24                       [512, 197, 384]           (recursive)
# │    └─Block: 2-25                       [512, 197, 384]           (recursive)
# │    └─Block: 2-26                       [512, 197, 384]           (recursive)
# │    └─Block: 2-27                       [512, 197, 384]           (recursive)
# │    └─Block: 2-28                       [512, 197, 384]           (recursive)
# ├─LayerNorm: 1-10                        [512, 197, 384]           (recursive)
# ├─Linear: 1-11                           [512, 1000]               (recursive)
# ==========================================================================================
# Total params: 23,252,200
# Trainable params: 23,252,200
# Non-trainable params: 0
# Total mult-adds (Units.GIGABYTES): 59.85
# ==========================================================================================
# Input size (MB): 385.35
# Forward/backward pass size (MB): 52526.12
# Params size (MB): 92.63
# Estimated Total Size (MB): 53004.10
# ==========================================================================================
