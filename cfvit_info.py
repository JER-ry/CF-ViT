from torchinfo import summary
import torch
from deit.models_deit import CFVisionTransformer


device = "cpu"

model = CFVisionTransformer(drop_rate=0.0, drop_path_rate=0.1).to(device)
model.informative_selection = True
model.use_early_exit = True
model.thresholds=[1.0, 0.0]

summary(
    model,
    input_data=[
        [
            torch.zeros(1, 3, 96, 96).to(device),
            torch.zeros(1, 3, 192, 192).to(device),
            torch.zeros(1, 3, 384, 384).to(device),
        ],
    ],
    depth=0,
)
