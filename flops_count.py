import torch
import deit.models_deit
from timm.models import create_model
from fvcore.nn import FlopCountAnalysis


device = "cuda"

model = create_model(
    "cf_deit_small",
    pretrained=False,
    num_classes=1000,
    img_size_list=[96, 192, 384],
    drop_rate=0.0,
    drop_path_rate=0.1,
    drop_block_rate=None,
)
model.to(device)
model.use_early_exit = True


def get_example_input(img_size_list):
    return [torch.zeros(1, 3, i, i, device=device) for i in img_size_list]


model.informative_selection = True

model.thresholds = [0.0, 0.0]
print(FlopCountAnalysis(model, get_example_input([96, 192, 384])).total())
# 811039104

model.thresholds = [1.0, 0.0]
print(FlopCountAnalysis(model, get_example_input([96, 192, 384])).total())
# 2909431296

model.thresholds = [1.0, 1.0]
print(FlopCountAnalysis(model, get_example_input([96, 192, 384])).total())
# 8426119680


model.informative_selection = False

model.thresholds = [0.0, 0.0]
print(FlopCountAnalysis(model, get_example_input([96, 192, 384])).total())
# 811039104

model.thresholds = [1.0, 0.0]
print(FlopCountAnalysis(model, get_example_input([96, 192, 384])).total())
# 4176089856

model.thresholds = [1.0, 1.0]
print(FlopCountAnalysis(model, get_example_input([96, 192, 384])).total())
# 16371843840
