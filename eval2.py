import pickle
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from deit.datasets import build_dataset
import deit.models_deit
from timm.models import create_model
from timm.utils import accuracy
import utils


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = dotdict(
    {
        "batch_size": 250,
        "coarse_stage_size": 6,
        "drop": 0.0,
        "drop_path": 0.1,
        "data_path": "ImageNet",
        "data_set": "IMNET",
        "device": "cuda:1",
        "seed": 0,
        "start_epoch": 0,
        "num_workers": 10,
        "pin_mem": False,
        "input_size_list": [96, 192, 384],
    }
)


args.input_size_list = [
    16 * args.coarse_stage_size,
    16 * 2 * args.coarse_stage_size,
    16 * 4 * args.coarse_stage_size,
]
args.input_size = max(args.input_size_list)

device = torch.device(args.device)

seed = args.seed + utils.get_rank()
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True

dataset_val, args.nb_classes = build_dataset(is_train=False, args=args)
sampler_val = torch.utils.data.RandomSampler(dataset_val)
data_loader_val = torch.utils.data.DataLoader(
    dataset_val,
    sampler=sampler_val,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=False,
)

model = create_model(
    "cf_deit_small",
    pretrained=False,
    num_classes=args.nb_classes,
    img_size_list=args.input_size_list,
    drop_rate=args.drop,
    drop_path_rate=args.drop_path,
    drop_block_rate=None,
)
model.load_state_dict(
    torch.load("CF-ViT/log/run8/model_best.pth", map_location=device)["model"]
)
model.informative_selection = False
model.use_early_exit = True
model.to(device)
model.eval()

with torch.no_grad():
    result_dict = {}
    thresholds_list = [
        (i, j)
        for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for j in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        if i >= j
    ]
    # thresholds_list = [(0.6, 0.5)]
    
    for i, j in thresholds_list:
        model.thresholds = [i, j]
        model.image_count_per_level = [0] * len(args.input_size_list)
        print(i, j)

        metric_logger = utils.MetricLogger(delimiter="  ")

        for images, target in metric_logger.log_every(data_loader_val, -1):
            target = target.to(device)
            images = images.to(device)
            images_list = []
            for i in range(0, len(args.input_size_list) - 1):
                resized_img = F.interpolate(
                    images,
                    (args.input_size_list[i], args.input_size_list[i]),
                    mode="bilinear",
                    align_corners=True,
                )
                resized_img = torch.squeeze(resized_img)
                images_list.append(resized_img)
            images_list.append(images)
            with torch.cuda.amp.autocast():
                results = model(images_list)
            batch_size = images.shape[0]

            acc1, acc5 = accuracy(results, target, topk=(1, 5))
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

        result_dict[(i, j)] = [
            metric_logger.meters["acc1"].global_avg,
            metric_logger.meters["acc5"].global_avg,
            [int(i) for i in model.image_count_per_level],
        ]
        print(result_dict[(i, j)])
