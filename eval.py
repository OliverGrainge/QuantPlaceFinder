import time

import pytorch_lightning as pl
import torch
import torch.nn as nn
import yaml

from dataloaders.ImageNet import ImageNet
from dataloaders.GSVCities import GSVCities
from models.helper import get_model
from parsers import get_args_parser
import pytorch_lightning as pl 
import yaml



def measure_latency(model, input_tensor, num_runs=100, warmup_runs=10, verbose=True):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run on a CUDA-enabled device."
        )

    device = torch.device("cuda")
    model.to(device)
    input_tensor = input_tensor.to(device)

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        # Warm-up runs
        for _ in range(warmup_runs):
            _ = model(input_tensor)
            torch.cuda.synchronize()

        # Timing runs
        latencies = []
        for _ in range(num_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            _ = model(input_tensor)
            end_event.record()

            # Wait for the events to be recorded
            torch.cuda.synchronize()

            # Calculate elapsed time in milliseconds
            elapsed_time = start_event.elapsed_time(end_event)
            latencies.append(elapsed_time)

    average_latency = sum(latencies) / len(latencies)
    if verbose:
        print(" ")
        print(" Average Latency: ", average_latency)
        print(" ")
    return average_latency


def measure_memory(model, verbose=True):
    state_dict = model.state_dict()
    total_bytes = 0

    for key, param in state_dict.items():
        # Ensure the parameter is a tensor
        if isinstance(param, torch.Tensor):
            param_size = param.numel() * param.element_size()
            total_bytes += param_size
        else:
            raise TypeError(
                f"Expected torch.Tensor for key '{key}', but got {type(param)}"
            )

    # Convert bytes to megabytes
    total_megabytes = total_bytes / (1024**2)
    if verbose:
        print(" ")
        print(" Model Size: ", total_megabytes)
        print(" ")
    return total_megabytes


def eval_imagenet(args):
    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
        out_dim=1000,
        normalize_output=False,
    )
    model.eval()
    module = ImageNet.load_from_checkpoint(
        checkpoint_path=args.load_checkpoint, model=model
    )

    measure_latency(model, torch.randn(1, 3, 224, 224))
    measure_memory(model)
    trainer = pl.Trainer(limit_test_batches=20)
    trainer.test(module)


def eval_vpr(args): 

    model = get_model(
        args.image_size,
        args.backbone_arch,
        args.agg_arch,
        preset=args.preset,
        out_dim=1000,
        normalize_output=False,
    )

    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)



    model_module = GSVCities(
        config["Training"]["GSVCities"], 
        model=model,
        batch_size=args.batch_size, 
        shuffle_all=False, 
        image_size=args.image_size,
        num_workers=args.num_workers,
        val_set_names=["pitts30k_val"],
        search_precision="float32",
    )

    trainer = pl.Trainer(
        enable_progress_bar=True, 
        strategy="auto", 
        accelerator="auto"
    )

    trainer.validate(model_module)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if "imagenet" in args.training_method.lower():
        eval_imagenet(args)
    elif "gsvcities" in args.training_method.lower():
        eval_vpr(args)
    else: 
        raise Exception
