from dataloaders.VPREval import VPREval
from dataloaders.utils.transforms import get_augmentation
from config import DataConfig, ModelConfig, EvalConfig 
from models.helper import get_model
from models.transforms import get_transform
import torch 
import argparse



def _load_model_and_transform(args): 
    if args.preset is not None: 
        model = get_model(preset=args.preset) 
        transform = get_transform(preset=args.preset)
    else: 
        model = get_model(backbone_arch=args.backbone_arch, agg_arch=args.agg_arch)
        transform = get_transform(transform_level="None", image_size=args.image_size)
    if args.weights_path is not None: 
        model.load_state_dict(torch.load(args.weights_path, map_location="cpu"))

    for param in model.parameters(): 
        param.requires_grad = False 

    return model, transform


def eval(args): 
    model, transform = _load_model_and_transform(args)
    module = VPREval(
        model=model, 
        transform=transform, 
        val_set_names=args.val_set_names, 
        val_dataset_dir=args.val_dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        matching_function=args.matching_function
        )
    
    module.validate()



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    for config in [DataConfig, ModelConfig, EvalConfig]: 
        parser = config.add_argparse_args(parser)
    args = parser.parse_args()

    eval(args)
