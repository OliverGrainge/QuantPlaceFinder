from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
from prettytable import PrettyTable
from torch.optim import lr_scheduler
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from transformers import get_cosine_schedule_with_warmup

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset
from matching.global_cosine_sim import global_cosine_sim
from matching.global_hamming_sim import global_hamming_sim

IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
VIT_MEAN_STD = {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}


def symmetric_binarize(x):
    return torch.where(x > 0.0, torch.ones_like(x), -torch.ones_like(x))


class QuART(pl.LightningModule):
    def __init__(
        self,
        model,
        batch_size=32,
        image_size=(480, 640),
        num_workers=4,
        show_data_stats=True,
        mean_std=IMAGENET_MEAN_STD,
        val_set_names=["pitts30k_val", "msls_val"],
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        matching_functions=[global_cosine_sim, global_hamming_sim],
        miner_margin=0.1,
        cities=["London", "Melbourne", "Boston"],
        lr=0.0001,
        img_per_place=4,
        min_img_per_place=4,
    ):
        super().__init__()
        # Model parameters
        self.lr = lr
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.cities = cities
        self.matching_functions = matching_functions
        self.batch_acc = []
        self.loss_fn = utils.get_loss(self.loss_name)
        self.miner = utils.get_miner(self.miner_name, self.miner_margin)

        self.model = model

        # Data parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_dataset = mean_std["mean"]
        self.std_dataset = mean_std["std"]
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []
        self.show_data_stats = show_data_stats

        # Train and valid transforms
        self.train_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        self.valid_transform = T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
            ]
        )

        # Dataloader configs
        self.train_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": True,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": False,
        }

    def setup(self, stage=None):
        # Setup for 'fit' or 'validate'self
        if stage == "fit" or stage == "validate":
            self.reload()
            self.val_datasets = []
            for val_set_name in self.val_set_names:
                if "pitts30k" in val_set_name.lower():
                    from dataloaders.val.PittsburghDataset import \
                        PittsburghDataset

                    self.val_datasets.append(
                        PittsburghDataset(
                            which_ds=val_set_name, input_transform=self.valid_transform
                        )
                    )
                elif val_set_name.lower() == "msls_val":
                    from dataloaders.val.MapillaryDataset import MSLS

                    self.val_datasets.append(MSLS(input_transform=self.valid_transform))
                elif val_set_name.lower() == "nordland":
                    from dataloaders.val.NordlandDataset import NordlandDataset

                    self.val_datasets.append(
                        NordlandDataset(input_transform=self.valid_transform)
                    )
                elif val_set_name.lower() == "sped":
                    from dataloaders.val.SPEDDataset import SPEDDataset

                    self.val_datasets.append(
                        SPEDDataset(input_transform=self.valid_transform)
                    )
                elif (
                    "sf_xl" in val_set_name.lower()
                    and "val" in val_set_name.lower()
                    and "small" in val_set_name.lower()
                ):
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(
                            which_ds="sf_xl_small_val", input_transform=self.transform
                        )
                    )
                elif (
                    "sf_xl" in val_set_name.lower()
                    and "test" in val_set_name.lower()
                    and "small" in val_set_name.lower()
                ):
                    from dataloaders.val.SF_XL import SF_XL

                    self.val_datasets.append(
                        SF_XL(
                            which_ds="sf_xl_small_test", input_transform=self.transform
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(
                DataLoader(dataset=val_dataset, **self.valid_loader_config)
            )
        return val_dataloaders

    def forward(self, x):
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
        warmup_steps = 0
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def loss_function(self, descriptors, labels, binary_mine=False):
        if self.miner is not None:
            if binary_mine:
                miner_desc = torch.nn.functional.normalize(
                    symmetric_binarize(descriptors), p=2, dim=1
                )
            else:
                miner_desc = descriptors
            miner_outputs = self.miner(miner_desc, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log(
            "b_acc",
            sum(self.batch_acc) / len(self.batch_acc),
            prog_bar=True,
            logger=True,
        )
        return loss

    def reg_function(self, descriptors):
        desc = descriptors["global_desc"]
        qdesc = symmetric_binarize(desc)
        # Cosine similarity ranges from -1 to 1 since both desc and qdesc are normalized
        cos_sim = torch.nn.functional.cosine_similarity(
            desc, qdesc, dim=1
        )  # Range: [-1, 1]
        cos_dist = 1 - cos_sim
        reg_loss = cos_dist.mean()
        return reg_loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        descriptors = self(images)

        # Main loss
        main_loss = self.loss_function(
            descriptors["global_desc"], labels, binary_mine=False
        )

        # Regularization loss with sine scheduled weight
        reg_loss = self.reg_function(descriptors)

        # Calculate sine schedule weight between 0 and 0.2
        total_steps = self.trainer.estimated_stepping_batches
        current_step = self.global_step

        reg_weight = 0.5 * (
            (
                1
                + torch.sin(
                    torch.tensor((current_step - total_steps / 2) * np.pi / total_steps)
                )
            )
            / 2
        )

        # Combined loss with scheduled reg weight
        loss = main_loss + reg_weight * reg_loss
        # Log losses and weight
        self.log("main_loss", main_loss)
        self.log("reg_loss", reg_loss)
        self.log("reg_weight", reg_weight)
        self.log("loss", loss)

        return {"loss": loss}

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = {}
        for name in self.val_set_names:
            self.validation_outputs[name] = defaultdict(list)

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        for key, value in descriptors.items():
            self.validation_outputs[self.val_set_names[dataloader_idx]][key].append(
                value.detach().cpu()
            )
        return descriptors["global_desc"].detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs_global."""

        full_recalls_dict = {}
        for val_set_name, val_dataset in zip(self.val_set_names, self.val_datasets):
            set_outputs = self.validation_outputs[val_set_name]
            for key, value in set_outputs.items():
                set_outputs[key] = torch.concat(value, dim=0)

            for matching_function in self.matching_functions:
                recalls_dict, _, search_time = matching_function(
                    **set_outputs,
                    num_references=val_dataset.num_references,
                    ground_truth=val_dataset.ground_truth,
                )

                for k, v in recalls_dict.items():
                    full_recalls_dict[
                        f"matching_function[{matching_function.__name__}]_{val_set_name}_{k}"
                    ] = v
                full_recalls_dict[
                    f"matching_function[{matching_function.__name__}]_{val_set_name}_search_time"
                ] = search_time
        self.log_dict(
            full_recalls_dict,
            logger=True,
        )
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        for metric, value in full_recalls_dict.items():
            table.add_row([metric, f"{value:.4f}"])

        print(f"\nResults for {val_set_name}:")
        print(table)
        return full_recalls_dict

    def state_dict(self):
        # Override the state_dict method to return only the student model's state dict
        return self.model.state_dict()
