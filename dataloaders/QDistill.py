import torch
import torch.nn.functional as F
from prettytable import PrettyTable
import pytorch_lightning as pl 
from pytorch_metric_learning import miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torchvision import transforms as T
from pytorch_metric_learning import losses

import utils
from dataloaders.train.GSVCitiesDataset import GSVCitiesDataset

from models.helper import get_model


def sinkhorn_knopp(A, num_iters=100, tol=1e-5):
    """
    Applies the Sinkhorn-Knopp algorithm to transform A into a doubly stochastic matrix.

    Args:
        A (torch.Tensor): Input non-negative matrix of shape (n, n).
        num_iters (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        torch.Tensor: Doubly stochastic matrix.
        bool: Convergence flag.
    """
    # Ensure A is non-negative
    assert torch.all(A >= 0), "All elements of A must be non-negative."

    # Initialize
    D = A.clone()
    n, m = D.shape
    if n != m:
        raise ValueError("Input matrix must be square.")

    for i in range(num_iters):
        # Row normalization
        row_sum = D.sum(dim=1, keepdim=True)
        # Avoid division by zero
        row_sum = torch.where(row_sum == 0, torch.tensor(1.0, device=D.device), row_sum)
        D = D / row_sum

        # Column normalization
        col_sum = D.sum(dim=0, keepdim=True)
        # Avoid division by zero
        col_sum = torch.where(col_sum == 0, torch.tensor(1.0, device=D.device), col_sum)
        D = D / col_sum

        # Check for convergence
        row_diff = torch.abs(D.sum(dim=1) - 1.0).max()
        col_diff = torch.abs(D.sum(dim=0) - 1.0).max()
        if row_diff.item() < tol and col_diff.item() < tol:
            print(f"Converged in {i+1} iterations.")
            return D, True

    print(f"Did not converge within {num_iters} iterations.")
    return D, False



IMAGENET_MEAN_STD = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

class QVPRDistill(pl.LightningModule):
    def __init__(
        self,
        config,
        teacher_arch="EigenPlaces",
        student_backbone_arch="vit_small",
        student_agg_arch="cls",
        student_out_dim=1024,
        batch_size=32,
        image_size=(224, 224),
        num_workers=4,
        val_set_names=["pitts30k_val", "msls_val"],
        margin=0.1,
        max_epochs=10,
        loss_type="rel"
    ):

        super().__init__()
        self.batch_size = (batch_size,)
        self.image_size = (image_size,)
        self.num_workers = num_workers
        self.val_set_names = val_set_names
        self.max_epochs = max_epochs
        self.loss_type = loss_type
    

        self.lr = config["lr"]
        self.optimizer_type = config["optimizer"]
        self.weight_decay = config["weight_decay"]
        self.momentum = config["momentum"]
        self.warmup_steps = config["warmup_steps"]
        self.milestones = config["milestones"]
        self.lr_mult = config["lr_mult"]
        self.miner_margin = config["miner_margin"]
        self.faiss_gpu = config["faiss_gpu"]
        self.img_per_place = config["img_per_place"]
        self.min_img_per_place = config["min_img_per_place"]
        self.cities = config["cities"]
        self.shuffle_all = config["shuffle_all"]


        # Data parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_dataset = IMAGENET_MEAN_STD["mean"]
        self.std_dataset = IMAGENET_MEAN_STD["std"]
        self.val_set_names = val_set_names
        self.random_sample_from_each_place = True
        self.train_dataset = None
        self.val_datasets = []
        self.show_data_stats = True
        self.warmup_epochs=1

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
            "shuffle": self.shuffle_all,
        }

        self.valid_loader_config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers // 2,
            "drop_last": False,
            "pin_memory": False,
            "shuffle": False,
        }

        self.miner = miners.MultiSimilarityMiner(
            epsilon=margin, distance=CosineSimilarity()
        )
        self.task_loss_fn = losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            )
        self.teacher = get_model(preset=teacher_arch)
        self.student = get_model(
            backbone_arch=student_backbone_arch,
            agg_arch=student_agg_arch,
            normalize_output=True,
            out_dim=student_out_dim,
        )

        self.freeze_model(self.teacher)

    def freeze_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.student(x)

    @staticmethod
    def cosine_sim_vec(a, b):
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
        return torch.diag(a @ b.t())
    
    @staticmethod 
    def cosine_sim_mat(a, b):
        a = F.normalize(a, dim=1, p=2)
        b = F.normalize(b, dim=1, p=2)
        return a @ b.t()
    

    def multisim_weights_vec(self, teacher_desc, student_desc, labels, alpha=1.0, beta=50.0, base=0.0, eps=0.1):
        miner_outputs = self.miner(student_desc, labels)
        a1, pos, a2, neg = miner_outputs 
        #print("Pairs", a1.shape, pos.shape, a2.shape, neg.shape)
        S = teacher_desc @ teacher_desc.t()
        #S = (S + 1) / 2  # Apply scaling
        #print("Similarity matrix", S.min(), S.max())
        S_pos = S.clone()
        S_neg = S.clone()
        pos_mask = torch.zeros_like(S).type(torch.bool)
        neg_mask = torch.zeros_like(S).type(torch.bool)
        pos_mask[a1, pos] = True 
        neg_mask[a2, neg] = True 

        w = torch.zeros_like(S) + eps
        #print("Weight Matrix", w.min(), w.max())
        S_neg[~neg_mask] = float('-inf')
        w[a2, neg] = torch.exp(beta * (S_neg[a2, neg] - base)) / (1 + torch.sum(torch.exp(beta * (S_neg[a2] - base)), dim=1))

        S_pos[~pos_mask] = float('-inf')
        w[a1, pos] = 1 / (torch.exp(-alpha * (base - S_pos[a1, pos])) + torch.sum(torch.exp(-alpha * (S_pos[a1] - S_pos[a1, pos][:, None])), dim=1))
        #print("Weight Matrix", w.min(), w.max())
        #w = w / w.max()
        #import matplotlib.pyplot as plt 
        #plt.imshow(w.float().detach().cpu())
        #plt.show()
        #w, _ = sinkhorn_knopp(w)
        #import matplotlib.pyplot as plt 
        #plt.imshow(w.float().detach().cpu())
        #plt.show()
        return w


    def loss_fn_rel(self, teacher_desc, student_desc, labels):
        teacher_rel = self.cosine_sim_mat(teacher_desc, teacher_desc)
        student_rel = self.cosine_sim_mat(student_desc, student_desc)
        loss = (teacher_rel - student_rel)**2 
        loss = loss.sum() / student_rel.shape[0]
        self.log("loss", loss)
        return loss
    
    def loss_fn_rel_weighted(self, teacher_desc, student_desc, labels): 
        weights = self.multisim_weights_vec(teacher_desc, student_desc, labels)
        teacher_rel = self.cosine_sim_mat(teacher_desc, teacher_desc) 
        student_rel = self.cosine_sim_mat(student_desc, student_desc) 
        loss = (teacher_rel - student_rel)**2 * weights
        loss = loss.sum()
        self.log("loss", loss)
        return loss
    
    def loss_fn_rel_weighted_task_loss(self, teacher_desc, student_desc, labels):
        weights = self.multisim_weights_vec(teacher_desc, student_desc, labels)
        teacher_rel = self.cosine_sim_mat(teacher_desc, teacher_desc) 
        student_rel = self.cosine_sim_mat(student_desc, student_desc) 
        loss_rel = (teacher_rel - student_rel)**2 * weights
        loss_rel = loss_rel.sum()
        miner_outputs = self.miner(student_desc, labels)
        loss_task = self.task_loss_fn(student_desc, labels, miner_outputs)
        return loss_task + loss_rel
    
    def loss_fn_rel_task_loss(self, teacher_desc, student_desc, labels):
        teacher_rel = self.cosine_sim_mat(teacher_desc, teacher_desc) 
        student_rel = self.cosine_sim_mat(student_desc, student_desc) 
        loss_rel = (teacher_rel - student_rel)**2
        loss_rel = loss_rel.sum() / teacher_rel.shape[0]
        miner_outputs = self.miner(student_desc, labels)
        loss_task = self.task_loss_fn(student_desc, labels, miner_outputs)
        #print(0.05 * loss_rel.item(), loss_task.item())
        return loss_task + (0.05 * loss_rel)
    
    def loss_fn_rel_mse_task_loss(self, teacher_desc, student_desc, labels):
        teacher_rel = self.cosine_sim_mat(teacher_desc, teacher_desc) 
        student_rel = self.cosine_sim_mat(student_desc, student_desc) 
        loss_rel = (teacher_rel - student_rel)**2
        loss_rel = loss_rel.sum() / teacher_rel.shape[0]
        S = teacher_desc @ student_desc.t()
        D = 1 - ((S+1)/2)
        D_squared = D**2
        loss_mse = torch.trace(D_squared)/teacher_desc.shape[0]
        miner_outputs = self.miner(student_desc, labels)
        loss_task = self.task_loss_fn(student_desc, labels, miner_outputs)
        #print(loss_mse, 0.05 * loss_rel.item(), loss_task.item())
        return loss_task + (0.05 * loss_rel)
    

    
    def loss_fn_mse_task_loss(self, teacher_desc, student_desc, labels):
        S = teacher_desc @ student_desc.t()
        D = 1 - ((S + 1)/2)
        D_squared = D**2 
        loss_mse = torch.trace(D_squared) / teacher_desc.shape[0]
        miner_outputs = self.miner(student_desc, labels)
        loss_task = self.task_loss_fn(student_desc, labels, miner_outputs)
        return loss_mse
    
    def loss_fn_mse(self, teacher_desc, student_desc, labels):
        S = teacher_desc @ student_desc.t()
        D = 1 - ((S + 1)/2)
        loss = torch.trace(D**2)/teacher_desc.shape[0]
        return loss
        
    def task_loss_function(self, descriptors, labels):
        miner_outputs = self.miner(descriptors, labels)
        loss = self.task_loss_fn(descriptors, labels, miner_outputs)
        self.log("task_loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        student_desc = F.normalize(self.student(images), dim=1, p=2)
        if self.loss_type != "task":
            teacher_desc = F.normalize(self.teacher(images), dim=1, p=2)
            
        if self.loss_type == "rel":
            loss = self.loss_fn_rel(teacher_desc, student_desc, labels)
        if self.loss_type == "task":
            loss = self.task_loss_function(student_desc, labels)
        if self.loss_type == "weighted_rel":
            loss = self.loss_fn_rel_weighted(teacher_desc, student_desc, labels)
        if self.loss_type == "weighted_rel_task":
            loss = self.loss_fn_rel_weighted_task_loss(teacher_desc, student_desc, labels)
        if self.loss_type == "rel_task":
            loss = self.loss_fn_rel_task_loss(teacher_desc, student_desc, labels)
        if self.loss_type == "rel_weighted_rel": 
            loss = self.loss_fn_rel_weighted_rel(teacher_desc, student_desc, labels)
        if self.loss_type == "mse": 
            loss = self.loss_fn_mse(teacher_desc, student_desc, labels)
        if self.loss_type == "task_mse": 
            loss = self.loss_fn_mse_task_loss(teacher_desc, student_desc, labels)
        if self.loss_type == "task_mse_rel": 
            loss = self.loss_fn_rel_mse_task_loss(teacher_desc, student_desc, labels)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.student.parameters(), lr=0.0002, weight_decay=0.0
        )

        # Compute total steps
        steps_per_epoch = len(self.train_dataloader())
        total_steps = steps_per_epoch * self.max_epochs
        warmup_steps = steps_per_epoch * self.warmup_epochs
        # Scheduler

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer]#, [scheduler]

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
                else:
                    raise NotImplementedError(
                        f"Validation set {val_set_name} not implemented"
                    )
            if self.show_data_stats:
                self.print_stats()

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

    def on_validation_epoch_start(self):
        # Initialize or reset the list to store validation outputs
        self.validation_outputs = []

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        # store the outputs
        self.validation_outputs.append(descriptors.detach().cpu())
        return descriptors.detach().cpu()

    def on_validation_epoch_end(self):
        """Process the validation outputs stored in self.validation_outputs."""

        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list
        if len(self.val_datasets) == 1:
            val_step_outputs = [self.validation_outputs]
        else:
            val_step_outputs = self.validation_outputs

        for i, (val_set_name, val_dataset) in enumerate(
            zip(self.val_set_names, self.val_datasets)
        ):
            feats = torch.concat(val_step_outputs[i], dim=0)

            num_references = val_dataset.num_references
            num_queries = val_dataset.num_queries
            ground_truth = val_dataset.ground_truth

            # split to ref and queries
            r_list = feats[:num_references]
            q_list = feats[num_references:]

            recalls_dict_float, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision="float32",
            )

            self.log(
                f"{val_set_name}/float_R1",
                recalls_dict_float[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/float_R5",
                recalls_dict_float[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/float_R10",
                recalls_dict_float[10],
                prog_bar=False,
                logger=True,
            )

            recalls_dict_binary, predictions = utils.get_validation_recalls(
                r_list=r_list,
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 25],
                gt=ground_truth,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu,
                precision="binary",
            )

            self.log(
                f"{val_set_name}/binary_R1",
                recalls_dict_binary[1],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/binary_R5",
                recalls_dict_binary[5],
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"{val_set_name}/binary_R10",
                recalls_dict_binary[10],
                prog_bar=False,
                logger=True,
            )

            del r_list, q_list, feats, num_references, ground_truth

        # Clear the outputs after processing
        self.validation_outputs.clear()
        print("\n\n")

    def print_stats(self):
        table = PrettyTable()
        table.field_names = ["Data", "Value"]
        table.add_row(["# of cities", f"{len(self.cities)}"])
        table.add_row(["# of places", f"{self.train_dataset.__len__()}"])
        table.add_row(["# of images", f"{self.train_dataset.total_nb_images}"])
        print(table.get_string(title="Training Dataset"))

        table = PrettyTable()
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))

        table = PrettyTable()
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__() // self.batch_size}"]
        )
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
