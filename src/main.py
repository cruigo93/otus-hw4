import os

import albumentations as albu
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pt
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchmetrics
import wandb

import datasets
import models
import transforms

DATA_PATH = "/hdd/zhuldyzzhan/imagenette2-320/noisy_imagenette.csv"
BATCH_SIZE = 128
NUM_WORKERS = 12
EPOCHS = 10
EXPERIMENT_NAME = "Effnet"
LR = 0.005
AUGMENTATION = "hard"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(model_name):
    if model_name == "Resnet":
        return models.Resnet()
    elif model_name == "VGG":
        return models.VGG()
    elif model_name == "Effnet":
        return models.Effnet()
    else:
        return models.Densenet()
    
def get_transforms(augmentation_style):
    if augmentation_style == "spatial":
        return transforms.get_spatial_transforms()
    elif augmentation_style == "hard":
        return transforms.get_hard_transforms()
    else:
        return transforms.get_light_transforms()
    
TRAIN_TRANSFORMS = get_transforms(AUGMENTATION)
train_transforms_dict = albu.to_dict(TRAIN_TRANSFORMS)

wandb.login(key="SECRET_KEY_HERE", relogin=True)
wandb.init(project=f"{EXPERIMENT_NAME}-{AUGMENTATION}", config=train_transforms_dict)


class ImageneteModel(pt.LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler, dataloaders):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloaders =  dataloaders
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10, top_k=5)
        self.f1score = torchmetrics.F1Score(task="multiclass", num_classes=10, top_k=5)
    
    def train_dataloader(self):
        return self.dataloaders["train"]
    
    def val_dataloader(self):
        return self.dataloaders["val"]
    
    def test_dataloader(self):
        return self.dataloaders["test"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        pred = torch.softmax(self(x), dim=1)
        # print(pred.shape, y)
        loss = self.criterion(pred, y)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        pred = torch.softmax(self(x), dim=1)
        loss = self.criterion(pred, y)
        accuracy = self.accuracy(pred, y)
        f1score = self.f1score(pred, y)
        history = {"val_loss": loss, "accuracy": accuracy, "f1score": f1score}
        self.log_dict(history, on_epoch=True, on_step=False, prog_bar=True)
        return history
        
    def validation_epoch_end(self, outputs):
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.log("avg_val_accuracy", avg_acc)
        
        wandb.log({"avg_val_accuracy": avg_acc})
        
    def test_step(self, batch, batch_nb):
        x, y = batch
        pred = torch.softmax(self(x), dim=1)
        loss = self.criterion(pred, y)
        accuracy = self.accuracy(pred, y)
        f1score = self.f1score(pred, y)
        history = {"test_loss": loss, "test_accuracy": accuracy, "test_f1score": f1score}
        self.log_dict(history, on_epoch=False, on_step=True, prog_bar=True)
        return history
        
    def test_epoch_end(self, outputs):
        avg_acc = torch.stack([x["test_accuracy"] for x in outputs]).mean()
        num_parameters = count_parameters(self.model)
        self.log("avg_test_accuracy", avg_acc)
        
        wandb.log({"avg_test_accuracy": avg_acc, "num_params": num_parameters})
        
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss", avg_loss)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

def main():
    items_df = pd.read_csv(DATA_PATH)
    items = items_df.loc[items_df["is_valid"] == False].to_dict("records")
    train_items, val_items = train_test_split(items, test_size=0.3, random_state=42)
    
    
    train_dataset = datasets.ImagenetDataset(train_items, TRAIN_TRANSFORMS)
    val_dataset = datasets.ImagenetDataset(val_items, transforms.get_valid_transforms())
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    test_items = items_df.loc[items_df["is_valid"] == True].to_dict("records")
    test_dataset = datasets.ImagenetDataset(test_items, transforms.get_valid_transforms())
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    dataLoaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }
    
    model = get_model(EXPERIMENT_NAME)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)

    criterion = nn.CrossEntropyLoss()
    
    logger = TensorBoardLogger("logs/"+EXPERIMENT_NAME, name=EXPERIMENT_NAME)

    learner = ImageneteModel(model, criterion, optimizer, scheduler, dataLoaders)
    # Initialize a trainer
    trainer = pt.Trainer(
        accelerator="gpu",
        max_epochs=EPOCHS,
        precision=16,
        logger=logger,
        num_sanity_val_steps=0
    )

    # Train the model ⚡
    trainer.fit(learner)
    trainer.test(learner)
    
    
    

if __name__ == "__main__":
    main()