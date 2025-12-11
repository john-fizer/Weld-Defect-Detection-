"""
PyTorch Lightning Training Module
Handles complete training with Ranger21, progressive resizing, MixUp/CutMix, TTA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import torchmetrics
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from loguru import logger

from .model import WeldClassifier
from .arcface import ArcFaceHead, CosFaceHead

try:
    from ranger21 import Ranger21
    RANGER_AVAILABLE = True
except ImportError:
    RANGER_AVAILABLE = False
    logger.warning("Ranger21 not available, using AdamW")


class WeldDataset(Dataset):
    """Dataset for weld images with Albumentations"""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[A.Compose] = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = np.array(image)

        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def create_transforms(config: dict, is_training: bool = True, image_size: int = 224) -> A.Compose:
    """Create Albumentations transforms"""

    if is_training:
        aug_config = config["augmentation"]["train"]

        transforms = [
            # Geometric augmentations
            A.HorizontalFlip(p=aug_config.get("horizontal_flip", 0.5)),
            A.VerticalFlip(p=aug_config.get("vertical_flip", 0.3)),
            A.Rotate(limit=aug_config.get("rotate_limit", 15), p=0.5),
            A.ShiftScaleRotate(
                shift_limit=aug_config.get("shift_limit", 0.1),
                scale_limit=aug_config.get("scale_limit", 0.2),
                rotate_limit=0,
                p=0.5,
            ),

            # Color augmentations
            A.HueSaturationValue(
                hue_shift_limit=aug_config.get("hue_shift_limit", 10),
                sat_shift_limit=aug_config.get("sat_shift_limit", 20),
                val_shift_limit=aug_config.get("val_shift_limit", 20),
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=aug_config.get("brightness_limit", 0.2),
                contrast_limit=aug_config.get("contrast_limit", 0.2),
                p=0.5,
            ),

            # Blur & noise
            A.OneOf([
                A.GaussianBlur(blur_limit=aug_config.get("blur_limit", 3), p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.3),

            A.GaussNoise(var_limit=aug_config.get("gaussian_noise", 0.02) * 255, p=0.3),

            # Industrial-specific
            A.RandomShadow(p=aug_config.get("random_shadow", 0.2)),
            A.RandomFog(p=aug_config.get("random_fog", 0.1)),

            # Resize and normalize
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        # Validation: only resize and normalize
        transforms = [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

    return A.Compose(transforms)


def mixup_data(x, y, alpha=1.0):
    """Apply MixUp augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix augmentation"""
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    # Get random crop box
    W, H = x.size()[2], x.size()[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


class WeldClassifierModule(pl.LightningModule):
    """PyTorch Lightning module for weld classification"""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Create model
        self.model = WeldClassifier(
            backbone_name=config["model"]["backbone"],
            num_classes=config["data"]["num_classes"],
            pretrained=config["model"]["pretrained"],
            freeze_stages=config["model"]["freeze_stages"],
            drop_rate=config["model"]["drop_rate"],
            drop_path_rate=config["model"]["drop_path_rate"],
            head_type=config["model"]["head"],
            embedding_size=config["model"].get("embedding_size", 512),
            scale=config["model"].get("scale", 30.0),
            margin=config["model"].get("margin", 0.5),
        )

        # Metrics
        num_classes = config["data"]["num_classes"]
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes)
        self.val_confusion = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config["training"].get("label_smoothing", 0.1)
        )

        # Augmentation config
        self.use_mixup = config["augmentation"]["mixup"]["enabled"]
        self.mixup_alpha = config["augmentation"]["mixup"]["alpha"]
        self.mixup_prob = config["augmentation"]["mixup"]["prob"]

        self.use_cutmix = config["augmentation"]["cutmix"]["enabled"]
        self.cutmix_alpha = config["augmentation"]["cutmix"]["alpha"]
        self.cutmix_prob = config["augmentation"]["cutmix"]["prob"]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        # Apply MixUp or CutMix randomly
        use_mix = False
        if self.use_mixup and np.random.random() < self.mixup_prob:
            x, y_a, y_b, lam = mixup_data(x, y, self.mixup_alpha)
            use_mix = True
        elif self.use_cutmix and np.random.random() < self.cutmix_prob:
            x, y_a, y_b, lam = cutmix_data(x, y, self.cutmix_alpha)
            use_mix = True

        # Forward pass
        if use_mix:
            logits = self.model(x, y_a)  # Pass y_a for ArcFace
            loss = lam * self.criterion(logits, y_a) + (1 - lam) * self.criterion(logits, y_b)
        else:
            logits = self.model(x, y)
            loss = self.criterion(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)

        # Logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        # Forward pass (no augmentation during validation)
        logits = self.model(x)
        loss = self.criterion(logits, y)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_confusion.update(preds, y)

        # Logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_recall", self.val_recall, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # Get parameter groups for discriminative LR
        if self.config["training"].get("discriminative_lr", {}).get("enabled", False):
            backbone_lr_mult = self.config["training"]["discriminative_lr"]["backbone_lr_multiplier"]
            head_lr_mult = self.config["training"]["discriminative_lr"]["head_lr_multiplier"]

            param_groups = self.model.get_parameter_groups(
                backbone_lr=self.config["training"]["lr"] * backbone_lr_mult,
                head_lr=self.config["training"]["lr"] * head_lr_mult,
            )
        else:
            param_groups = self.parameters()

        # Optimizer
        optimizer_name = self.config["training"]["optimizer"]

        if optimizer_name == "ranger21" and RANGER_AVAILABLE:
            optimizer = Ranger21(
                param_groups,
                lr=self.config["training"]["lr"],
                weight_decay=self.config["training"]["weight_decay"],
                num_epochs=self.config["training"]["epochs"],
                num_batches_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=self.config["training"]["lr"],
                weight_decay=self.config["training"]["weight_decay"],
                betas=self.config["training"].get("betas", [0.9, 0.999]),
            )

        # Scheduler
        scheduler_name = self.config["training"].get("scheduler", "cosine")

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config["training"]["epochs"],
                eta_min=self.config["training"].get("min_lr", 1e-6),
            )
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1,
            )
        else:
            scheduler = None

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        else:
            return optimizer


class WeldDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module"""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.current_image_size = config["data"]["initial_size"]

    def setup(self, stage=None):
        # Load image paths and labels
        data_dir = Path(self.config["data"]["merged_path"])

        # Find all images and create labels
        self.train_images = []
        self.train_labels = []
        self.val_images = []
        self.val_labels = []

        class_names = self.config["data"]["class_names"]

        for class_idx, class_name in enumerate(class_names):
            class_dir = data_dir / class_name
            if not class_dir.exists():
                continue

            image_paths = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))

            # Split train/val
            split_idx = int(len(image_paths) * self.config["data"]["train_split"])

            self.train_images.extend(image_paths[:split_idx])
            self.train_labels.extend([class_idx] * split_idx)

            self.val_images.extend(image_paths[split_idx:])
            self.val_labels.extend([class_idx] * (len(image_paths) - split_idx))

        logger.info(f"Train images: {len(self.train_images)}")
        logger.info(f"Val images: {len(self.val_images)}")

    def train_dataloader(self):
        transform = create_transforms(
            self.config,
            is_training=True,
            image_size=self.current_image_size,
        )

        dataset = WeldDataset(
            self.train_images,
            self.train_labels,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.config["data"]["pin_memory"],
            persistent_workers=self.config["data"].get("persistent_workers", True),
        )

    def val_dataloader(self):
        transform = create_transforms(
            self.config,
            is_training=False,
            image_size=self.current_image_size,
        )

        dataset = WeldDataset(
            self.val_images,
            self.val_labels,
            transform=transform,
        )

        return DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["data"]["num_workers"],
            pin_memory=self.config["data"]["pin_memory"],
            persistent_workers=self.config["data"].get("persistent_workers", True),
        )


def train_from_config(config: dict) -> pl.LightningModule:
    """Train model from Hydra config"""

    # Create data module
    data_module = WeldDataModule(config)

    # Create model
    model = WeldClassifierModule(config)

    # Callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["paths"]["checkpoint_dir"],
        filename=config["training"]["checkpoint"]["filename"],
        monitor=config["training"]["checkpoint"]["monitor"],
        mode=config["training"]["checkpoint"]["mode"],
        save_top_k=config["training"]["checkpoint"]["save_top_k"],
        save_last=config["training"]["checkpoint"]["save_last"],
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config["training"]["early_stopping"]["enabled"]:
        early_stop_callback = EarlyStopping(
            monitor=config["training"]["early_stopping"]["monitor"],
            patience=config["training"]["early_stopping"]["patience"],
            mode=config["training"]["early_stopping"]["mode"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
        )
        callbacks.append(early_stop_callback)

    # LR monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Loggers
    loggers = []

    if config["logging"]["wandb"]["enabled"]:
        wandb_logger = WandbLogger(
            project=config["logging"]["wandb"]["project"],
            name=config["logging"]["wandb"].get("name"),
            entity=config["logging"]["wandb"].get("entity"),
            tags=config["logging"]["wandb"].get("tags", []),
        )
        loggers.append(wandb_logger)

    if config["logging"]["tensorboard"]["enabled"]:
        tb_logger = TensorBoardLogger(
            save_dir=config["logging"]["tensorboard"]["save_dir"],
            name="weld_classifier",
        )
        loggers.append(tb_logger)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator=config["hardware"]["accelerator"],
        devices=config["hardware"]["devices"],
        precision=config["training"]["precision"],
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        check_val_every_n_epoch=config["validation"]["check_val_every_n_epoch"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
        benchmark=config["hardware"]["benchmark"],
        deterministic=config["hardware"]["deterministic"],
    )

    # Train
    trainer.fit(model, data_module)

    return model


if __name__ == "__main__":
    # This would typically be called from scripts/train.py with Hydra
    pass
