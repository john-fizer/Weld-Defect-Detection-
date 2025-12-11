"""
Elite Weld Defect Classifier Model
ConvNeXtV2 / TinyViT backbone + ArcFace head
Optimized for <1000 industrial images with 22k ImageNet pre-training
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Tuple
from loguru import logger

from .arcface import create_head


class WeldClassifier(nn.Module):
    """
    Complete weld defect classifier
    Backbone + optional projection + classification head
    """

    def __init__(
        self,
        backbone_name: str = "convnextv2_nano.fcmae_ft_in22k_in1k_384",
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_stages: int = 2,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        head_type: str = "arcface",
        embedding_size: int = 512,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        """
        Initialize weld classifier

        Args:
            backbone_name: TIMM model name
            num_classes: Number of classes
            pretrained: Use ImageNet pre-trained weights
            freeze_stages: Number of early stages to freeze
            drop_rate: Dropout rate
            drop_path_rate: DropPath rate (stochastic depth)
            head_type: Classification head ('arcface', 'cosface', 'softmax')
            embedding_size: Embedding dimension for metric learning
            scale: Scale factor for ArcFace/CosFace
            margin: Margin for ArcFace/CosFace
            easy_margin: Use easy margin for ArcFace
        """
        super().__init__()

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.head_type = head_type
        self.embedding_size = embedding_size

        logger.info(f"Initializing WeldClassifier")
        logger.info(f"  Backbone: {backbone_name}")
        logger.info(f"  Head: {head_type}")
        logger.info(f"  Num classes: {num_classes}")

        # Create backbone from TIMM
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Get backbone output dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[1]

        logger.info(f"  Backbone output dim: {backbone_dim}")

        # Freeze early stages
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)
            logger.info(f"  Frozen {freeze_stages} stages")

        # Projection layer for embedding
        if head_type in ["arcface", "cosface"]:
            self.projection = nn.Sequential(
                nn.Linear(backbone_dim, embedding_size),
                nn.BatchNorm1d(embedding_size),
            )
            head_in_features = embedding_size
        else:
            self.projection = nn.Identity()
            head_in_features = backbone_dim

        # Classification head
        self.head = create_head(
            head_type=head_type,
            in_features=head_in_features,
            num_classes=num_classes,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
        )

        # Dropout
        self.dropout = nn.Dropout(drop_rate)

        logger.success(f"WeldClassifier initialized successfully")

    def _freeze_stages(self, freeze_stages: int) -> None:
        """
        Freeze early stages of backbone

        Args:
            freeze_stages: Number of stages to freeze
        """
        # This is backbone-specific
        # For ConvNeXt: freeze stages.0, stages.1, etc.
        # For ViT: freeze patch_embed and first N blocks

        if "convnext" in self.backbone_name.lower():
            # ConvNeXt has 4 stages
            if hasattr(self.backbone, 'stages'):
                for i in range(min(freeze_stages, len(self.backbone.stages))):
                    for param in self.backbone.stages[i].parameters():
                        param.requires_grad = False
            # Also freeze stem
            if hasattr(self.backbone, 'stem'):
                for param in self.backbone.stem.parameters():
                    param.requires_grad = False

        elif "vit" in self.backbone_name.lower() or "tiny_vit" in self.backbone_name.lower():
            # ViT has patch_embed + blocks
            if hasattr(self.backbone, 'patch_embed'):
                for param in self.backbone.patch_embed.parameters():
                    param.requires_grad = False

            if hasattr(self.backbone, 'blocks'):
                num_blocks = len(self.backbone.blocks)
                freeze_blocks = int(num_blocks * freeze_stages / 4)  # Map stages to blocks
                for i in range(freeze_blocks):
                    for param in self.backbone.blocks[i].parameters():
                        param.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_embedding: bool = False,
    ) -> torch.Tensor or Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input images (batch_size, 3, H, W)
            labels: Ground truth labels (batch_size,)
            return_embedding: Return both logits and embeddings

        Returns:
            logits or (logits, embeddings)
        """
        # Extract features
        features = self.backbone(x)

        # Project to embedding space
        embeddings = self.projection(features)
        embeddings = self.dropout(embeddings)

        # Classification
        logits = self.head(embeddings, labels)

        if return_embedding:
            return logits, embeddings
        else:
            return logits

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings only

        Args:
            x: Input images (batch_size, 3, H, W)

        Returns:
            Embeddings (batch_size, embedding_size)
        """
        features = self.backbone(x)
        embeddings = self.projection(features)
        return embeddings

    def freeze_backbone(self) -> None:
        """Freeze entire backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self) -> None:
        """Unfreeze entire backbone"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

    def get_parameter_groups(
        self,
        backbone_lr: float,
        head_lr: float,
    ) -> list:
        """
        Get parameter groups for discriminative learning rates

        Args:
            backbone_lr: Learning rate for backbone
            head_lr: Learning rate for head

        Returns:
            List of parameter groups
        """
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": head_lr},
        ]

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }


def create_weld_classifier(config: dict) -> WeldClassifier:
    """
    Create weld classifier from Hydra config

    Args:
        config: Hydra configuration dictionary

    Returns:
        WeldClassifier model
    """
    model_config = config["model"]

    model = WeldClassifier(
        backbone_name=model_config["backbone"],
        num_classes=config["data"]["num_classes"],
        pretrained=model_config["pretrained"],
        freeze_stages=model_config["freeze_stages"],
        drop_rate=model_config["drop_rate"],
        drop_path_rate=model_config["drop_path_rate"],
        head_type=model_config["head"],
        embedding_size=model_config.get("embedding_size", 512),
        scale=model_config.get("scale", 30.0),
        margin=model_config.get("margin", 0.5),
        easy_margin=model_config.get("easy_margin", False),
    )

    # Log parameter counts
    param_counts = model.count_parameters()
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {param_counts['total']:,}")
    logger.info(f"  Trainable: {param_counts['trainable']:,}")
    logger.info(f"  Frozen: {param_counts['frozen']:,}")

    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing ConvNeXtV2 model...")
    model_convnext = WeldClassifier(
        backbone_name="convnextv2_nano.fcmae_ft_in22k_in1k_384",
        num_classes=2,
        freeze_stages=2,
        head_type="arcface",
    )

    # Test forward pass
    x = torch.randn(4, 3, 384, 384)
    labels = torch.randint(0, 2, (4,))

    logits = model_convnext(x, labels)
    print(f"ConvNeXt output shape: {logits.shape}")

    # Test embedding extraction
    embeddings = model_convnext.get_embedding(x)
    print(f"Embedding shape: {embeddings.shape}")

    # Test parameter counts
    param_counts = model_convnext.count_parameters()
    print(f"Parameters: {param_counts}")

    print("\n" + "="*60 + "\n")

    # Test TinyViT model
    print("Testing TinyViT model...")
    model_tinyvit = WeldClassifier(
        backbone_name="tiny_vit_21m_384.dist_in22k_ft_in1k",
        num_classes=2,
        freeze_stages=1,
        head_type="cosface",
    )

    logits = model_tinyvit(x, labels)
    print(f"TinyViT output shape: {logits.shape}")

    param_counts = model_tinyvit.count_parameters()
    print(f"Parameters: {param_counts}")

    print("\nModel tests passed!")
