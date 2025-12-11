"""
ArcFace and CosFace Heads
Creates huge margin between good/bad clusters on tiny data
Paper: ArcFace (s=30, m=0.5) gives best results on <1000 industrial images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class ArcFaceHead(nn.Module):
    """
    ArcFace head for metric learning
    Adds angular margin to enhance discriminability

    Reference: ArcFace: Additive Angular Margin Loss for Deep Face Recognition
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.5,
        easy_margin: bool = False,
    ):
        """
        Initialize ArcFace head

        Args:
            in_features: Embedding dimension
            out_features: Number of classes
            scale: Feature scale (s)
            margin: Angular margin (m) in radians
            easy_margin: Use easy margin (recommended: False)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Weight matrix (embedding centers)
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m) for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embeddings: Feature embeddings (batch_size, in_features)
            labels: Ground truth labels (batch_size,) - required for training

        Returns:
            Logits (batch_size, out_features)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Normalize weight matrix
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight_norm)
        cosine = cosine.clamp(-1, 1)  # Numerical stability

        if labels is None:
            # Inference mode
            return self.scale * cosine

        # Training mode: add angular margin
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            # Easy margin: use phi when theta+m < pi
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Hard margin: use phi when theta < pi-m
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert labels to one-hot
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to ground truth class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        return output


class CosFaceHead(nn.Module):
    """
    CosFace head for metric learning
    Adds cosine margin to enhance discriminability

    Simpler than ArcFace, sometimes better on very small datasets
    Reference: CosFace: Large Margin Cosine Loss for Deep Face Recognition
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 30.0,
        margin: float = 0.35,
    ):
        """
        Initialize CosFace head

        Args:
            in_features: Embedding dimension
            out_features: Number of classes
            scale: Feature scale (s)
            margin: Cosine margin (m)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embeddings: Feature embeddings (batch_size, in_features)
            labels: Ground truth labels (batch_size,)

        Returns:
            Logits (batch_size, out_features)
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Normalize weight matrix
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Compute cosine similarity
        cosine = F.linear(embeddings, weight_norm)

        if labels is None:
            # Inference mode
            return self.scale * cosine

        # Training mode: subtract margin from ground truth class
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin
        output = cosine - one_hot * self.margin
        output *= self.scale

        return output


class SoftmaxHead(nn.Module):
    """
    Standard softmax head (baseline)
    Use this for comparison with ArcFace/CosFace
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """
        Initialize softmax head

        Args:
            in_features: Input feature dimension
            out_features: Number of classes
        """
        super().__init__()

        self.fc = nn.Linear(in_features, out_features)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embeddings: Feature embeddings (batch_size, in_features)
            labels: Unused (for API compatibility)

        Returns:
            Logits (batch_size, out_features)
        """
        return self.fc(embeddings)


def create_head(
    head_type: str,
    in_features: int,
    num_classes: int,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create classification head

    Args:
        head_type: 'arcface', 'cosface', or 'softmax'
        in_features: Input feature dimension
        num_classes: Number of classes
        **kwargs: Additional arguments for head

    Returns:
        Classification head module
    """
    if head_type == "arcface":
        return ArcFaceHead(
            in_features=in_features,
            out_features=num_classes,
            scale=kwargs.get("scale", 30.0),
            margin=kwargs.get("margin", 0.5),
            easy_margin=kwargs.get("easy_margin", False),
        )
    elif head_type == "cosface":
        return CosFaceHead(
            in_features=in_features,
            out_features=num_classes,
            scale=kwargs.get("scale", 30.0),
            margin=kwargs.get("margin", 0.35),
        )
    elif head_type == "softmax":
        return SoftmaxHead(
            in_features=in_features,
            out_features=num_classes,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")


if __name__ == "__main__":
    # Test ArcFace head
    batch_size = 8
    embedding_dim = 512
    num_classes = 2

    # Create head
    head = ArcFaceHead(
        in_features=embedding_dim,
        out_features=num_classes,
        scale=30.0,
        margin=0.5,
    )

    # Random embeddings and labels
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # Forward pass (training)
    logits = head(embeddings, labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    # Forward pass (inference)
    logits_inf = head(embeddings)
    print(f"Inference logits shape: {logits_inf.shape}")

    # Test gradient flow
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    print(f"Loss: {loss.item():.4f}")
    print(f"Weight grad norm: {head.weight.grad.norm():.4f}")

    print("\nArcFace head test passed!")
