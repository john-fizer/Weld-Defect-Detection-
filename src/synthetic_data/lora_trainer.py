"""
LoRA Training Module for SDXL-Turbo
Train weld-specific LoRA on 100-300 clean real images in <15 min on RTX 4090
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from PIL import Image
from loguru import logger
from tqdm import tqdm

from diffusers import (
    StableDiffusionXLPipeline,
    DDPMScheduler,
    AutoencoderKL,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
import bitsandbytes as bnb


class WeldImageDataset(Dataset):
    """Dataset for weld images"""

    def __init__(
        self,
        image_dir: Path,
        resolution: int = 1024,
        center_crop: bool = True,
    ):
        """
        Initialize dataset

        Args:
            image_dir: Directory containing clean weld images
            resolution: Target resolution (SDXL default: 1024)
            center_crop: Center crop images
        """
        self.image_paths = list(Path(image_dir).glob("**/*.jpg")) + \
                          list(Path(image_dir).glob("**/*.png")) + \
                          list(Path(image_dir).glob("**/*.jpeg"))

        self.resolution = resolution
        self.center_crop = center_crop

        logger.info(f"Found {len(self.image_paths)} training images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # Resize and crop
        if self.center_crop:
            # Center crop to square
            width, height = image.size
            crop_size = min(width, height)
            left = (width - crop_size) // 2
            top = (height - crop_size) // 2
            image = image.crop((left, top, left + crop_size, top + crop_size))

        # Resize to target resolution
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)

        # Convert to tensor and normalize to [-1, 1]
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        return {
            "pixel_values": image_tensor,
            "input_ids": torch.zeros(77, dtype=torch.long),  # Dummy tokens
        }


class LoRATrainer:
    """
    LoRA trainer for SDXL-Turbo on weld images
    800-1200 steps on 100-300 images
    """

    def __init__(
        self,
        model_name: str = "stabilityai/sdxl-turbo",
        output_dir: Path = Path("models/lora"),
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        device: str = "cuda",
    ):
        """
        Initialize LoRA trainer

        Args:
            model_name: Base SDXL model name
            output_dir: Directory to save LoRA weights
            rank: LoRA rank (16 is good balance)
            alpha: LoRA alpha (typically 2*rank)
            dropout: LoRA dropout rate
            device: Training device
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        logger.info(f"Initializing LoRA Trainer")
        logger.info(f"Model: {model_name}")
        logger.info(f"LoRA rank: {rank}, alpha: {alpha}")

        # Load base model
        logger.info("Loading SDXL-Turbo base model...")
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipeline.to(device)

        # Configure LoRA
        self.lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            init_lora_weights="gaussian",
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
                "proj_in", "proj_out",  # Projection layers
            ],
            lora_dropout=dropout,
        )

        # Apply LoRA to UNet
        logger.info("Applying LoRA to UNet...")
        self.unet = get_peft_model(self.pipeline.unet, self.lora_config)
        self.unet.print_trainable_parameters()

        logger.success("LoRA Trainer initialized")

    def train(
        self,
        train_data_dir: Path,
        num_train_epochs: int = 10,
        train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        lr_scheduler: str = "cosine",
        warmup_steps: int = 100,
        save_steps: int = 200,
        logging_steps: int = 10,
        mixed_precision: str = "fp16",
        resolution: int = 1024,
        seed: int = 42,
    ) -> None:
        """
        Train LoRA on weld images

        Args:
            train_data_dir: Directory with clean weld images
            num_train_epochs: Number of training epochs
            train_batch_size: Batch size
            gradient_accumulation_steps: Gradient accumulation steps
            learning_rate: Learning rate
            max_grad_norm: Max gradient norm for clipping
            lr_scheduler: LR scheduler type
            warmup_steps: Warmup steps
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            mixed_precision: Mixed precision training ('fp16' or 'bf16')
            resolution: Image resolution
            seed: Random seed
        """
        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create dataset
        dataset = WeldImageDataset(
            image_dir=train_data_dir,
            resolution=resolution,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        # Calculate total steps
        num_update_steps_per_epoch = len(dataloader) // gradient_accumulation_steps
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

        logger.info(f"Training configuration:")
        logger.info(f"  Num examples: {len(dataset)}")
        logger.info(f"  Num epochs: {num_train_epochs}")
        logger.info(f"  Batch size: {train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {max_train_steps}")

        # Optimizer (AdamW 8-bit from bitsandbytes)
        optimizer = bnb.optim.AdamW8bit(
            self.unet.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )

        # LR scheduler
        if lr_scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max_train_steps,
                eta_min=1e-7,
            )
        elif lr_scheduler == "constant_with_warmup":
            from transformers import get_constant_schedule_with_warmup
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
            )
        else:
            scheduler = None

        # Training loop
        global_step = 0
        progress_bar = tqdm(total=max_train_steps, desc="Training LoRA")

        self.unet.train()

        for epoch in range(num_train_epochs):
            for step, batch in enumerate(dataloader):
                pixel_values = batch["pixel_values"].to(
                    self.device,
                    dtype=torch.float16 if mixed_precision == "fp16" else torch.bfloat16
                )

                # Encode images to latent space
                with torch.no_grad():
                    latents = self.pipeline.vae.encode(
                        pixel_values
                    ).latent_dist.sample()
                    latents = latents * self.pipeline.vae.config.scaling_factor

                # Sample noise
                noise = torch.randn_like(latents)

                # Sample timestep
                timesteps = torch.randint(
                    0,
                    self.pipeline.scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                # Add noise to latents
                noisy_latents = self.pipeline.scheduler.add_noise(
                    latents, noise, timesteps
                )

                # Predict noise
                with torch.cuda.amp.autocast(enabled=(mixed_precision != "no")):
                    model_pred = self.unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=torch.zeros(
                            latents.shape[0], 77, 2048,
                            device=latents.device,
                            dtype=latents.dtype
                        ),
                    ).sample

                    # Compute loss
                    loss = torch.nn.functional.mse_loss(
                        model_pred.float(),
                        noise.float(),
                        reduction="mean",
                    )

                # Backward pass
                loss = loss / gradient_accumulation_steps
                loss.backward()

                # Update weights
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            self.unet.parameters(),
                            max_grad_norm
                        )

                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1
                    progress_bar.update(1)

                    # Logging
                    if global_step % logging_steps == 0:
                        lr = optimizer.param_groups[0]["lr"]
                        logger.info(
                            f"Step {global_step}: loss={loss.item():.4f}, lr={lr:.2e}"
                        )

                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_checkpoint(global_step)

                if global_step >= max_train_steps:
                    break

            if global_step >= max_train_steps:
                break

        progress_bar.close()

        # Save final checkpoint
        self.save_checkpoint("final")
        logger.success(f"Training complete! LoRA saved to {self.output_dir}")

    def save_checkpoint(self, step: int or str) -> None:
        """Save LoRA checkpoint"""
        checkpoint_dir = self.output_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA weights only
        self.unet.save_pretrained(checkpoint_dir)
        logger.info(f"Saved checkpoint to {checkpoint_dir}")


if __name__ == "__main__":
    # Example usage
    trainer = LoRATrainer(
        model_name="stabilityai/sdxl-turbo",
        output_dir=Path("models/lora"),
        rank=16,
        alpha=32,
    )

    trainer.train(
        train_data_dir=Path("data/clean_real"),
        num_train_epochs=10,
        train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        save_steps=200,
    )
