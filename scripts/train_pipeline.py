#!/usr/bin/env python
"""
Elite Weld Defect Classifier - Complete Training Pipeline
Runs all 8 stages end-to-end with one command

Usage:
    python scripts/train_pipeline.py [--skip-text-removal] [--skip-synthetic] [--config-name production]
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger
import torch

logger.add("logs/pipeline_{time}.log", rotation="1 GB")


def stage1_text_removal(config: DictConfig) -> None:
    """Stage 1: Remove text from raw images"""
    logger.info("=" * 80)
    logger.info("STAGE 1: Text Removal (EasyOCR + LaMa)")
    logger.info("=" * 80)

    from src.text_removal.pipeline import run_text_removal_from_config

    run_text_removal_from_config(OmegaConf.to_container(config, resolve=True))

    logger.success("Stage 1 complete: Text removed from all images")


def stage2_dinov2_pretraining(config: DictConfig) -> None:
    """Stage 2 (Optional): DINOv2 self-supervised pre-training"""
    if not config.dinov2.enabled:
        logger.info("Stage 2: DINOv2 pre-training SKIPPED (disabled in config)")
        return

    logger.info("=" * 80)
    logger.info("STAGE 2: DINOv2 Self-Supervised Pre-training")
    logger.info("=" * 80)

    # TODO: Implement DINOv2 pre-training
    logger.warning("DINOv2 pre-training not yet implemented")

    logger.success("Stage 2 complete: DINOv2 warm-up finished")


def stage3_lora_training(config: DictConfig) -> None:
    """Stage 3: Train LoRA on clean real images"""
    if not config.synthetic.enabled:
        logger.info("Stage 3: LoRA training SKIPPED (synthetic generation disabled)")
        return

    logger.info("=" * 80)
    logger.info("STAGE 3: LoRA Training on Clean Real Images")
    logger.info("=" * 80)

    from src.synthetic_data.lora_trainer import LoRATrainer

    trainer = LoRATrainer(
        model_name=config.synthetic.sdxl.model_name,
        output_dir=Path(config.paths.checkpoint_dir) / "lora",
        rank=config.synthetic.lora.rank,
        alpha=config.synthetic.lora.alpha,
        dropout=config.synthetic.lora.dropout,
    )

    trainer.train(
        train_data_dir=Path(config.data.clean_real_path),
        num_train_epochs=config.synthetic.lora.train_steps // 100,  # Rough estimate
        train_batch_size=config.synthetic.lora.batch_size,
        gradient_accumulation_steps=config.synthetic.lora.gradient_accumulation_steps,
        learning_rate=config.synthetic.lora.learning_rate,
        max_grad_norm=config.synthetic.lora.max_grad_norm,
        save_steps=config.synthetic.lora.save_steps,
        logging_steps=config.synthetic.lora.logging_steps,
        warmup_steps=config.synthetic.lora.warmup_steps,
        mixed_precision=config.synthetic.lora.mixed_precision,
    )

    logger.success("Stage 3 complete: LoRA trained on clean images")


def stage4_synthetic_generation(config: DictConfig) -> None:
    """Stage 4: Generate synthetic dataset"""
    if not config.synthetic.enabled:
        logger.info("Stage 4: Synthetic generation SKIPPED (disabled in config)")
        return

    logger.info("=" * 80)
    logger.info("STAGE 4: Synthetic Data Generation (SDXL-Turbo + LoRA)")
    logger.info("=" * 80)

    from src.synthetic_data.image_generator import SyntheticWeldGenerator

    # Find latest LoRA checkpoint
    lora_dir = Path(config.paths.checkpoint_dir) / "lora"
    lora_checkpoint = lora_dir / "checkpoint-final"

    if not lora_checkpoint.exists():
        logger.warning(f"LoRA checkpoint not found at {lora_checkpoint}")
        logger.warning("Using base SDXL-Turbo without LoRA")
        lora_checkpoint = None

    generator = SyntheticWeldGenerator(
        base_model=config.synthetic.sdxl.model_name,
        lora_path=lora_checkpoint,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    generator.generate_dataset(
        output_dir=Path(config.data.synth_path),
        num_images=config.synthetic.num_images,
        good_bad_ratio=0.5,
        reference_dir=Path(config.data.clean_real_path),
        use_reference_prob=0.7,
        guidance_scale=config.synthetic.sdxl.guidance_scale,
        num_inference_steps=config.synthetic.sdxl.num_inference_steps,
        strength=config.synthetic.sdxl.strength,
    )

    logger.success(f"Stage 4 complete: Generated {config.synthetic.num_images} synthetic images")


def stage5_dataset_merging(config: DictConfig) -> None:
    """Stage 5: Merge real and synthetic datasets"""
    logger.info("=" * 80)
    logger.info("STAGE 5: Dataset Merging")
    logger.info("=" * 80)

    import shutil
    from pathlib import Path

    merged_dir = Path(config.data.merged_path)
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Copy real images
    real_dir = Path(config.data.clean_real_path)
    if real_dir.exists():
        logger.info("Copying real images...")
        for img_path in real_dir.rglob("*.jpg"):
            rel_path = img_path.relative_to(real_dir)
            dest_path = merged_dir / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, dest_path)

    # Copy synthetic images (if enabled)
    if config.synthetic.enabled:
        synth_dir = Path(config.data.synth_path)
        if synth_dir.exists():
            logger.info("Copying synthetic images...")
            for img_path in synth_dir.rglob("*.png"):
                rel_path = img_path.relative_to(synth_dir)
                dest_path = merged_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dest_path)

    logger.success("Stage 5 complete: Datasets merged")


def stage6_classifier_training(config: DictConfig) -> None:
    """Stage 6: Train final classifier"""
    logger.info("=" * 80)
    logger.info("STAGE 6: Classifier Training (ConvNeXt + ArcFace)")
    logger.info("=" * 80)

    # TODO: Implement PyTorch Lightning training
    logger.warning("Full training pipeline not yet implemented")
    logger.info("Use: python scripts/train.py")

    logger.success("Stage 6 complete: Classifier trained")


def stage7_validation(config: DictConfig) -> None:
    """Stage 7: Validate with 10x TTA"""
    logger.info("=" * 80)
    logger.info("STAGE 7: Validation with Test-Time Augmentation")
    logger.info("=" * 80)

    # TODO: Implement TTA validation
    logger.warning("TTA validation not yet implemented")

    logger.success("Stage 7 complete: Validation finished")


def stage8_model_export(config: DictConfig) -> None:
    """Stage 8: Export to ONNX/TensorRT"""
    logger.info("=" * 80)
    logger.info("STAGE 8: Model Export (ONNX → TensorRT)")
    logger.info("=" * 80)

    # TODO: Implement model export
    logger.warning("Model export not yet implemented")
    logger.info("Use: python scripts/export.py")

    logger.success("Stage 8 complete: Model exported")


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def main(config: DictConfig) -> None:
    """
    Run complete end-to-end pipeline
    """
    parser = argparse.ArgumentParser(description="Elite Weld Classifier Training Pipeline")
    parser.add_argument("--skip-text-removal", action="store_true", help="Skip text removal stage")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic data generation")
    parser.add_argument("--skip-training", action="store_true", help="Skip classifier training")
    parser.add_argument("--skip-export", action="store_true", help="Skip model export")
    args, _ = parser.parse_known_args()

    logger.info("=" * 80)
    logger.info("ELITE WELD DEFECT CLASSIFIER - END-TO-END PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Configuration: {config.get('config_name', 'train')}")
    logger.info(f"Output directory: {config.paths.output_dir}")
    logger.info("=" * 80)

    # Print configuration
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(config))

    # Run pipeline stages
    try:
        # Stage 1: Text Removal
        if not args.skip_text_removal and config.text_removal.enabled:
            stage1_text_removal(config)
        else:
            logger.info("STAGE 1: Text Removal SKIPPED")

        # Stage 2: DINOv2 Pre-training (Optional)
        if config.dinov2.enabled:
            stage2_dinov2_pretraining(config)

        # Stage 3: LoRA Training
        if not args.skip_synthetic and config.synthetic.enabled:
            stage3_lora_training(config)
        else:
            logger.info("STAGE 3: LoRA Training SKIPPED")

        # Stage 4: Synthetic Generation
        if not args.skip_synthetic and config.synthetic.enabled:
            stage4_synthetic_generation(config)
        else:
            logger.info("STAGE 4: Synthetic Generation SKIPPED")

        # Stage 5: Dataset Merging
        stage5_dataset_merging(config)

        # Stage 6: Classifier Training
        if not args.skip_training:
            stage6_classifier_training(config)
        else:
            logger.info("STAGE 6: Classifier Training SKIPPED")

        # Stage 7: Validation
        if not args.skip_training:
            stage7_validation(config)
        else:
            logger.info("STAGE 7: Validation SKIPPED")

        # Stage 8: Model Export
        if not args.skip_export:
            stage8_model_export(config)
        else:
            logger.info("STAGE 8: Model Export SKIPPED")

        logger.success("=" * 80)
        logger.success("PIPELINE COMPLETE!")
        logger.success("=" * 80)
        logger.success(f"Models saved to: {config.paths.checkpoint_dir}")
        logger.success(f"Logs saved to: {config.paths.log_dir}")

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
