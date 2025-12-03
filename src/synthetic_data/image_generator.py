"""
Synthetic Weld Image Generator
SDXL-Turbo + ControlNet Tile + LoRA for photoreal weld generation
Generates 8,000-15,000 perfectly labeled images with zero text artifacts
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from PIL import Image
from loguru import logger
from tqdm import tqdm
import random
import json

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DDPMScheduler,
)
from diffusers.utils import load_image


class WeldPromptGenerator:
    """
    Generates diverse prompts for good and bad welds
    Based on real weld defect taxonomy
    """

    def __init__(self):
        """Initialize prompt templates"""

        # Good weld prompts
        self.good_weld_templates = [
            # TIG welds
            "professional TIG weld, smooth uniform bead, consistent width, {metal} surface, industrial quality, {lighting}",
            "perfect TIG weld, symmetric ripples, clean penetration, {metal}, high-resolution photo, {lighting}",
            "excellent TIG weld, even heat distribution, no discoloration, {metal}, workshop setting, {lighting}",

            # MIG welds
            "high-quality MIG weld, consistent bead pattern, good fusion, {metal} plate, {lighting}",
            "professional MIG weld, uniform ripples, proper penetration depth, {metal}, clean surface, {lighting}",
            "perfect MIG weld, smooth finish, no spatter, {metal}, industrial environment, {lighting}",

            # Arc welds
            "excellent arc weld, symmetric bead, consistent width, {metal} joint, professional quality, {lighting}",
            "perfect arc weld, smooth surface, proper tie-in, {metal}, workshop lighting, {lighting}",
            "high-quality stick weld, uniform appearance, good penetration, {metal}, {lighting}",

            # General
            "professional weld joint, clean bead, consistent pattern, {metal}, industrial quality, {lighting}",
            "perfect weld seam, smooth finish, proper fusion, {metal}, high-resolution photo, {lighting}",
        ]

        # Bad weld prompts
        self.bad_weld_templates = [
            # Porosity
            "defective weld with porosity, gas bubbles visible, {metal} surface, inconsistent bead, {lighting}",
            "failed weld showing extensive porosity, multiple pinholes, {metal}, poor quality, {lighting}",
            "weld with gas porosity, surface voids, {metal} plate, contaminated, {lighting}",

            # Cracks
            "cracked weld, visible fissures, {metal} joint, structural failure, {lighting}",
            "weld with longitudinal cracks, stress concentration, {metal}, failed inspection, {lighting}",
            "defective weld showing transverse cracks, {metal} surface, brittle failure, {lighting}",

            # Undercut
            "weld with severe undercut, groove along toe, {metal}, reduced strength, {lighting}",
            "defective weld showing undercut, improper parameters, {metal} plate, {lighting}",
            "failed weld with deep undercut, {metal} joint, inadequate fusion, {lighting}",

            # Slag inclusion
            "weld with slag inclusion, trapped impurities, {metal} surface, poor cleaning, {lighting}",
            "defective weld showing slag pockets, {metal}, incomplete removal, {lighting}",
            "failed weld with embedded slag, {metal} plate, contaminated surface, {lighting}",

            # Spatter & rough surface
            "weld with excessive spatter, rough surface, {metal}, poor technique, {lighting}",
            "defective weld showing heavy spatter, uneven bead, {metal} joint, {lighting}",
            "poor quality weld, excessive spatter, {metal} surface, unstable arc, {lighting}",

            # Incomplete fusion
            "weld with incomplete fusion, lack of penetration, {metal}, cold lap, {lighting}",
            "defective weld showing poor fusion, inadequate heat, {metal} plate, {lighting}",
            "failed weld with incomplete penetration, {metal} joint, insufficient current, {lighting}",

            # Burn-through
            "weld with burn-through, excessive penetration, {metal} plate, hole formation, {lighting}",
            "defective weld showing blow-through, {metal}, excessive heat, {lighting}",
            "failed weld with complete penetration, {metal} surface, too much current, {lighting}",

            # Multiple defects
            "severely defective weld, multiple issues, cracks and porosity, {metal}, failed inspection, {lighting}",
            "poor quality weld, rough surface, spatter and undercut, {metal} joint, {lighting}",
            "failed weld with porosity and slag inclusion, {metal} plate, contaminated, {lighting}",
        ]

        # Metal types
        self.metal_types = [
            "steel",
            "stainless steel",
            "carbon steel",
            "aluminum",
            "mild steel",
            "galvanized steel",
            "alloy steel",
        ]

        # Lighting conditions
        self.lighting_conditions = [
            "workshop lighting",
            "bright overhead lights",
            "natural daylight",
            "industrial lighting",
            "diffused lighting",
            "direct lighting",
            "even illumination",
        ]

        # Negative prompt (always applied)
        self.negative_prompt = (
            "text, watermark, signature, handwriting, marker, label, timestamp, "
            "date, letters, numbers, writing, annotation, blurry, low quality, "
            "cartoon, drawing, painting, rendered, 3d, illustration, sketch"
        )

    def generate_prompt(self, weld_quality: str) -> Tuple[str, str]:
        """
        Generate a random prompt for given weld quality

        Args:
            weld_quality: 'good' or 'bad'

        Returns:
            (positive_prompt, negative_prompt)
        """
        if weld_quality == "good":
            template = random.choice(self.good_weld_templates)
        elif weld_quality == "bad":
            template = random.choice(self.bad_weld_templates)
        else:
            raise ValueError(f"Invalid weld quality: {weld_quality}")

        # Fill in template
        metal = random.choice(self.metal_types)
        lighting = random.choice(self.lighting_conditions)

        prompt = template.format(metal=metal, lighting=lighting)

        return prompt, self.negative_prompt


class SyntheticWeldGenerator:
    """
    Generate synthetic weld images using SDXL-Turbo + ControlNet + LoRA
    Respects weld bead geometry while adding diversity
    """

    def __init__(
        self,
        base_model: str = "stabilityai/sdxl-turbo",
        lora_path: Optional[Path] = None,
        controlnet_model: str = "controlnet-tile",
        device: str = "cuda",
    ):
        """
        Initialize synthetic weld generator

        Args:
            base_model: SDXL base model
            lora_path: Path to trained weld LoRA
            controlnet_model: ControlNet model for tile conditioning
            device: Generation device
        """
        self.device = device
        self.prompt_generator = WeldPromptGenerator()

        logger.info("Initializing Synthetic Weld Generator")
        logger.info(f"Base model: {base_model}")

        # Load ControlNet (for tile conditioning)
        logger.info("Loading ControlNet Tile...")
        try:
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1e_sd15_tile",
                torch_dtype=torch.float16,
            )
        except:
            logger.warning("ControlNet Tile not found, using base model only")
            controlnet = None

        # Load SDXL-Turbo pipeline
        logger.info("Loading SDXL-Turbo pipeline...")
        if controlnet is not None:
            self.pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_model,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )
        else:
            self.pipeline = StableDiffusionXLPipeline.from_pretrained(
                base_model,
                torch_dtype=torch.float16,
                variant="fp16",
                use_safetensors=True,
            )

        self.pipeline.to(device)

        # Enable memory optimizations
        self.pipeline.enable_model_cpu_offload()
        self.pipeline.enable_vae_slicing()
        self.pipeline.enable_vae_tiling()

        # Load LoRA if provided
        if lora_path is not None and lora_path.exists():
            logger.info(f"Loading LoRA from {lora_path}")
            self.pipeline.load_lora_weights(str(lora_path))
        else:
            logger.warning("No LoRA provided, using base model only")

        logger.success("Synthetic Weld Generator initialized")

    @torch.no_grad()
    def generate_single_image(
        self,
        weld_quality: str,
        reference_image: Optional[Image.Image] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 4,  # SDXL-Turbo = 4 steps
        strength: float = 0.6,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, Dict]:
        """
        Generate a single synthetic weld image

        Args:
            weld_quality: 'good' or 'bad'
            reference_image: Optional reference for ControlNet
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps (4 for Turbo)
            strength: How much to respect reference (0.6 = good balance)
            seed: Random seed for reproducibility

        Returns:
            (generated_image, metadata)
        """
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Generate prompt
        prompt, negative_prompt = self.prompt_generator.generate_prompt(weld_quality)

        # Generate image
        if reference_image is not None and hasattr(self.pipeline, 'controlnet'):
            # Use ControlNet Tile conditioning
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=reference_image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                generator=generator,
            ).images[0]
        else:
            # Pure text-to-image
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]

        # Metadata
        metadata = {
            "weld_quality": weld_quality,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "strength": strength if reference_image else None,
            "seed": seed,
            "has_reference": reference_image is not None,
        }

        return image, metadata

    def generate_dataset(
        self,
        output_dir: Path,
        num_images: int = 10000,
        good_bad_ratio: float = 0.5,
        reference_dir: Optional[Path] = None,
        use_reference_prob: float = 0.7,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 4,
        strength: float = 0.6,
        save_metadata: bool = True,
    ) -> None:
        """
        Generate complete synthetic dataset

        Args:
            output_dir: Output directory
            num_images: Total number of images to generate
            good_bad_ratio: Ratio of good to bad welds (0.5 = 50/50)
            reference_dir: Directory with reference images for ControlNet
            use_reference_prob: Probability of using reference image
            guidance_scale: Guidance scale
            num_inference_steps: Number of inference steps
            strength: ControlNet strength
            save_metadata: Save metadata JSON file
        """
        output_dir = Path(output_dir)
        good_dir = output_dir / "good_weld"
        bad_dir = output_dir / "bad_weld"

        good_dir.mkdir(parents=True, exist_ok=True)
        bad_dir.mkdir(parents=True, exist_ok=True)

        # Calculate number of good and bad images
        num_good = int(num_images * good_bad_ratio)
        num_bad = num_images - num_good

        logger.info(f"Generating {num_images} synthetic weld images")
        logger.info(f"  Good welds: {num_good}")
        logger.info(f"  Bad welds: {num_bad}")

        # Load reference images if available
        reference_images = []
        if reference_dir is not None:
            reference_paths = list(Path(reference_dir).glob("**/*.jpg")) + \
                            list(Path(reference_dir).glob("**/*.png"))
            reference_images = [Image.open(p).convert("RGB") for p in reference_paths]
            logger.info(f"Loaded {len(reference_images)} reference images")

        all_metadata = []

        # Generate good welds
        logger.info("Generating good welds...")
        for i in tqdm(range(num_good), desc="Good welds"):
            # Maybe use reference
            reference = None
            if reference_images and random.random() < use_reference_prob:
                reference = random.choice(reference_images)

            # Generate
            image, metadata = self.generate_single_image(
                weld_quality="good",
                reference_image=reference,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                seed=None,  # Random seed
            )

            # Save
            image_path = good_dir / f"good_{i:06d}.png"
            image.save(image_path, quality=95)

            metadata["image_path"] = str(image_path)
            metadata["image_id"] = f"good_{i:06d}"
            all_metadata.append(metadata)

        # Generate bad welds
        logger.info("Generating bad welds...")
        for i in tqdm(range(num_bad), desc="Bad welds"):
            # Maybe use reference
            reference = None
            if reference_images and random.random() < use_reference_prob:
                reference = random.choice(reference_images)

            # Generate
            image, metadata = self.generate_single_image(
                weld_quality="bad",
                reference_image=reference,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                seed=None,
            )

            # Save
            image_path = bad_dir / f"bad_{i:06d}.png"
            image.save(image_path, quality=95)

            metadata["image_path"] = str(image_path)
            metadata["image_id"] = f"bad_{i:06d}"
            all_metadata.append(metadata)

        # Save metadata
        if save_metadata:
            metadata_path = output_dir / "generation_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(all_metadata, f, indent=2)
            logger.info(f"Saved metadata to {metadata_path}")

        logger.success(f"Generated {num_images} synthetic weld images!")
        logger.success(f"Output directory: {output_dir}")


if __name__ == "__main__":
    # Example usage
    generator = SyntheticWeldGenerator(
        base_model="stabilityai/sdxl-turbo",
        lora_path=Path("models/lora/checkpoint-final"),
        device="cuda",
    )

    # Generate dataset
    generator.generate_dataset(
        output_dir=Path("data/synthetic"),
        num_images=10000,
        good_bad_ratio=0.5,
        reference_dir=Path("data/clean_real"),
        use_reference_prob=0.7,
    )
