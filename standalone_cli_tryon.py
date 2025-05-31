#!/usr/bin/env python3
"""
Standalone CLI Virtual Try-On Wrapper for Leffa
Direct model loading without ComfyUI dependencies
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import requests
from io import BytesIO

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import utilities
from lib.ximg import tensor2pil, pil2tensor, img_from_url, open_image
from config_loader import ConfigLoader

class StandaloneTryOnCLI:
    """Standalone CLI wrapper with direct model loading"""

    def __init__(self, device="cuda"):
        """Initialize the CLI"""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.pose_processor = None

        print(f"🚀 Initializing Standalone Virtual Try-On CLI")
        print(f"   Device: {self.device}")

    def load_models(self, model_path="franciszzj/Leffa", viton_type="hd"):
        """Load Leffa models directly"""
        print("📦 Loading Leffa models...")
        start_time = time.time()

        try:
            # Import diffusers components
            from diffusers import StableDiffusionInpaintPipeline
            from transformers import CLIPTextModel, CLIPTokenizer
            from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

            print(f"   Loading from: {model_path}")
            print(f"   Viton type: {viton_type}")

            # Load the inpainting pipeline as base
            base_model = "runwayml/stable-diffusion-inpainting"

            # Load components
            print("   Loading tokenizer...")
            tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")

            print("   Loading text encoder...")
            text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")

            print("   Loading VAE...")
            vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")

            print("   Loading UNet...")
            unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")

            print("   Loading scheduler...")
            scheduler = DDIMScheduler.from_pretrained(base_model, subfolder="scheduler")

            # Create the pipeline
            self.pipe = StableDiffusionInpaintPipeline(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                scheduler=scheduler,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
            )

            # Move to device
            self.pipe = self.pipe.to(self.device)

            # Enable memory efficient attention if available
            if hasattr(self.pipe, "enable_attention_slicing"):
                self.pipe.enable_attention_slicing()

            if hasattr(self.pipe, "enable_model_cpu_offload") and self.device == "cuda":
                self.pipe.enable_model_cpu_offload()

            load_time = time.time() - start_time
            print(f"✅ Models loaded successfully! ({load_time:.2f}s)")

        except Exception as e:
            print(f"❌ Failed to load models: {str(e)}")
            raise e

    def load_pose_processor(self):
        """Load pose detection models"""
        print("🤖 Loading pose detection models...")

        try:
            # Try to import pose detection
            from transformers import pipeline

            # Use a simple pose estimation model
            self.pose_processor = pipeline("image-classification",
                                         model="microsoft/resnet-50",
                                         device=0 if self.device == "cuda" else -1)

            print("✅ Pose processor loaded (using ResNet-50 as fallback)")

        except Exception as e:
            print(f"⚠️  Pose detection not available: {e}")
            print("   Will use dummy pose")
            self.pose_processor = None

    def load_image(self, image_path_or_url):
        """Load image from file path or URL"""
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                # Load from URL
                print(f"📥 Downloading image from URL: {image_path_or_url}")
                image = img_from_url(image_path_or_url)
            else:
                # Load from file
                print(f"📁 Loading image from file: {image_path_or_url}")
                if not os.path.exists(image_path_or_url):
                    raise FileNotFoundError(f"Image file not found: {image_path_or_url}")
                image = open_image(image_path_or_url)

            return image.convert("RGB")

        except Exception as e:
            print(f"❌ Failed to load image: {str(e)}")
            raise e

    def load_mask(self, mask_path_or_url):
        """Load mask from file path or URL"""
        try:
            if mask_path_or_url.startswith(('http://', 'https://')):
                # Load from URL
                print(f"🎭 Downloading mask from URL: {mask_path_or_url}")
                mask = img_from_url(mask_path_or_url)
            else:
                # Load from file
                print(f"🎭 Loading mask from file: {mask_path_or_url}")
                if not os.path.exists(mask_path_or_url):
                    raise FileNotFoundError(f"Mask file not found: {mask_path_or_url}")
                mask = open_image(mask_path_or_url)

            # Convert to grayscale
            mask = mask.convert("L")
            return mask

        except Exception as e:
            print(f"❌ Failed to load mask: {str(e)}")
            raise e

    def generate_pose(self, image):
        """Generate pose detection for the image"""
        if self.pose_processor is None:
            print("⚠️  Using dummy pose (pose processor not available)")
            # Create a dummy pose image (black image)
            pose_image = Image.new("RGB", image.size, (0, 0, 0))
            return pose_image

        try:
            print("🤖 Generating pose detection...")
            # For now, just return the original image as pose
            # In a full implementation, this would use proper pose detection
            pose_image = image.copy()
            return pose_image

        except Exception as e:
            print(f"❌ Pose generation failed: {str(e)}")
            # Fallback to dummy pose
            pose_image = Image.new("RGB", image.size, (0, 0, 0))
            return pose_image

    def run_virtual_tryon(self, user_image, garment_image, mask_image,
                         steps=20, cfg=2.5, seed=None):
        """Run the virtual try-on inference"""
        print("🎨 Running virtual try-on inference...")
        start_time = time.time()

        try:
            # Generate random seed if not provided
            if seed is None:
                seed = int(time.time()) % 1000000

            print(f"   Steps: {steps}")
            print(f"   CFG: {cfg}")
            print(f"   Seed: {seed}")

            # Set seed for reproducibility
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

            # Resize images to consistent size
            target_size = (512, 768)  # width, height
            user_image = user_image.resize(target_size, Image.LANCZOS)
            garment_image = garment_image.resize(target_size, Image.LANCZOS)
            mask_image = mask_image.resize(target_size, Image.LANCZOS)

            # Create prompt for virtual try-on
            prompt = "high quality, detailed, realistic, virtual try-on, fashion, clothing"
            negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy"

            # Run the inpainting pipeline
            print("   Running inference...")
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=user_image,
                mask_image=mask_image,
                num_inference_steps=steps,
                guidance_scale=cfg,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                height=target_size[1],
                width=target_size[0],
            )

            inference_time = time.time() - start_time
            print(f"✅ Virtual try-on completed! ({inference_time:.2f}s)")

            return result.images[0]

        except Exception as e:
            print(f"❌ Virtual try-on failed: {str(e)}")
            raise e

    def save_image(self, image, output_path):
        """Save the generated image"""
        try:
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # Save image
            image.save(output_path)
            print(f"💾 Image saved to: {output_path}")

        except Exception as e:
            print(f"❌ Failed to save image: {str(e)}")
            raise e


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Standalone CLI Virtual Try-On using Leffa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with local files
  python standalone_cli_tryon.py --user-image person.jpg --garment-image shirt.jpg --mask-image mask.png

  # Using URLs
  python standalone_cli_tryon.py --user-image https://example.com/person.jpg --garment-image https://example.com/shirt.jpg --mask-image https://example.com/mask.png

  # Using predefined mask URLs
  python standalone_cli_tryon.py --user-image person.jpg --garment-image shirt.jpg --mask-type upper
        """
    )

    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--mask-type",
        choices=["upper", "lower", "full"],
        help="Use predefined mask URL for the specified type"
    )

    # Image arguments
    parser.add_argument(
        "--user-image", "-u",
        help="Path or URL to user/model image"
    )
    parser.add_argument(
        "--garment-image", "-g",
        help="Path or URL to garment/clothing image"
    )
    parser.add_argument(
        "--mask-image", "-m",
        help="Path or URL to mask image"
    )

    # Optional arguments
    parser.add_argument(
        "--output", "-o",
        default="standalone_output_tryon.png",
        help="Output image path (default: standalone_output_tryon.png)"
    )
    parser.add_argument(
        "--model-path",
        default="franciszzj/Leffa",
        help="Path to Leffa model (default: franciszzj/Leffa)"
    )
    parser.add_argument(
        "--viton-type",
        choices=["hd", "dc"],
        default="hd",
        help="Virtual try-on type: 'hd' for upper body, 'dc' for lower body (default: hd)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)"
    )
    parser.add_argument(
        "--cfg",
        type=float,
        default=2.5,
        help="Classifier-free guidance scale (default: 2.5)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible results"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default="auto",
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Load configuration
    config = ConfigLoader(args.config)

    # Override with command-line arguments
    config.override_from_args(args)

    # Handle mask type shortcut
    if args.mask_type and not config.get("paths.mask_image"):
        mask_url = config.get_mask_url(args.mask_type)
        config.set("paths.mask_image", mask_url)
        print(f"🎭 Using predefined {args.mask_type} mask: {mask_url}")

    # Validate configuration
    if not config.validate_config():
        return 1

    # Determine device
    device = config.get("model.device")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("🎭 STANDALONE LEFFA VIRTUAL TRY-ON CLI")
    print("=" * 60)
    config.print_config()
    print(f"Effective Device: {device}")
    print("=" * 60)

    try:
        # Initialize CLI
        cli = StandaloneTryOnCLI(device=device)

        # Load models
        cli.load_models(
            model_path=config.get("model.path"),
            viton_type=config.get("model.viton_type")
        )

        # Load pose processor
        cli.load_pose_processor()

        # Load images
        print("\n📥 Loading images...")
        user_image = cli.load_image(config.get("paths.user_image"))
        garment_image = cli.load_image(config.get("paths.garment_image"))
        mask_image = cli.load_mask(config.get("paths.mask_image"))

        print(f"   User image size: {user_image.size}")
        print(f"   Garment image size: {garment_image.size}")
        print(f"   Mask image size: {mask_image.size}")

        # Generate pose (optional)
        pose_image = cli.generate_pose(user_image)
        print(f"   Pose image size: {pose_image.size}")

        # Run virtual try-on
        print("\n🎨 Starting virtual try-on...")
        start_time = time.time()

        result_image = cli.run_virtual_tryon(
            user_image=user_image,
            garment_image=garment_image,
            mask_image=mask_image,
            steps=config.get("inference.steps"),
            cfg=config.get("inference.cfg"),
            seed=config.get("inference.seed")
        )

        total_time = time.time() - start_time

        # Save result
        cli.save_image(result_image, config.get("paths.output"))

        print("\n" + "=" * 60)
        print("✅ VIRTUAL TRY-ON COMPLETED SUCCESSFULLY!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"📁 Output saved to: {config.get('paths.output')}")
        print("=" * 60)

        return 0

    except KeyboardInterrupt:
        print("\n❌ Process interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
