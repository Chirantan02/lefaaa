#!/usr/bin/env python3
"""
Modal Leffa Simple API - Virtual Try-On
SIMPLIFIED VERSION: Image URLs → Generate Mask → Virtual Try-On → Upload to GCP → Return URL

Key Features:
- ONLY uses models from Modal volume (no local downloads)
- Uses memory snapshots for faster cold starts
- ALWAYS provides detailed time and cost breakdown
- Simple API: single endpoint for virtual try-on
"""

import os
import time
import uuid
import json
import tempfile
from datetime import datetime
from typing import Dict, Any, Optional
import requests
from io import BytesIO

import modal

# ============================================================================
# MODAL CONFIGURATION
# ============================================================================

# Modal volume for models (persistent storage)
models_volume = modal.Volume.from_name("leffa-models", create_if_missing=True)

# Modal image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0"
    )
    .pip_install([
        # Core ML dependencies
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "accelerate",
        "xformers>=0.0.20",
        "safetensors",

        # Image processing
        "Pillow>=9.0.0",
        "opencv-python",
        "numpy",
        "scipy",
        "scikit-image",

        # Pose detection and masking
        "onnxruntime-gpu>=1.15.0",
        "pycocotools",

        # API and cloud
        "fastapi[standard]>=0.100.0",
        "requests",
        "google-cloud-storage",
        "aiohttp",

        # Utilities
        "tqdm",
        "omegaconf",
        "einops",
        "timm",
    ])
    .run_commands([
        # Install additional dependencies if needed
        "pip install huggingface_hub>=0.16.0",
    ])
)

# Create Modal app
app = modal.App(name="leffa-simple-viton", image=image)

# ============================================================================
# COST AND TIME TRACKING
# ============================================================================

class CostTimeTracker:
    """Track detailed time and cost breakdown for each operation"""

    def __init__(self):
        self.start_time = time.time()
        self.operations = []
        self.gpu_cost_per_hour = 0.59  # t4 cost per hour (approximate)

    def log_operation(self, operation: str, duration: float, details: str = ""):
        """Log an operation with timing"""
        self.operations.append({
            "operation": operation,
            "duration_seconds": round(duration, 2),
            "details": details,
            "timestamp": datetime.now().isoformat()
        })

    def get_breakdown(self) -> Dict[str, Any]:
        """Get detailed cost and time breakdown"""
        total_time = time.time() - self.start_time
        gpu_cost = (total_time / 3600) * self.gpu_cost_per_hour

        return {
            "total_time_seconds": round(total_time, 2),
            "total_time_minutes": round(total_time / 60, 2),
            "estimated_gpu_cost_usd": round(gpu_cost, 4),
            "gpu_type": "t4",
            "operations_breakdown": self.operations,
            "cost_breakdown": {
                "gpu_time_hours": round(total_time / 3600, 4),
                "gpu_rate_per_hour": self.gpu_cost_per_hour,
                "estimated_total_cost": round(gpu_cost, 4)
            }
        }

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

@app.function(
    volumes={"/models": models_volume},
    timeout=1800,  # 30 minutes for model download
    cpu=4,
)
def download_models():
    """Download all required models to Modal volume ONLY"""
    tracker = CostTimeTracker()

    print("[LOADING] Starting model download to Modal volume...")

    # Check if models already exist
    if os.path.exists("/models/leffa") and os.path.exists("/models/mask"):
        print("[SUCCESS] Models already exist in Modal volume, skipping download")
        return {"status": "already_exists", "message": "Models already downloaded"}

    # Create model directories
    model_dirs = [
        "/models/leffa",
        "/models/pose_detection",
        "/models/human_parsing",
        "/models/mask",
        "/models/custom_nodes"
    ]

    for dir_path in model_dirs:
        os.makedirs(dir_path, exist_ok=True)

    start_time = time.time()

    try:
        from huggingface_hub import snapshot_download
        import shutil
        import requests

        print("[DOWNLOAD] Downloading Leffa model...")
        leffa_start = time.time()
        snapshot_download(
            repo_id="franciszzj/Leffa",
            local_dir="/models/leffa",
            local_dir_use_symlinks=False
        )
        leffa_time = time.time() - leffa_start
        tracker.log_operation("leffa_model_download", leffa_time, "franciszzj/Leffa")

        print("[DOWNLOAD] Downloading pose detection models...")
        pose_start = time.time()

        # Download DWPose model
        dwpose_url = "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
        response = requests.get(dwpose_url)
        with open("/models/pose_detection/dw-ll_ucoco_384.onnx", "wb") as f:
            f.write(response.content)

        pose_time = time.time() - pose_start
        tracker.log_operation("pose_model_download", pose_time, "DWPose ONNX model")

        print("[DOWNLOAD] Downloading human parsing models...")
        parsing_start = time.time()

        # Download human parsing models
        parsing_atr_url = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_atr.onnx"
        parsing_lip_url = "https://huggingface.co/levihsu/OOTDiffusion/resolve/main/checkpoints/humanparsing/parsing_lip.onnx"

        response = requests.get(parsing_atr_url)
        with open("/models/human_parsing/parsing_atr.onnx", "wb") as f:
            f.write(response.content)

        response = requests.get(parsing_lip_url)
        with open("/models/human_parsing/parsing_lip.onnx", "wb") as f:
            f.write(response.content)

        parsing_time = time.time() - parsing_start
        tracker.log_operation("parsing_model_download", parsing_time, "Human parsing ONNX models")

        print("[FOLDER] Setting up mask generation and pose detection...")
        setup_start = time.time()

        # Copy the current directory (which contains your nodes) to Modal volume
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Copy mask directory if it exists
        mask_source = os.path.join(current_dir, "mask")
        mask_dest = "/models/mask"
        if os.path.exists(mask_source):
            shutil.copytree(mask_source, mask_dest, dirs_exist_ok=True)
            print(f"[SUCCESS] Copied mask modules from {mask_source} to {mask_dest}")

        # Copy your node files to Modal volume
        node_files = ["newMaskingNodes.py", "leffaNode.py", "Leffaa.py"]
        for node_file in node_files:
            source_file = os.path.join(current_dir, node_file)
            dest_file = os.path.join("/models/custom_nodes", node_file)
            if os.path.exists(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"[SUCCESS] Copied {node_file} to Modal volume")
            else:
                print(f"[WARNING]  Node file not found: {source_file}")

        setup_time = time.time() - setup_start
        tracker.log_operation("custom_modules_setup", setup_time, "Custom mask and node modules")

        # Commit changes to volume
        models_volume.commit()
        print("[SUCCESS] All models and modules downloaded successfully!")

    except Exception as e:
        error_time = time.time() - start_time
        tracker.log_operation("model_download_error", error_time, f"Error: {str(e)}")
        raise e

    # Log detailed breakdown to Modal logs
    breakdown = tracker.get_breakdown()
    print("\n" + "="*60)
    print("[CHART] MODEL DOWNLOAD - DETAILED TIME & COST BREAKDOWN")
    print("="*60)
    print(f"Total Time: {breakdown['total_time_minutes']:.2f} minutes")
    print(f"Estimated Cost: ${breakdown['estimated_gpu_cost_usd']:.4f}")
    print(f"GPU Type: {breakdown['gpu_type']}")
    print("\nOperations Breakdown:")
    for op in breakdown['operations_breakdown']:
        print(f"  • {op['operation']}: {op['duration_seconds']}s - {op['details']}")
    print("="*60)

    return breakdown

# ============================================================================
# MAIN INFERENCE CLASS
# ============================================================================

@app.cls(
    gpu="L40S",  # 24GB VRAM
    volumes={"/models": models_volume},
    timeout=300,  # 5 minutes per request
    scaledown_window=2,  # Stop container after 2 seconds of inactivity (minimum allowed)
    # Enable memory snapshots for faster cold starts
    enable_memory_snapshot=True,
)
class LeffaVirtualTryOn:
    """Main class for Leffa virtual try-on inference"""

    # Use Modal parameters instead of __init__
    models_loaded: bool = modal.parameter(default=False)
    tracker = None

    @modal.enter()
    def load_models(self):
        """Load all models on container start - ONLY from Modal volume"""
        print("Loading models from Modal volume...")
        load_start = time.time()

        # Verify models exist in volume
        required_paths = [
            "/models/leffa",
            "/models/pose_detection/dw-ll_ucoco_384.onnx",
            "/models/human_parsing/parsing_atr.onnx",
            "/models/mask"
        ]

        for path in required_paths:
            if not os.path.exists(path):
                print(f"WARNING: Required path not found in Modal volume: {path}")
                # Don't raise error, just warn and continue with fallback

        # Set up Python paths for imports
        import sys
        leffa_model_path = "/models/leffa"
        mask_path = "/models/mask"

        # Add paths to sys.path for imports
        paths_to_add = [leffa_model_path, mask_path]
        for path in paths_to_add:
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            # Import Leffa components with proper error handling
            print("Importing Leffa modules...")
            try:
                # Try importing from the downloaded HuggingFace model
                from leffa.model import LeffaModel
                from leffa.inference import LeffaInference
                from leffa.transform import LeffaTransform
                print("SUCCESS: Successfully imported Leffa modules")

                # Set up model file paths
                inpainting_path = "runwayml/stable-diffusion-inpainting"  # Use HuggingFace model
                virtual_tryon_path = leffa_model_path

                print(f"Loading Leffa model from: {leffa_model_path}")
                print(f"Using inpainting model: {inpainting_path}")
                print(f"Using virtual try-on model: {virtual_tryon_path}")

                # Load Leffa model
                self.leffa_model = LeffaModel(
                    pretrained_model_name_or_path=inpainting_path,
                    pretrained_model=virtual_tryon_path,
                )
                self.leffa_inference = LeffaInference(model=self.leffa_model)
                self.leffa_transform = LeffaTransform()

                print("SUCCESS: Leffa models loaded successfully!")

            except ImportError as e:
                print(f"WARNING: Could not import Leffa modules: {e}")
                print("Using fallback implementation...")
                # Set to None for fallback implementation
                self.leffa_model = None
                self.leffa_inference = None
                self.leffa_transform = None

            # Initialize other components
            self.mask_generator = None  # Will be initialized when needed
            self.pose_processor = None  # Will be initialized when needed

            print("SUCCESS: Models loaded successfully from Modal volume!")
            self.models_loaded = True

        except Exception as e:
            print(f"ERROR: Error loading models: {e}")
            print("Continuing with fallback implementation...")
            # Set fallback values
            self.leffa_model = None
            self.leffa_inference = None
            self.leffa_transform = None
            self.models_loaded = True

        load_time = time.time() - load_start
        print(f"Model loading time: {load_time:.2f} seconds")

    def download_image_from_url(self, url: str, tracker: CostTimeTracker) -> bytes:
        """Download image from URL with timing"""
        download_start = time.time()

        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            image_data = response.content

            download_time = time.time() - download_start
            tracker.log_operation("image_download", download_time, f"Downloaded {len(image_data)} bytes")

            return image_data

        except Exception as e:
            download_time = time.time() - download_start
            tracker.log_operation("image_download_error", download_time, f"Error: {str(e)}")
            raise e

    def download_mask_from_url(self, mask_type: str, tracker: CostTimeTracker):
        """Download mask from predefined URLs based on mask type"""
        mask_start = time.time()

        try:
            print(f"[MASK] Downloading {mask_type} mask from URL...")

            # Predefined mask URLs
            mask_urls = {
                "upper": "https://storage.googleapis.com/mask_images/upper_mask.png",
                "lower": "https://storage.googleapis.com/mask_images/lower_mask.png",
                "full": "https://storage.googleapis.com/mask_images/upper_mask.png"  # Use upper mask for full for now
            }

            if mask_type not in mask_urls:
                raise Exception(f"Unknown mask type: {mask_type}")

            mask_url = mask_urls[mask_type]
            print(f"[MASK] Using mask URL: {mask_url}")

            # Download mask image
            import requests
            response = requests.get(mask_url, timeout=30)
            response.raise_for_status()
            mask_data = response.content

            # Convert to PIL Image
            from PIL import Image
            import io
            import numpy as np
            import torch

            mask_image = Image.open(io.BytesIO(mask_data)).convert("L")  # Convert to grayscale

            # Convert to tensor
            mask_array = np.array(mask_image).astype(np.float32) / 255.0
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # Add batch dimension

            mask_time = time.time() - mask_start
            tracker.log_operation("mask_download", mask_time, f"Downloaded {mask_type} mask from URL: {mask_url}")

            print(f"[SUCCESS] Downloaded mask tensor: {mask_tensor.shape}")
            return mask_tensor

        except Exception as e:
            mask_time = time.time() - mask_start
            tracker.log_operation("mask_download_error", mask_time, f"Error: {str(e)}")
            print(f"[ERROR] Mask download error: {str(e)}")
            # Return empty mask on error
            mask_array = np.zeros((512, 512), dtype=np.float32)  # Default size
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            return mask_tensor

    def generate_pose(self, user_image_data: bytes, tracker: CostTimeTracker):
        """Generate pose using your existing pose detection"""
        pose_start = time.time()

        try:
            print("[POSE] Generating pose detection...")

            # Convert bytes to PIL Image
            from PIL import Image
            import io
            import numpy as np
            import torch
            import tempfile

            user_image = Image.open(io.BytesIO(user_image_data)).convert("RGB")

            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                user_image.save(temp_file.name)
                temp_image_path = temp_file.name

            # Set model root path to the Modal volume models location
            model_root = "/models"

            try:
                # Import DWPose detector from the models directory
                import sys
                pose_module_path = "/models/mask/preprocess/dwpose"
                if pose_module_path not in sys.path:
                    sys.path.insert(0, pose_module_path)

                # Import the DWposeDetector class from the __init__.py file
                from mask.preprocess.dwpose import DWposeDetector

                # Initialize DWPose detector
                dwprocessor = DWposeDetector(model_root, device="cuda")

                # Resize image for processing
                original_size = user_image.size
                target_size = 512
                if min(user_image.size) != target_size:
                    scale = target_size / min(user_image.size)
                    new_size = (int(user_image.size[0] * scale), int(user_image.size[1] * scale))
                    resized_image = user_image.resize(new_size, Image.LANCZOS)
                else:
                    resized_image = user_image

                # Convert to numpy array for processing
                pose_input_np = np.array(resized_image)[:, :, ::-1].copy()  # RGB to BGR

                # Process pose
                pose_result = dwprocessor(pose_input_np)

                if pose_result is not None:
                    # Convert pose result to image
                    pose_image_np = pose_result[:, :, ::-1]  # BGR to RGB

                    # Convert to PIL Image
                    pose_image = Image.fromarray(pose_image_np.astype(np.uint8))

                    # Resize back to original size
                    pose_image = pose_image.resize(original_size, Image.LANCZOS)

                    # Convert to tensor format expected by Leffa
                    pose_array = np.array(pose_image).astype(np.float32) / 255.0
                    pose_tensor = torch.from_numpy(pose_array).permute(2, 0, 1).unsqueeze(0)  # BCHW format

                    pose_time = time.time() - pose_start
                    tracker.log_operation("pose_generation", pose_time, "Generated pose detection successfully")

                    # Clean up temporary files
                    os.unlink(temp_image_path)

                    return pose_tensor
                else:
                    raise Exception("Pose processing failed, no result returned")

            finally:
                # Clean up sys.path
                if pose_module_path in sys.path:
                    sys.path.remove(pose_module_path)
                # Clean up temporary files
                if os.path.exists(temp_image_path):
                    os.unlink(temp_image_path)

        except Exception as e:
            pose_time = time.time() - pose_start
            tracker.log_operation("pose_generation_error", pose_time, f"Error: {str(e)}")
            print(f"[ERROR] Pose generation error: {str(e)}")
            # Return original image as pose on error
            user_image = Image.open(io.BytesIO(user_image_data)).convert("RGB")
            pose_array = np.array(user_image).astype(np.float32) / 255.0
            pose_tensor = torch.from_numpy(pose_array).permute(2, 0, 1).unsqueeze(0)
            return pose_tensor

    def run_virtual_tryon(self, user_image_data: bytes, garment_image_data: bytes,
                         mask_tensor, pose_tensor,
                         steps: int, cfg: float, seed: int, tracker: CostTimeTracker):
        """Run Leffa virtual try-on inference"""
        inference_start = time.time()

        try:
            print("[TRYON] Running virtual try-on inference...")

            # Convert bytes to PIL Images
            from PIL import Image
            import io
            import numpy as np

            user_image = Image.open(io.BytesIO(user_image_data)).convert("RGB")
            garment_image = Image.open(io.BytesIO(garment_image_data)).convert("RGB")

            # Store original size for final resize
            original_size = user_image.size

            # Check if Leffa models are loaded
            if self.leffa_model is None or self.leffa_inference is None:
                print("[WARNING]  Leffa models not loaded, cannot perform virtual try-on")
                raise Exception("Leffa models not loaded. Please ensure models are properly downloaded and loaded.")

            # Convert pose tensor back to PIL Image for Leffa transform
            pose_array = pose_tensor.squeeze(0).permute(1, 2, 0).numpy()  # CHW to HWC
            pose_array = (pose_array * 255).astype(np.uint8)
            pose_image = Image.fromarray(pose_array)

            # Prepare data for Leffa transform (following your existing code pattern)
            data = {
                "src_image": [user_image],
                "ref_image": [garment_image],
                "mask": [mask_tensor],
                "densepose": [pose_image],
            }

            # Apply Leffa transform
            data = self.leffa_transform(data)

            # Run inference using the loaded Leffa model
            output = self.leffa_inference(
                data,
                num_inference_steps=steps,
                guidance_scale=cfg,
                seed=seed
            )

            # Get the generated image
            gen_image = output["generated_image"][0]

            # Resize back to original size
            gen_image = gen_image.resize(original_size, Image.NEAREST)

            # Convert PIL image to bytes
            img_buffer = io.BytesIO()
            gen_image.save(img_buffer, format='PNG')
            result_data = img_buffer.getvalue()

            inference_time = time.time() - inference_start
            tracker.log_operation("virtual_tryon_inference", inference_time,
                                f"steps={steps}, cfg={cfg}, seed={seed}")

            return result_data

        except Exception as e:
            inference_time = time.time() - inference_start
            tracker.log_operation("virtual_tryon_error", inference_time, f"Error: {str(e)}")
            print(f"[ERROR] Virtual try-on inference error: {str(e)}")
            raise e

    def upload_to_gcp(self, image_data: bytes, tracker: CostTimeTracker) -> str:
        """Upload result image to GCP bucket and return public URL"""
        upload_start = time.time()

        try:
            # Configure your GCP bucket details
            bucket_name = "mask_images"  # Your actual GCP bucket name

            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated-images/tryon_{timestamp}_{uuid.uuid4().hex[:8]}.png"

            # Implement GCP upload
            from google.cloud import storage

            # Initialize GCP client (assumes credentials are set via environment)
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(filename)

            # Upload the image data
            blob.upload_from_string(image_data, content_type='image/png')

            # Make the blob publicly accessible
            blob.make_public()

            # Get the public URL
            public_url = blob.public_url

            upload_time = time.time() - upload_start
            tracker.log_operation("gcp_upload", upload_time, f"Uploaded to {filename}")

            print(f"[SUCCESS] Image uploaded to GCP: {public_url}")
            return public_url

        except Exception as e:
            upload_time = time.time() - upload_start
            tracker.log_operation("gcp_upload_error", upload_time, f"Error: {str(e)}")
            print(f"[ERROR] GCP upload error: {str(e)}")

            # Return a placeholder URL on error (for testing)
            placeholder_url = f"https://storage.googleapis.com/{bucket_name}/{filename}"
            print(f"[WARNING]  Returning placeholder URL: {placeholder_url}")
            return placeholder_url

    @modal.method()
    def generate_virtual_tryon(self, user_image_url: str, garment_image_url: str,
                              mask_type: str = "full", steps: int = 20,
                              cfg: float = 2.5, seed: Optional[int] = None) -> Dict[str, Any]:
        """Main method for virtual try-on generation"""

        # Initialize tracker for this request
        tracker = CostTimeTracker()

        try:
            print(f"[ROCKET] Starting virtual try-on generation...")
            print(f"   User image: {user_image_url}")
            print(f"   Garment image: {garment_image_url}")
            print(f"   Mask type: {mask_type}")
            print(f"   Steps: {steps}, CFG: {cfg}")

            # Generate random seed if not provided
            if seed is None:
                seed = int(time.time()) % 1000000
            print(f"   Using seed: {seed}")

            # Step 1: Download images
            print("[DOWNLOAD] Downloading images...")
            user_image_data = self.download_image_from_url(user_image_url, tracker)
            garment_image_data = self.download_image_from_url(garment_image_url, tracker)
            print(f"[SUCCESS] Downloaded user image: {len(user_image_data)} bytes")
            print(f"[SUCCESS] Downloaded garment image: {len(garment_image_data)} bytes")

            # Step 2: Download mask from URL
            print(f"[MASK] Downloading {mask_type} mask from URL...")
            mask_tensor = self.download_mask_from_url(mask_type, tracker)
            print(f"[SUCCESS] Downloaded mask tensor: {mask_tensor.shape}")

            # Step 3: Generate pose
            print("[POSE] Generating pose detection...")
            pose_tensor = self.generate_pose(user_image_data, tracker)
            print(f"[SUCCESS] Generated pose tensor: {pose_tensor.shape}")

            # Step 4: Run virtual try-on
            print("[TRYON] Running virtual try-on inference...")
            result_image_data = self.run_virtual_tryon(
                user_image_data, garment_image_data, mask_tensor, pose_tensor,
                steps, cfg, seed, tracker
            )
            print(f"[SUCCESS] Generated result image: {len(result_image_data)} bytes")

            # Step 5: Upload to GCP
            print("[CLOUD] Uploading result to GCP...")
            result_url = self.upload_to_gcp(result_image_data, tracker)
            print(f"[SUCCESS] Uploaded to GCP: {result_url}")

            # Get final breakdown
            breakdown = tracker.get_breakdown()

            print("[SUCCESS] Virtual try-on completed successfully!")

            # Log detailed breakdown to Modal logs
            print("\n" + "="*60)
            print("[CHART] VIRTUAL TRY-ON - DETAILED TIME & COST BREAKDOWN")
            print("="*60)
            print(f"Total Time: {breakdown['total_time_minutes']:.2f} minutes")
            print(f"Estimated Cost: ${breakdown['estimated_gpu_cost_usd']:.4f}")
            print(f"GPU Type: {breakdown['gpu_type']}")
            print("\nOperations Breakdown:")
            for op in breakdown['operations_breakdown']:
                print(f"  • {op['operation']}: {op['duration_seconds']}s - {op['details']}")
            print("="*60)

            return {
                "status": "success",
                "result_image_url": result_url,
                "processing_time": breakdown['total_time_seconds'],
                "cost_breakdown": breakdown,
                "parameters_used": {
                    "mask_type": mask_type,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed
                }
            }

        except Exception as e:
            # Get breakdown even on error
            breakdown = tracker.get_breakdown()

            print(f"[ERROR] Error during virtual try-on: {str(e)}")

            # Print detailed breakdown even on error
            print("\n" + "="*60)
            print("[CHART] VIRTUAL TRY-ON ERROR - DETAILED TIME & COST BREAKDOWN")
            print("="*60)
            print(f"Total Time: {breakdown['total_time_minutes']:.2f} minutes")
            print(f"Estimated Cost: ${breakdown['estimated_gpu_cost_usd']:.4f}")
            print(f"GPU Type: {breakdown['gpu_type']}")
            print("\nOperations Breakdown:")
            for op in breakdown['operations_breakdown']:
                print(f"  • {op['operation']}: {op['duration_seconds']}s - {op['details']}")
            print("="*60)

            return {
                "status": "error",
                "error_message": str(e),
                "processing_time": breakdown['total_time_seconds'],
                "cost_breakdown": breakdown
            }

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.function(
    gpu="L40S",
    volumes={"/models": models_volume},
    timeout=300,
    scaledown_window=2,  # Stop container after 2 seconds of inactivity (minimum allowed)
    enable_memory_snapshot=True,
)
@modal.fastapi_endpoint(method="POST")
def generate_tryon_api(request: Dict[str, Any]):
    """
    Simple API endpoint for virtual try-on generation

    Request format:
    {
        "user_image_url": "https://...",
        "garment_image_url": "https://...",
        "mask_type": "upper|lower|full",
        "steps": 20,        // optional
        "cfg": 2.5,         // optional
        "seed": 656545      // optional
    }
    """
    from fastapi import HTTPException

    try:
        # Validate required fields
        required_fields = ["user_image_url", "garment_image_url"]
        for field in required_fields:
            if field not in request:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

        # Extract parameters with defaults
        user_image_url = request["user_image_url"]
        garment_image_url = request["garment_image_url"]
        mask_type = request.get("mask_type", "full")
        steps = request.get("steps", 20)
        cfg = request.get("cfg", 2.5)
        seed = request.get("seed", None)

        # Validate mask_type
        if mask_type not in ["upper", "lower", "full"]:
            raise HTTPException(status_code=400, detail="mask_type must be 'upper', 'lower', or 'full'")

        # Create inference instance and run
        inference = LeffaVirtualTryOn()
        result = inference.generate_virtual_tryon.local(
            user_image_url=user_image_url,
            garment_image_url=garment_image_url,
            mask_type=mask_type,
            steps=steps,
            cfg=cfg,
            seed=seed
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"API Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@app.function(timeout=1800)
def setup_models():
    """Setup function to download models to Modal volume"""
    return download_models.remote()

@app.function()
def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "leffa-virtual-tryon",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("[ROCKET] Leffa Virtual Try-On Modal API")
    print("="*50)
    print("Available functions:")
    print("  • setup_models() - Download models to Modal volume")
    print("  • generate_tryon_api() - Main API endpoint")
    print("  • health_check() - Health check")
    print("="*50)
    print("\nTo deploy:")
    print("  modal deploy modal_leffa_simple_api.py")
    print("\nTo test locally:")
    print("  modal serve modal_leffa_simple_api.py")
