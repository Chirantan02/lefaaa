#!/usr/bin/env python3
"""
Modal ComfyUI Leffa Application
Deploys ComfyUI with custom Leffa nodes as a scalable API endpoint.
"""

import json
import subprocess
import uuid
import os
from typing import Dict, Any

import modal

# Reference to the models volume we created
models_volume = modal.Volume.from_name("comfyui-leffa-models")

# Build Modal image with ComfyUI and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "curl")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.3.8")
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.10"
    )
    # Install additional dependencies for Leffa
    .pip_install([
        "huggingface_hub[hf_transfer]==0.30.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "diffusers>=0.21.0",
        "xformers>=0.0.20",
        "accelerate",
        "einops",
        "fvcore",
        "imageio",
        "matplotlib",
        "numpy",
        "omegaconf",
        "opencv-python",
        "opencv-contrib-python>=4.8.0",
        "pandas",
        "pillow",
        "pycocotools",
        "pyyaml",
        "psutil",
        "peft",
        "regex==2024.5.15",
        "safetensors",
        "scikit-image",
        "scipy",
        "timm",
        "tokenizers",
        "torchmetrics",
        "tqdm",
        "cloudpickle",
        "onnxruntime-gpu>=1.15.0"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Copy custom node files to the image
def copy_custom_nodes():
    """Copy custom node files to ComfyUI custom_nodes directory"""
    import os

    print("Copying custom Leffa nodes...")

    # ComfyUI custom nodes directory
    custom_nodes_dir = "/root/comfy/ComfyUI/custom_nodes/Comfyui_leffa"
    os.makedirs(custom_nodes_dir, exist_ok=True)

    # Copy all necessary files (this will be done during image build)
    # The files will be copied from the local directory to the image
    print(f"Custom nodes directory created: {custom_nodes_dir}")

# Add custom node files to the image
image = (
    image.run_function(copy_custom_nodes)
    .copy_local_dir(".", "/root/comfy/ComfyUI/custom_nodes/Comfyui_leffa")
    .run_commands(
        # Install any additional requirements from the custom node
        "cd /root/comfy/ComfyUI/custom_nodes/Comfyui_leffa && pip install -r requirements.txt || true"
    )
)

# Create the Modal app
app = modal.App(name="comfyui-leffa-viton", image=image)

def setup_model_paths():
    """Set up model paths to point to the Modal volume"""
    import os

    # Create symlinks from ComfyUI models directory to our volume
    comfy_models_dir = "/root/comfy/ComfyUI/models"

    # Link Leffa models
    leffa_source = "/models/leffa"
    leffa_target = os.path.join(comfy_models_dir, "leffa")
    if not os.path.exists(leffa_target):
        os.makedirs(os.path.dirname(leffa_target), exist_ok=True)
        os.symlink(leffa_source, leffa_target)
        print(f"Linked Leffa models: {leffa_source} -> {leffa_target}")

    # Link FitDiT models
    fitdit_source = "/models/FitDiT_models"
    fitdit_target = os.path.join(comfy_models_dir, "FitDiT_models")
    if not os.path.exists(fitdit_target):
        os.makedirs(os.path.dirname(fitdit_target), exist_ok=True)
        os.symlink(fitdit_source, fitdit_target)
        print(f"Linked FitDiT models: {fitdit_source} -> {fitdit_target}")

    # Also create the mask model path that the custom nodes expect
    mask_model_source = "/models/mask"
    mask_model_target = "/root/comfy/ComfyUI/custom_nodes/Comfyui_leffa/mask/model"
    if not os.path.exists(mask_model_target):
        os.makedirs(os.path.dirname(mask_model_target), exist_ok=True)
        os.symlink(mask_model_source, mask_model_target)
        print(f"Linked mask models: {mask_model_source} -> {mask_model_target}")

@app.function(
    max_containers=1,
    gpu="L40S",  # Good GPU for inference
    volumes={"/models": models_volume},
    timeout=1800,  # 30 minutes
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=120)
def ui():
    """Launch ComfyUI web interface"""
    setup_model_paths()
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

@app.cls(
    scaledown_window=300,  # 5 minute container keep alive
    gpu="L40S",
    volumes={"/models": models_volume},
    timeout=1800,
)
@modal.concurrent(max_inputs=3)  # Process up to 3 requests per container
class ComfyUILeffa:
    port: int = 8000

    @modal.enter()
    def launch_comfy_background(self):
        """Launch ComfyUI server when container starts"""
        setup_model_paths()

        # Launch ComfyUI in background
        cmd = f"comfy launch --background -- --port {self.port}"
        subprocess.run(cmd, shell=True, check=True)
        print("ComfyUI server launched successfully")

    @modal.method()
    def infer(self, workflow_data: Dict[str, Any]) -> bytes:
        """Run inference on the ComfyUI workflow"""
        # Check server health
        self.poll_server_health()

        # Create temporary workflow file
        workflow_id = uuid.uuid4().hex
        workflow_file = f"/tmp/workflow_{workflow_id}.json"

        with open(workflow_file, 'w') as f:
            json.dump(workflow_data, f)

        # Run the workflow
        cmd = f"comfy run --workflow {workflow_file} --wait --timeout 1200 --verbose"
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)

        # Find output image
        output_dir = "/root/comfy/ComfyUI/output"

        # Look for the most recent output file
        output_files = []
        if os.path.exists(output_dir):
            for f in os.listdir(output_dir):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(output_dir, f)
                    output_files.append((file_path, os.path.getmtime(file_path)))

        if not output_files:
            raise Exception("No output image found")

        # Get the most recent file
        latest_file = max(output_files, key=lambda x: x[1])[0]

        # Read and return the image
        with open(latest_file, 'rb') as f:
            return f.read()

    @modal.fastapi_endpoint(method="POST")
    def api(self, item: Dict):
        """API endpoint for virtual try-on"""
        from fastapi import Response, HTTPException

        try:
            # Validate input
            required_fields = ["model_image", "garment_image"]
            for field in required_fields:
                if field not in item:
                    raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

            # Create workflow data based on your workflow.json
            workflow_data = self.create_workflow_data(item)

            # Run inference
            img_bytes = self.infer.local(workflow_data)

            return Response(img_bytes, media_type="image/png")

        except Exception as e:
            print(f"API Error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def create_workflow_data(self, input_data: Dict) -> Dict:
        """Create ComfyUI workflow data from API input"""

        # Determine viton_type based on mask_type
        mask_type = input_data.get("mask_type", "full")
        if mask_type == "lower":
            viton_type = "dc"  # Use dress code model for lower body
        else:
            viton_type = "hd"  # Use HD model for upper body and full body

        # Allow manual override if specified
        viton_type = input_data.get("viton_type", viton_type)

        print(f"Mask type: {mask_type} -> Viton type: {viton_type}")

        # Base workflow structure from your workflow.json
        workflow = {
            "1": {
                "inputs": {"image": input_data["model_image"], "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Model Image"}
            },
            "2": {
                "inputs": {"image": input_data["garment_image"], "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Garment Image"}
            },
            "3": {
                "inputs": {
                    "image": ["1", 0],
                    "mask_type": mask_type,
                    "device": "cuda",
                    "target_size": input_data.get("target_size", 768),
                    "horizontal_expansion": input_data.get("horizontal_expansion", 15),
                    "vertical_expansion": input_data.get("vertical_expansion", 0),
                    "closing_kernel_size": input_data.get("closing_kernel_size", 9),
                    "closing_iterations": input_data.get("closing_iterations", 3),
                    "opening_kernel_size": input_data.get("opening_kernel_size", 5),
                    "opening_iterations": input_data.get("opening_iterations", 1),
                    "gap_fill_kernel_size": input_data.get("gap_fill_kernel_size", 5),
                    "gap_fill_iterations": input_data.get("gap_fill_iterations", 2),
                    "offset_top": 0, "offset_bottom": 0, "offset_left": 0, "offset_right": 0,
                    "dilate_shoes": False, "force_gpu": True, "half_precision": False, "optimize_memory": True
                },
                "class_type": "CXH_NEW_Advanced_Mask_Generator",
                "_meta": {"title": "NEW Advanced Mask Generator"}
            },
            "4": {
                "inputs": {
                    "image": ["1", 0],
                    "device": "cuda",
                    "target_size": 512,
                    "include_hands": True,
                    "include_face": False
                },
                "class_type": "CXH_NEW_Advanced_Pose_Preprocessor",
                "_meta": {"title": "NEW Advanced Pose Preprocessor"}
            },
            "5": {
                "inputs": {
                    "model": "franciszzj/Leffa",
                    "viton_type": viton_type
                },
                "class_type": "CXH_Leffa_Viton_Load",
                "_meta": {"title": "Load Leffa Model"}
            },
            "6": {
                "inputs": {
                    "pipe": ["5", 0],
                    "model": ["7", 0],
                    "cloth": ["2", 0],
                    "pose": ["4", 0],
                    "mask": ["3", 0],
                    "steps": input_data.get("steps", 20),
                    "cfg": input_data.get("cfg", 2.5),
                    "seed": input_data.get("seed", 656545)
                },
                "class_type": "CXH_Leffa_Viton_Run",
                "_meta": {"title": "Run Virtual Try-On"}
            },
            "7": {
                "inputs": {
                    "image": ["1", 0],
                    "upscale_method": "lanczos",
                    "width": 768,
                    "height": 1024,
                    "crop": "disabled"
                },
                "class_type": "ImageScale",
                "_meta": {"title": "Scale Model Image"}
            },
            "8": {
                "inputs": {
                    "filename_prefix": f"leffa_output_{uuid.uuid4().hex[:8]}",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage",
                "_meta": {"title": "Save Result"}
            }
        }

        return workflow

    def poll_server_health(self) -> Dict:
        """Check if ComfyUI server is healthy"""
        import socket
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.port}/system_stats")
            urllib.request.urlopen(req, timeout=5)
            print("ComfyUI server is healthy")
        except (socket.timeout, urllib.error.URLError) as e:
            print(f"Server health check failed: {str(e)}")
            raise Exception("ComfyUI server is not healthy")
