#!/usr/bin/env python3
"""
Modal Model Downloader for ComfyUI Leffa Workflow
Downloads all required models to a Modal Volume for deployment.
"""

import os
import modal

# Create Modal volume for storing all models
models_volume = modal.Volume.from_name("comfyui-leffa-models", create_if_missing=True)

# Modal image with required dependencies for downloading
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "huggingface_hub[hf_transfer]==0.30.0",
        "torch>=2.0.0",
        "onnxruntime-gpu>=1.15.0",
        "requests",
        "tqdm"
    ])
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("comfyui-leffa-model-downloader", image=download_image)


def download_leffa_models():
    """Download Leffa models from HuggingFace"""
    from huggingface_hub import snapshot_download
    import os

    print("Downloading Leffa models...")

    # Download main Leffa model
    leffa_path = "/models/leffa/Leffa"
    os.makedirs(leffa_path, exist_ok=True)

    try:
        snapshot_download(
            repo_id="franciszzj/Leffa",
            local_dir=leffa_path,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("SUCCESS: Leffa models downloaded successfully!")

        # Verify required files exist
        required_files = [
            "stable-diffusion-inpainting",
            "virtual_tryon.pth",
            "virtual_tryon_dc.pth"
        ]

        for file_path in required_files:
            full_path = os.path.join(leffa_path, file_path)
            if os.path.exists(full_path):
                print(f"FOUND: {file_path}")
            else:
                print(f"MISSING: {file_path}")

    except Exception as e:
        print(f"ERROR: Failed to download Leffa models: {e}")
        raise


def download_fitdit_models():
    """Download FitDiT models for pose detection and human parsing"""
    from huggingface_hub import hf_hub_download
    import os

    print("Downloading FitDiT models...")

    # Create FitDiT models directory structure to match expected paths
    fitdit_base = "/models/FitDiT_models"
    os.makedirs(f"{fitdit_base}/dwpose", exist_ok=True)
    os.makedirs(f"{fitdit_base}/humanparsing", exist_ok=True)
    os.makedirs(f"{fitdit_base}/pose_guider", exist_ok=True)

    # Also create the nested structure that the mask code expects
    mask_fitdit_base = "/models/mask/model/FitDiT"
    os.makedirs(f"{mask_fitdit_base}/dwpose", exist_ok=True)
    os.makedirs(f"{mask_fitdit_base}/humanparsing", exist_ok=True)
    os.makedirs(f"{mask_fitdit_base}/pose_guider", exist_ok=True)

    # Download DWPose models
    dwpose_models = [
        ("dw-ll_ucoco_384.onnx", "dwpose"),
        ("yolox_l.onnx", "dwpose")
    ]

    # Download Human Parsing models
    parsing_models = [
        ("parsing_atr.onnx", "humanparsing"),
        ("parsing_lip.onnx", "humanparsing")
    ]

    # Download Pose Guider model
    pose_guider_models = [
        ("diffusion_pytorch_model.bin", "pose_guider")
    ]

    try:
        # Download from BoyuanJiang/FitDiT repository
        repo_id = "BoyuanJiang/FitDiT"

        # Download DWPose models to both locations
        for filename, subfolder in dwpose_models:
            print(f"Downloading {filename}...")
            # Download to main FitDiT location
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=fitdit_base,
                local_dir_use_symlinks=False
            )
            # Also download to mask location
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=mask_fitdit_base,
                local_dir_use_symlinks=False
            )
            print(f"SUCCESS: Downloaded {filename}")

        # Download Human Parsing models to both locations
        for filename, subfolder in parsing_models:
            print(f"Downloading {filename}...")
            # Download to main FitDiT location
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=fitdit_base,
                local_dir_use_symlinks=False
            )
            # Also download to mask location
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=mask_fitdit_base,
                local_dir_use_symlinks=False
            )
            print(f"SUCCESS: Downloaded {filename}")

        # Download Pose Guider model to both locations
        for filename, subfolder in pose_guider_models:
            print(f"Downloading {filename}...")
            # Download to main FitDiT location
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=fitdit_base,
                local_dir_use_symlinks=False
            )
            # Also download to mask location
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{subfolder}/{filename}",
                local_dir=mask_fitdit_base,
                local_dir_use_symlinks=False
            )
            print(f"SUCCESS: Downloaded {filename}")

        # Download model_index.json to both locations
        print("Downloading model_index.json...")
        hf_hub_download(
            repo_id=repo_id,
            filename="model_index.json",
            local_dir=fitdit_base,
            local_dir_use_symlinks=False
        )
        hf_hub_download(
            repo_id=repo_id,
            filename="model_index.json",
            local_dir=mask_fitdit_base,
            local_dir_use_symlinks=False
        )
        print("SUCCESS: Downloaded model_index.json")

        print("SUCCESS: All FitDiT models downloaded successfully!")

    except Exception as e:
        print(f"ERROR: Failed to download FitDiT models: {e}")
        raise


def verify_model_structure():
    """Verify all models are downloaded correctly"""
    print("Verifying model structure...")

    # Check Leffa models
    leffa_path = "/models/leffa/Leffa"
    leffa_required = [
        "stable-diffusion-inpainting",
        "virtual_tryon.pth",
        "virtual_tryon_dc.pth"
    ]

    print("\nLeffa Models:")
    for item in leffa_required:
        path = os.path.join(leffa_path, item)
        if os.path.exists(path):
            if os.path.isdir(path):
                print(f"FOUND Directory: {item}")
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"FOUND File: {item} ({size_mb:.1f} MB)")
        else:
            print(f"MISSING: {item}")

    # Check FitDiT models in main location
    fitdit_path = "/models/FitDiT_models"
    fitdit_required = [
        "dwpose/dw-ll_ucoco_384.onnx",
        "dwpose/yolox_l.onnx",
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx",
        "pose_guider/diffusion_pytorch_model.bin",
        "model_index.json"
    ]

    print("\nFitDiT Models (Main Location):")
    for item in fitdit_required:
        path = os.path.join(fitdit_path, item)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"FOUND File: {item} ({size_mb:.1f} MB)")
        else:
            print(f"MISSING: {item}")

    # Check FitDiT models in mask location
    mask_fitdit_path = "/models/mask/model/FitDiT"
    print("\nFitDiT Models (Mask Location):")
    for item in fitdit_required:
        path = os.path.join(mask_fitdit_path, item)
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"FOUND File: {item} ({size_mb:.1f} MB)")
        else:
            print(f"MISSING: {item}")

    # Calculate total size
    total_size = 0
    for root, _, files in os.walk("/models"):  # Use _ for unused dirs variable
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)

    total_size_gb = total_size / (1024 * 1024 * 1024)
    print(f"\nTotal model size: {total_size_gb:.2f} GB")


@app.function(
    volumes={"/models": models_volume},
    timeout=3600,  # 1 hour timeout for downloads
    cpu=2.0,
    memory=4096
)
def download_all_models():
    """Download all required models to the Modal volume"""
    print("Starting model download process...")

    try:
        # Download Leffa models
        download_leffa_models()

        # Download FitDiT models
        download_fitdit_models()

        # Verify everything downloaded correctly
        verify_model_structure()

        # Commit changes to volume
        models_volume.commit()

        print("\nSUCCESS: All models downloaded successfully!")
        print("Models are now stored in the 'comfyui-leffa-models' Modal volume")

    except Exception as e:
        print(f"\nERROR: Model download failed: {e}")
        raise


if __name__ == "__main__":
    print("Running model downloader...")
    with app.run():
        download_all_models.remote()
