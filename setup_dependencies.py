#!/usr/bin/env python3
"""
Setup script for ComfyUI Leffa dependencies and models.
This script helps install missing dependencies and download required model files.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("🔍 Checking dependencies...")
    
    required_packages = {
        'cv2': 'opencv-python>=4.8.0',
        'onnxruntime': 'onnxruntime-gpu>=1.15.0',
        'diffusers': 'diffusers>=0.21.0',
        'transformers': 'transformers>=4.30.0',
        'huggingface_hub': 'huggingface_hub>=0.16.0',
        'skimage': 'scikit-image>=0.20.0'
    }
    
    missing_packages = []
    
    for import_name, package_spec in required_packages.items():
        if not check_package(import_name):
            missing_packages.append(package_spec)
            print(f"❌ Missing: {import_name}")
        else:
            print(f"✅ Found: {import_name}")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        for package in missing_packages:
            print(f"Installing {package}...")
            if install_package(package):
                print(f"✅ Successfully installed {package}")
            else:
                print(f"❌ Failed to install {package}")
    else:
        print("✅ All required packages are installed!")

def check_model_files():
    """Check if required model files exist."""
    print("\n🔍 Checking model files...")
    
    # Check FitDiT models
    fitdit_path = Path("C:/Users/chira/Downloads/ninja downlaod/ComfyUI_windows_portable_nvidia_1/ComfyUI_windows_portable/ComfyUI/models/FitDiT_models")
    
    required_fitdit_files = [
        "dwpose/dw-ll_ucoco_384.onnx",
        "dwpose/yolox_l.onnx", 
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx"
    ]
    
    print(f"FitDiT models path: {fitdit_path}")
    
    missing_fitdit = []
    for file_path in required_fitdit_files:
        full_path = fitdit_path / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_fitdit.append(file_path)
    
    # Check Leffa models
    leffa_path = Path("C:/Users/chira/Downloads/ninja downlaod/ComfyUI_windows_portable_nvidia_1/ComfyUI_windows_portable/ComfyUI/models/leffa")
    
    required_leffa_files = [
        "virtual_tryon.pth",
        "virtual_tryon_dc.pth"
    ]
    
    print(f"\nLeffa models path: {leffa_path}")
    
    missing_leffa = []
    for file_path in required_leffa_files:
        full_path = leffa_path / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            missing_leffa.append(file_path)
    
    # Check for stable-diffusion-inpainting
    inpainting_paths = [
        leffa_path / "stable-diffusion-inpainting",
        leffa_path / "inpainting",
        leffa_path / "sd-inpainting"
    ]
    
    inpainting_found = False
    for path in inpainting_paths:
        if path.exists() and path.is_dir():
            print(f"✅ Found inpainting model: {path}")
            inpainting_found = True
            break
    
    if not inpainting_found:
        print("❌ Missing: stable-diffusion-inpainting directory")
        missing_leffa.append("stable-diffusion-inpainting/")
    
    return missing_fitdit, missing_leffa

def print_download_instructions(missing_fitdit, missing_leffa):
    """Print instructions for downloading missing model files."""
    if missing_fitdit:
        print(f"\n📥 Missing FitDiT model files:")
        print("Please download these files and place them in the FitDiT_models directory:")
        for file_path in missing_fitdit:
            print(f"  - {file_path}")
        print("\nFitDiT models can be downloaded from the official repositories.")
    
    if missing_leffa:
        print(f"\n📥 Missing Leffa model files:")
        print("Please download these files from: https://huggingface.co/franciszzj/Leffa")
        for file_path in missing_leffa:
            print(f"  - {file_path}")
        print("\nSee DOWNLOAD_INSTRUCTIONS.md for detailed download steps.")

def main():
    """Main setup function."""
    print("🚀 ComfyUI Leffa Setup Script")
    print("=" * 50)
    
    # Check and install dependencies
    check_and_install_dependencies()
    
    # Check model files
    missing_fitdit, missing_leffa = check_model_files()
    
    # Print download instructions if needed
    if missing_fitdit or missing_leffa:
        print_download_instructions(missing_fitdit, missing_leffa)
        print(f"\n⚠️  Setup incomplete. Please download missing model files.")
        return False
    else:
        print(f"\n✅ Setup complete! All dependencies and model files are ready.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
