#!/usr/bin/env python3
"""
Test script to verify the updated validation logic without folder_paths dependency.
"""

import os
import sys

def validate_model_directory(model_path: str) -> bool:
    """
    Validate that a directory contains the required Leffa model files.
    (Copy of the updated function from lib/xmodel.py)
    """
    if not os.path.exists(model_path):
        print(f"Path does not exist: {model_path}")
        return False

    # Check for required files/directories
    required_items = [
        "stable-diffusion-inpainting",  # Directory
        "virtual_tryon.pth",           # File
        "virtual_tryon_dc.pth"         # File
    ]

    for item in required_items:
        item_path = os.path.join(model_path, item)
        if not os.path.exists(item_path):
            print(f"Missing required item: {item_path}")
            return False

    # Additional check: stable-diffusion-inpainting should be a directory
    inpainting_path = os.path.join(model_path, "stable-diffusion-inpainting")
    if not os.path.isdir(inpainting_path):
        print(f"stable-diffusion-inpainting should be a directory: {inpainting_path}")
        return False

    # Check that stable-diffusion-inpainting has the expected subdirectories
    required_subdirs = ["scheduler", "unet", "vae"]
    for subdir in required_subdirs:
        subdir_path = os.path.join(inpainting_path, subdir)
        if not os.path.exists(subdir_path):
            print(f"Missing required subdirectory: {subdir_path}")
            return False
        if not os.path.isdir(subdir_path):
            print(f"Expected directory but found file: {subdir_path}")
            return False
        
        # Check for config.json in each subdirectory
        config_path = os.path.join(subdir_path, "config.json")
        if subdir in ["unet", "vae"] and not os.path.exists(config_path):
            print(f"Missing config.json in: {subdir_path}")
            return False
        
        # For scheduler, check for scheduler_config.json
        if subdir == "scheduler":
            scheduler_config_path = os.path.join(subdir_path, "scheduler_config.json")
            if not os.path.exists(scheduler_config_path):
                print(f"Missing scheduler_config.json in: {subdir_path}")
                return False

    return True

def test_inpainting_detection(model_path: str):
    """Test the inpainting model detection logic from leffaNode.py"""
    
    print(f"\nTesting inpainting detection for: {model_path}")
    print("-" * 50)
    
    # Check required files exist with multiple possible locations
    inpainting_paths = [
        os.path.join(model_path, "stable-diffusion-inpainting"),
        os.path.join(model_path, "inpainting"),
        os.path.join(model_path, "sd-inpainting"),
        model_path  # Sometimes the model_path itself contains the inpainting model
    ]

    inpainting = None
    for path in inpainting_paths:
        print(f"Checking path: {path}")
        if os.path.exists(path):
            print(f"  ✅ Path exists")
            # Check if it contains model files
            if os.path.isdir(path):
                print(f"  ✅ Is directory")
                # Look for diffusers model structure (scheduler, unet, vae subdirectories)
                diffusers_subdirs = ['scheduler', 'unet', 'vae']
                if all(os.path.exists(os.path.join(path, subdir)) for subdir in diffusers_subdirs):
                    print(f"  ✅ Has diffusers structure (scheduler, unet, vae)")
                    inpainting = path
                    break
                else:
                    missing_subdirs = [subdir for subdir in diffusers_subdirs if not os.path.exists(os.path.join(path, subdir))]
                    print(f"  ❌ Missing diffusers subdirs: {missing_subdirs}")
                
                # Fallback: Look for common model files
                model_files = ['model_index.json', 'config.json', 'pytorch_model.bin', 'diffusion_pytorch_model.bin']
                found_files = [f for f in model_files if os.path.exists(os.path.join(path, f))]
                if found_files:
                    print(f"  ✅ Has model files: {found_files}")
                    inpainting = path
                    break
                else:
                    print(f"  ❌ No model files found")
            else:
                print(f"  ❌ Is not a directory")
                # If it's a file, check if it's a model file
                if path.endswith(('.bin', '.safetensors', '.ckpt')):
                    print(f"  ✅ Is a model file")
                    inpainting = path
                    break
        else:
            print(f"  ❌ Path does not exist")

    if inpainting:
        print(f"\n✅ Found inpainting model at: {inpainting}")
        return True
    else:
        print(f"\n❌ No inpainting model found")
        return False

def main():
    print("Testing updated Leffa model validation...")
    print("=" * 60)
    
    # Test the specific user path
    user_path = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\Leffa"
    
    print(f"Testing path: {user_path}")
    
    # Test validation
    is_valid = validate_model_directory(user_path)
    print(f"\nValidation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    
    # Test inpainting detection
    inpainting_found = test_inpainting_detection(user_path)
    
    print("\n" + "=" * 60)
    print(f"Overall result: {'✅ SUCCESS' if is_valid and inpainting_found else '❌ FAILED'}")
    
    if is_valid and inpainting_found:
        print("\nThe Leffa model should now be detected correctly!")
        print("Try running your ComfyUI workflow again.")
    else:
        print("\nThere are still issues with the model detection.")

if __name__ == "__main__":
    main()
