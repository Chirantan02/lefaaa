#!/usr/bin/env python3
"""
Test script to verify Leffa model path detection.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_model_paths():
    """Test if the Leffa model can be found in the expected locations."""
    
    print("Testing Leffa model path detection...")
    print("=" * 50)
    
    # Test the updated model detection
    try:
        from lib.xmodel import get_manual_model_path, validate_model_directory
        
        model_id = "franciszzj/Leffa"
        
        print(f"Looking for model: {model_id}")
        print()
        
        # Test manual path detection
        manual_path = get_manual_model_path(model_id, exDir='leffa')
        
        if manual_path:
            print(f"✅ Found model at: {manual_path}")
            
            # Verify the contents
            print("\nChecking model contents:")
            required_items = [
                "stable-diffusion-inpainting",
                "virtual_tryon.pth", 
                "virtual_tryon_dc.pth"
            ]
            
            for item in required_items:
                item_path = os.path.join(manual_path, item)
                exists = os.path.exists(item_path)
                print(f"  {item}: {'✅' if exists else '❌'}")
                
                if item == "stable-diffusion-inpainting" and exists:
                    # Check if it's a directory and has model files
                    if os.path.isdir(item_path):
                        model_files = ['model_index.json', 'config.json', 'pytorch_model.bin', 'diffusion_pytorch_model.bin']
                        found_files = [f for f in model_files if os.path.exists(os.path.join(item_path, f))]
                        print(f"    Found model files: {found_files}")
                    else:
                        print(f"    ❌ stable-diffusion-inpainting should be a directory")
            
        else:
            print("❌ Model not found in any expected location")
            
            # Show what paths were checked
            print("\nChecked these locations:")
            potential_paths = [
                os.path.join(os.getcwd(), "Leffa"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "Leffa"),
                os.path.join(os.path.dirname(__file__), "..", "models", "Leffa"),
                r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\Leffa",
                r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\leffa",
            ]
            
            for path in potential_paths:
                exists = os.path.exists(path)
                print(f"  {path}: {'✅' if exists else '❌'}")
                
                if exists:
                    # Check what's inside
                    try:
                        contents = os.listdir(path)
                        print(f"    Contents: {contents[:5]}{'...' if len(contents) > 5 else ''}")
                    except:
                        print(f"    Could not list contents")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

def test_specific_path():
    """Test the specific path mentioned by the user."""
    
    print("\n" + "=" * 50)
    print("Testing specific user path...")
    
    user_path = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\Leffa"
    
    print(f"Checking: {user_path}")
    
    if os.path.exists(user_path):
        print("✅ Base path exists")
        
        # Check contents
        try:
            contents = os.listdir(user_path)
            print(f"Contents: {contents}")
            
            # Check for stable-diffusion-inpainting
            inpainting_path = os.path.join(user_path, "stable-diffusion-inpainting")
            if os.path.exists(inpainting_path):
                print("✅ stable-diffusion-inpainting folder exists")
                
                if os.path.isdir(inpainting_path):
                    print("✅ stable-diffusion-inpainting is a directory")
                    
                    # Check its contents
                    try:
                        inpainting_contents = os.listdir(inpainting_path)
                        print(f"Inpainting folder contents: {inpainting_contents[:10]}{'...' if len(inpainting_contents) > 10 else ''}")
                        
                        # Check for key model files
                        model_files = ['model_index.json', 'config.json', 'pytorch_model.bin', 'diffusion_pytorch_model.bin']
                        found_files = [f for f in model_files if os.path.exists(os.path.join(inpainting_path, f))]
                        print(f"Found model files: {found_files}")
                        
                    except Exception as e:
                        print(f"❌ Could not list inpainting folder contents: {e}")
                else:
                    print("❌ stable-diffusion-inpainting is not a directory")
            else:
                print("❌ stable-diffusion-inpainting folder not found")
                
            # Check for virtual tryon files
            for filename in ["virtual_tryon.pth", "virtual_tryon_dc.pth"]:
                filepath = os.path.join(user_path, filename)
                if os.path.exists(filepath):
                    print(f"✅ {filename} exists")
                else:
                    print(f"❌ {filename} not found")
                    
        except Exception as e:
            print(f"❌ Could not list directory contents: {e}")
    else:
        print("❌ Base path does not exist")

if __name__ == "__main__":
    test_model_paths()
    test_specific_path()
    
    print("\n" + "=" * 50)
    print("Test completed!")
