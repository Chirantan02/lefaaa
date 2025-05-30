#!/usr/bin/env python3
"""
Test the complete model loading process.
"""

import os
import sys

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def test_model_loading():
    """Test the complete model loading process."""
    
    print("Testing Leffa model loading process...")
    print("=" * 60)
    
    try:
        # Import the updated functions
        from lib.xmodel import download_hg_model_safe, get_manual_model_path, validate_model_directory
        
        model_id = "franciszzj/Leffa"
        exDir = "leffa"
        
        print(f"Model ID: {model_id}")
        print(f"Extra directory: {exDir}")
        print()
        
        # Test manual path detection first
        print("1. Testing manual path detection...")
        manual_path = get_manual_model_path(model_id, exDir)
        
        if manual_path:
            print(f"✅ Manual path found: {manual_path}")
            
            # Validate the directory
            is_valid = validate_model_directory(manual_path)
            print(f"✅ Directory validation: {'PASSED' if is_valid else 'FAILED'}")
            
            if is_valid:
                print("\n2. Testing inpainting model detection...")
                
                # Test the inpainting paths
                inpainting_paths = [
                    os.path.join(manual_path, "stable-diffusion-inpainting"),
                    os.path.join(manual_path, "inpainting"),
                    os.path.join(manual_path, "sd-inpainting"),
                    manual_path
                ]
                
                inpainting_found = None
                for path in inpainting_paths:
                    if os.path.exists(path) and os.path.isdir(path):
                        # Check for diffusers structure
                        diffusers_subdirs = ['scheduler', 'unet', 'vae']
                        if all(os.path.exists(os.path.join(path, subdir)) for subdir in diffusers_subdirs):
                            inpainting_found = path
                            break
                
                if inpainting_found:
                    print(f"✅ Inpainting model found: {inpainting_found}")
                    
                    print("\n3. Testing virtual try-on model detection...")
                    
                    # Test virtual try-on models
                    virtual_tryon_hd = os.path.join(manual_path, "virtual_tryon.pth")
                    virtual_tryon_dc = os.path.join(manual_path, "virtual_tryon_dc.pth")
                    
                    hd_exists = os.path.exists(virtual_tryon_hd)
                    dc_exists = os.path.exists(virtual_tryon_dc)
                    
                    print(f"✅ virtual_tryon.pth: {'FOUND' if hd_exists else 'MISSING'}")
                    print(f"✅ virtual_tryon_dc.pth: {'FOUND' if dc_exists else 'MISSING'}")
                    
                    if hd_exists and dc_exists:
                        print("\n🎉 ALL TESTS PASSED!")
                        print("The Leffa model should load successfully now.")
                        return True
                    else:
                        print("\n❌ Missing virtual try-on models")
                        return False
                else:
                    print("\n❌ Inpainting model not found")
                    return False
            else:
                print("\n❌ Directory validation failed")
                return False
        else:
            print("❌ Manual path not found")
            
            # Try the safe download function (offline mode)
            print("\n2. Testing safe download function (offline mode)...")
            try:
                result_path = download_hg_model_safe(model_id, exDir=exDir, offline_mode=True)
                print(f"✅ Safe download found model at: {result_path}")
                return True
            except Exception as e:
                print(f"❌ Safe download failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_node_loading():
    """Simulate the actual node loading process."""
    
    print("\n" + "=" * 60)
    print("Simulating CXH_Leffa_Viton_Load.gen() process...")
    print("-" * 60)
    
    try:
        from lib.xmodel import download_hg_model_safe
        
        model = "franciszzj/Leffa"
        viton_type = "hd"
        
        print(f"Model: {model}")
        print(f"Viton type: {viton_type}")
        print()
        
        # Step 1: Try to download or find the model
        print("Step 1: Finding model...")
        model_path = download_hg_model_safe(model, exDir='leffa', offline_mode=True, max_retries=3)
        print(f"✅ Model path: {model_path}")
        
        # Step 2: Check inpainting paths
        print("\nStep 2: Checking inpainting paths...")
        inpainting_paths = [
            os.path.join(model_path, "stable-diffusion-inpainting"),
            os.path.join(model_path, "inpainting"),
            os.path.join(model_path, "sd-inpainting"),
            model_path
        ]
        
        inpainting = None
        for path in inpainting_paths:
            print(f"  Checking: {path}")
            if os.path.exists(path):
                if os.path.isdir(path):
                    # Look for diffusers model structure
                    diffusers_subdirs = ['scheduler', 'unet', 'vae']
                    if all(os.path.exists(os.path.join(path, subdir)) for subdir in diffusers_subdirs):
                        print(f"  ✅ Found diffusers structure")
                        inpainting = path
                        break
        
        if inpainting:
            print(f"✅ Inpainting model: {inpainting}")
        else:
            print("❌ No inpainting model found")
            return False
        
        # Step 3: Check virtual try-on model
        print("\nStep 3: Checking virtual try-on model...")
        if viton_type == 'hd':
            virtual_tryon = os.path.join(model_path, "virtual_tryon.pth")
        else:
            virtual_tryon = os.path.join(model_path, "virtual_tryon_dc.pth")
        
        if os.path.exists(virtual_tryon):
            print(f"✅ Virtual try-on model: {virtual_tryon}")
        else:
            print(f"❌ Virtual try-on model not found: {virtual_tryon}")
            return False
        
        print("\n🎉 SIMULATION SUCCESSFUL!")
        print("The node should load without errors now.")
        return True
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_model_loading()
    success2 = simulate_node_loading()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ALL TESTS PASSED! The Leffa model should work now.")
    else:
        print("❌ Some tests failed. Please check the output above.")
