#!/usr/bin/env python3
"""
Simple script to generate masks for the downloaded image
"""

import os
import sys
import tempfile
from PIL import Image

def generate_masks():
    """Generate upper and lower masks for the test image"""
    
    image_path = "test_image.jpg"
    output_dir = "output"
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Generating masks for: {image_path}")
    print(f"📁 Output directory: {output_dir}")
    
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import mask generation modules
        print("📦 Importing mask generation modules...")
        
        # Try to import upper mask module
        try:
            import importlib.util
            
            # Import upper mask module
            spec = importlib.util.spec_from_file_location("upper", "mask/upper.py")
            upper_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upper_module)
            
            print("✅ Upper mask module imported successfully")
            
            # Generate upper mask
            print("🎭 Generating upper body mask...")
            upper_result = upper_module.generate_upper_body_mask(
                image_path=image_path,
                model_root=".",
                output_dir=output_dir,
                device="cuda",
                target_size=768
            )
            
            if upper_result['status'] == 'OK':
                print("✅ Upper body mask generated successfully!")
                print(f"📄 Upper mask result: {upper_result}")
            else:
                print(f"❌ Upper mask generation failed: {upper_result}")
                
        except Exception as e:
            print(f"❌ Error generating upper mask: {e}")
        
        # Try to import lower mask module
        try:
            # Import lower mask module
            spec = importlib.util.spec_from_file_location("lower", "mask/lower.py")
            lower_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(lower_module)
            
            print("✅ Lower mask module imported successfully")
            
            # Generate lower mask
            print("🎭 Generating lower body mask...")
            lower_result = lower_module.generate_lower_body_with_shoes_mask_improved(
                image_path=image_path,
                model_root=".",
                output_dir=output_dir,
                device="cuda",
                target_size=768
            )
            
            if lower_result['status'] == 'OK':
                print("✅ Lower body mask generated successfully!")
                print(f"📄 Lower mask result: {lower_result}")
            else:
                print(f"❌ Lower mask generation failed: {lower_result}")
                
        except Exception as e:
            print(f"❌ Error generating lower mask: {e}")
        
        # Check output directory for generated files
        print("\n📁 Checking output directory...")
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            if files:
                print(f"✅ Generated files:")
                for file in files:
                    file_path = os.path.join(output_dir, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  📄 {file} ({file_size} bytes)")
            else:
                print("❌ No files generated in output directory")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during mask generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎭 MASK GENERATION SCRIPT")
    print("=" * 50)
    
    success = generate_masks()
    
    if success:
        print("\n🎉 Mask generation process completed!")
    else:
        print("\n💥 Mask generation failed!")
        sys.exit(1)
