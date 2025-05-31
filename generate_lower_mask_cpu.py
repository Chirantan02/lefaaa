#!/usr/bin/env python3
"""
Generate lower mask with CPU explicitly
"""

import os
import sys
import importlib.util

def generate_lower_mask_cpu():
    """Generate lower mask using CPU"""
    
    image_path = "test_image.jpg"
    output_dir = "output"
    
    print("🎭 Generating lower body mask with CPU...")
    
    try:
        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import lower mask module
        spec = importlib.util.spec_from_file_location("lower", "mask/lower.py")
        lower_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lower_module)
        
        print("✅ Lower mask module imported successfully")
        
        # Generate lower mask with CPU
        print("🎭 Generating lower body mask with CPU device...")
        lower_result = lower_module.generate_lower_body_with_shoes_mask_improved(
            image_path=image_path,
            model_root=".",
            output_dir=output_dir,
            device="cpu",  # Force CPU
            target_size=768
        )
        
        if lower_result['status'] == 'OK':
            print("✅ Lower body mask generated successfully!")
            print(f"📄 Lower mask result: {lower_result}")
            return True
        else:
            print(f"❌ Lower mask generation failed: {lower_result}")
            return False
            
    except Exception as e:
        print(f"❌ Error generating lower mask: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🎭 LOWER MASK GENERATION (CPU)")
    print("=" * 50)
    
    success = generate_lower_mask_cpu()
    
    if success:
        print("\n🎉 Lower mask generation completed!")
        
        # Check output directory
        output_dir = "output"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"\n📁 All generated files:")
            for file in files:
                file_path = os.path.join(output_dir, file)
                file_size = os.path.getsize(file_path)
                print(f"  📄 {file} ({file_size} bytes)")
    else:
        print("\n💥 Lower mask generation failed!")
        sys.exit(1)
