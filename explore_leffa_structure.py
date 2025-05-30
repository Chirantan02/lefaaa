#!/usr/bin/env python3
"""
Explore the Leffa model directory structure.
"""

import os

def explore_directory(path, max_depth=3, current_depth=0):
    """Recursively explore directory structure."""
    if current_depth > max_depth:
        return
        
    try:
        items = os.listdir(path)
        for item in sorted(items):
            item_path = os.path.join(path, item)
            indent = "  " * current_depth
            
            if os.path.isdir(item_path):
                print(f"{indent}{item}/")
                explore_directory(item_path, max_depth, current_depth + 1)
            else:
                # Show file size for files
                try:
                    size = os.path.getsize(item_path)
                    if size > 1024*1024:  # > 1MB
                        size_str = f"{size/(1024*1024):.1f}MB"
                    elif size > 1024:  # > 1KB
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size}B"
                    print(f"{indent}{item} ({size_str})")
                except:
                    print(f"{indent}{item}")
    except PermissionError:
        print(f"{indent}[Permission Denied]")
    except Exception as e:
        print(f"{indent}[Error: {e}]")

def main():
    leffa_path = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\Leffa"
    
    print(f"Exploring Leffa model directory: {leffa_path}")
    print("=" * 60)
    
    if os.path.exists(leffa_path):
        explore_directory(leffa_path)
    else:
        print("❌ Directory does not exist")
        
    print("\n" + "=" * 60)
    
    # Also check the stable-diffusion-inpainting subdirectories
    inpainting_path = os.path.join(leffa_path, "stable-diffusion-inpainting")
    if os.path.exists(inpainting_path):
        print(f"\nDetailed view of stable-diffusion-inpainting:")
        print("-" * 40)
        
        for subdir in ['scheduler', 'unet', 'vae']:
            subdir_path = os.path.join(inpainting_path, subdir)
            if os.path.exists(subdir_path):
                print(f"\n{subdir}/:")
                explore_directory(subdir_path, max_depth=1)

if __name__ == "__main__":
    main()
