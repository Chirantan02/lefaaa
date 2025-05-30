#!/usr/bin/env python3
"""
Manual download script for Leffa model.
This script helps users download the Leffa model manually when automatic download fails.
"""

import os
import sys
import argparse
from pathlib import Path

def download_leffa_model(output_dir=None, use_git=False):
    """
    Download the Leffa model manually.
    
    Args:
        output_dir: Directory to save the model (default: ./Leffa)
        use_git: Whether to use git clone instead of huggingface_hub
    """
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "Leffa")
    
    print(f"Downloading Leffa model to: {output_dir}")
    
    if use_git:
        # Use git clone method
        try:
            import subprocess
            cmd = [
                "git", "clone", 
                "https://huggingface.co/franciszzj/Leffa",
                output_dir
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            print("Download completed successfully using git!")
            
        except subprocess.CalledProcessError as e:
            print(f"Git clone failed: {e}")
            return False
        except FileNotFoundError:
            print("Git not found. Please install git or use --no-git option.")
            return False
    else:
        # Use huggingface_hub method
        try:
            from huggingface_hub import snapshot_download
            
            print("Downloading using huggingface_hub...")
            snapshot_download(
                repo_id="franciszzj/Leffa",
                local_dir=output_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            print("Download completed successfully using huggingface_hub!")
            
        except ImportError:
            print("huggingface_hub not installed. Installing...")
            try:
                import subprocess
                subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"], check=True)
                from huggingface_hub import snapshot_download
                
                snapshot_download(
                    repo_id="franciszzj/Leffa",
                    local_dir=output_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print("Download completed successfully!")
                
            except Exception as e:
                print(f"Failed to install huggingface_hub: {e}")
                return False
                
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nTrying alternative method with git...")
            return download_leffa_model(output_dir, use_git=True)
    
    # Verify download
    required_files = [
        "stable-diffusion-inpainting",
        "virtual_tryon.pth",
        "virtual_tryon_dc.pth"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = os.path.join(output_dir, file_path)
        if not os.path.exists(full_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nWarning: Some required files are missing:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        print("\nPlease check the download or download manually from:")
        print("https://huggingface.co/franciszzj/Leffa")
        return False
    else:
        print(f"\nDownload verification successful!")
        print(f"Model saved to: {output_dir}")
        print("\nRequired files found:")
        for file_path in required_files:
            print(f"  ✓ {file_path}")
        return True


def main():
    parser = argparse.ArgumentParser(description="Download Leffa model manually")
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for the model (default: ./Leffa)",
        default=None
    )
    parser.add_argument(
        "--use-git", 
        action="store_true",
        help="Use git clone instead of huggingface_hub"
    )
    
    args = parser.parse_args()
    
    print("Leffa Model Manual Download Script")
    print("=" * 40)
    
    success = download_leffa_model(args.output_dir, args.use_git)
    
    if success:
        print("\n" + "=" * 40)
        print("Download completed successfully!")
        print("\nTo use the model in ComfyUI:")
        print("1. Copy the downloaded 'Leffa' folder to one of these locations:")
        print(f"   - {os.path.join(os.getcwd(), 'Leffa')}")
        print(f"   - {os.path.join(os.path.dirname(__file__), 'models', 'Leffa')}")
        print("2. Restart ComfyUI")
        print("3. The Leffa nodes should now work without download errors")
    else:
        print("\n" + "=" * 40)
        print("Download failed!")
        print("\nManual download instructions:")
        print("1. Go to: https://huggingface.co/franciszzj/Leffa")
        print("2. Click 'Download' or use git clone")
        print("3. Extract/place the files in the correct location")
        print("4. Ensure these files exist:")
        print("   - stable-diffusion-inpainting/ (directory)")
        print("   - virtual_tryon.pth")
        print("   - virtual_tryon_dc.pth")


if __name__ == "__main__":
    main()
