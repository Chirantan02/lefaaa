#!/usr/bin/env python3
"""
Deployment script for ComfyUI Leffa Modal application
"""

import subprocess
import sys
import time
import requests
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors"""
    if description:
        print(f"\n{description}...")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def check_modal_auth():
    """Check if Modal is authenticated"""
    try:
        result = subprocess.run("modal profile current", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Modal authenticated as: {result.stdout.strip()}")
            return True
        else:
            print("ERROR: Modal not authenticated. Run 'modal token new'")
            return False
    except Exception as e:
        print(f"ERROR: Failed to check Modal auth: {e}")
        return False

def deploy_app():
    """Deploy the Modal application"""
    print("=" * 60)
    print("DEPLOYING COMFYUI LEFFA TO MODAL")
    print("=" * 60)
    
    # Check Modal authentication
    if not check_modal_auth():
        return False
    
    # Deploy the application
    if not run_command("modal deploy modal_comfyui_app.py", "Deploying Modal application"):
        return False
    
    print("\nSUCCESS: Application deployed successfully!")
    return True

def serve_app():
    """Serve the Modal application for development"""
    print("=" * 60)
    print("SERVING COMFYUI LEFFA FOR DEVELOPMENT")
    print("=" * 60)
    
    # Check Modal authentication
    if not check_modal_auth():
        return False
    
    print("Starting Modal serve (press Ctrl+C to stop)...")
    print("This will give you live URLs for testing...")
    
    try:
        subprocess.run("modal serve modal_comfyui_app.py", shell=True, check=True)
    except KeyboardInterrupt:
        print("\nServe stopped by user")
    except Exception as e:
        print(f"ERROR: Serve failed: {e}")
        return False
    
    return True

def test_deployment(api_url, model_image="test_model.jpg", garment_image="test_garment.jpg"):
    """Test the deployed application"""
    print("=" * 60)
    print("TESTING DEPLOYMENT")
    print("=" * 60)
    
    # Check if test images exist
    if not Path(model_image).exists():
        print(f"WARNING: Test model image not found: {model_image}")
        print("Please provide test images to run the full test")
        return False
    
    if not Path(garment_image).exists():
        print(f"WARNING: Test garment image not found: {garment_image}")
        print("Please provide test images to run the full test")
        return False
    
    # Run the test client
    cmd = f"python test_client.py --api-url {api_url} --model-image {model_image} --garment-image {garment_image}"
    return run_command(cmd, "Running API test")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python deploy.py deploy    - Deploy to Modal")
        print("  python deploy.py serve     - Serve for development")
        print("  python deploy.py test <api_url> [model_image] [garment_image] - Test deployment")
        return 1
    
    command = sys.argv[1]
    
    if command == "deploy":
        success = deploy_app()
        if success:
            print("\n" + "=" * 60)
            print("DEPLOYMENT COMPLETE!")
            print("=" * 60)
            print("Next steps:")
            print("1. Get your API URL from the Modal dashboard")
            print("2. Test with: python deploy.py test <your_api_url>")
            print("3. Or use the test client directly:")
            print("   python test_client.py --api-url <url> --model-image <img1> --garment-image <img2>")
        return 0 if success else 1
    
    elif command == "serve":
        success = serve_app()
        return 0 if success else 1
    
    elif command == "test":
        if len(sys.argv) < 3:
            print("ERROR: API URL required for test command")
            print("Usage: python deploy.py test <api_url> [model_image] [garment_image]")
            return 1
        
        api_url = sys.argv[2]
        model_image = sys.argv[3] if len(sys.argv) > 3 else "test_model.jpg"
        garment_image = sys.argv[4] if len(sys.argv) > 4 else "test_garment.jpg"
        
        success = test_deployment(api_url, model_image, garment_image)
        return 0 if success else 1
    
    else:
        print(f"ERROR: Unknown command: {command}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
