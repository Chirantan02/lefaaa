#!/usr/bin/env python3
"""
Redeploy the fixed Leffa Virtual Try-On Modal API
"""

import subprocess
import time
import sys

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}...")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print("Error:", result.stderr)
            if result.stdout:
                print("Output:", result.stdout)
            return False
            
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def main():
    print("🚀 REDEPLOYING FIXED LEFFA VIRTUAL TRY-ON API")
    print("="*60)
    
    # Step 1: Deploy the app
    print("\n📦 Step 1: Deploying the fixed Modal app...")
    deploy_success = run_command(
        "modal deploy modal_leffa_simple_api.py",
        "Deploying Modal app"
    )
    
    if not deploy_success:
        print("\n❌ Deployment failed! Please check the errors above.")
        return False
    
    # Step 2: Setup models (only if needed)
    print("\n📥 Step 2: Setting up models in Modal volume...")
    print("Note: This will only download if models don't already exist.")
    
    setup_success = run_command(
        "modal run modal_leffa_simple_api.py::download_models",
        "Setting up models"
    )
    
    if not setup_success:
        print("\n⚠️  Model setup had issues, but continuing...")
        print("The models might already exist in the volume.")
    
    # Step 3: Test the deployment
    print("\n🧪 Step 3: Testing the deployment...")
    time.sleep(5)  # Give the deployment a moment to be ready
    
    test_success = run_command(
        "python test_fixed_deployment.py",
        "Testing deployment"
    )
    
    if test_success:
        print("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📋 NEXT STEPS:")
        print("1. The API is now deployed and ready to use")
        print("2. Use the test script to verify functionality")
        print("3. Update your frontend to use the new API")
        print("="*60)
        print("\n🔗 API ENDPOINTS:")
        print("• Main API: https://zebels-main--leffa-simple-viton-generate-virtual-tryon.modal.run")
        print("• Health Check: https://zebels-main--leffa-simple-viton-health-check.modal.run")
        print("="*60)
        return True
    else:
        print("\n⚠️  DEPLOYMENT COMPLETED BUT TESTS FAILED")
        print("The API is deployed but may have runtime issues.")
        print("Check the Modal logs for more details.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
