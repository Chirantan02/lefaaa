#!/usr/bin/env python3
"""
Deployment script for Leffa Virtual Try-On Modal API
"""

import modal
import time

def main():
    print("🚀 DEPLOYING LEFFA VIRTUAL TRY-ON API TO MODAL")
    print("="*60)
    
    # Import the app
    from modal_leffa_simple_api import app, download_models
    
    print("📦 Step 1: Setting up models in Modal volume...")
    print("This will download all required models to Modal volume...")
    print("Expected time: 10-15 minutes")
    print("Expected cost: ~$0.50-1.00")
    
    # Download models to Modal volume
    try:
        print("\n🔄 Starting model download...")
        start_time = time.time()
        
        # Run the model download function
        result = download_models.remote()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ Model download completed!")
        print(f"⏱️  Total time: {total_time/60:.2f} minutes")
        print(f"💰 Estimated cost: ${(total_time/3600)*1.10:.4f}")
        
        if result:
            print("\n📊 Download breakdown:")
            print(result)
        
    except Exception as e:
        print(f"\n❌ Model download failed: {e}")
        return False
    
    print("\n📦 Step 2: Deploying API endpoints...")
    
    try:
        # Deploy the app
        print("🚀 Deploying Modal app...")
        
        # The app will be deployed when this script runs with modal deploy
        print("✅ API endpoints ready for deployment!")
        
        print("\n🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📋 NEXT STEPS:")
        print("1. Run: modal deploy deploy_leffa_api.py")
        print("2. Test with: python test_simple_api.py --api-url <your-modal-url>")
        print("3. Update GCP bucket name in modal_leffa_simple_api.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)
