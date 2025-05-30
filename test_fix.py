#!/usr/bin/env python3
"""
Test script to verify the Leffa model download fixes.
"""

import os
import sys

# Add the current directory to Python path so we can import our modules
sys.path.insert(0, os.path.dirname(__file__))

def test_manual_model_detection():
    """Test if manual model detection works."""
    print("Testing manual model detection...")
    
    try:
        from lib.xmodel import get_manual_model_path
        
        # Test with a fake model ID
        model_id = "franciszzj/Leffa"
        result = get_manual_model_path(model_id)
        
        if result:
            print(f"✓ Found manual model at: {result}")
            return True
        else:
            print("✗ No manual model found (this is expected if not downloaded)")
            return False
            
    except Exception as e:
        print(f"✗ Error testing manual model detection: {e}")
        return False


def test_safe_download_function():
    """Test the safe download function (without actually downloading)."""
    print("Testing safe download function...")
    
    try:
        from lib.xmodel import download_hg_model_safe
        
        # Test with offline mode to avoid actual download
        model_id = "franciszzj/Leffa"
        
        try:
            result = download_hg_model_safe(model_id, offline_mode=True)
            print(f"✓ Found model at: {result}")
            return True
        except FileNotFoundError as e:
            print(f"✓ Offline mode working correctly: {str(e)[:100]}...")
            return True
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error importing safe download function: {e}")
        return False


def test_node_import():
    """Test if the updated node can be imported."""
    print("Testing node import...")
    
    try:
        # Mock folder_paths since it's not available in this context
        import sys
        import types
        
        # Create a mock folder_paths module
        folder_paths = types.ModuleType('folder_paths')
        folder_paths.models_dir = os.path.join(os.getcwd(), "models")
        sys.modules['folder_paths'] = folder_paths
        
        from leffaNode import CXH_Leffa_Viton_Load
        print("✓ Node imported successfully")
        
        # Test node instantiation
        node = CXH_Leffa_Viton_Load()
        print("✓ Node instantiated successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Error importing/instantiating node: {e}")
        return False


def main():
    """Run all tests."""
    print("Leffa Model Download Fix - Test Suite")
    print("=" * 50)
    
    tests = [
        test_manual_model_detection,
        test_safe_download_function,
        test_node_import
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
        print("-" * 30)
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! The fixes should work correctly.")
        print("\nNext steps:")
        print("1. Try running ComfyUI with the Leffa nodes")
        print("2. If download fails, use: python download_leffa_model.py")
        print("3. Or download manually from: https://huggingface.co/franciszzj/Leffa")
    else:
        print("✗ Some tests failed. Please check the error messages above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
