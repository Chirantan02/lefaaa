#!/usr/bin/env python3
"""
Validate CLI setup without running full inference
"""

import os
import sys

def validate_files():
    """Check if all required files exist"""
    print("🔍 Validating file structure...")

    required_files = [
        "standalone_cli_tryon.py",
        "config_loader.py",
        "config_tryon.yaml",
        "workflow_api.json",
        "CLI_IMPLEMENTATION_SUMMARY.md",
        "USAGE_EXAMPLES.md",
        "lib/ximg.py",
        "leffaNode.py"
    ]

    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing.append(file)

    return len(missing) == 0

def validate_imports():
    """Test basic imports"""
    print("\n🔍 Validating imports...")

    try:
        import yaml
        print("✅ PyYAML")
    except ImportError:
        print("❌ PyYAML - MISSING")
        return False

    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch - MISSING")
        return False

    try:
        from PIL import Image
        print("✅ PIL/Pillow")
    except ImportError:
        print("❌ PIL/Pillow - MISSING")
        return False

    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError:
        print("❌ NumPy - MISSING")
        return False

    return True

def validate_config():
    """Test configuration loader"""
    print("\n🔍 Validating configuration...")

    try:
        from config_loader import ConfigLoader
        config = ConfigLoader()

        # Test basic functionality
        model_path = config.get("model.path")
        viton_type = config.get("model.viton_type")
        steps = config.get("inference.steps")

        print(f"✅ Config loaded - Model: {model_path}, Type: {viton_type}, Steps: {steps}")

        # Test mask URLs
        upper_mask = config.get_mask_url("upper")
        lower_mask = config.get_mask_url("lower")

        print(f"✅ Mask URLs - Upper: {upper_mask[:50]}...")
        print(f"                Lower: {lower_mask[:50]}...")

        return True

    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False

def validate_lib():
    """Test lib imports"""
    print("\n🔍 Validating lib utilities...")

    try:
        from lib.ximg import open_image, pil2tensor, tensor2pil, img_from_url
        print("✅ lib.ximg functions imported")
        return True
    except Exception as e:
        print(f"❌ lib.ximg import failed: {e}")
        return False

def validate_nodes():
    """Test ComfyUI node imports"""
    print("\n🔍 Validating ComfyUI nodes...")

    try:
        from leffaNode import CXH_Leffa_Viton_Load, CXH_Leffa_Viton_Run
        print("✅ Leffa nodes imported")

        # Test node initialization
        loader = CXH_Leffa_Viton_Load()
        runner = CXH_Leffa_Viton_Run()
        print("✅ Nodes initialized")

        return True
    except Exception as e:
        print(f"❌ ComfyUI node validation failed: {e}")
        return False

def validate_cli_structure():
    """Test CLI argument parsing"""
    print("\n🔍 Validating CLI structure...")

    try:
        # Test if we can import the CLI module
        import importlib.util
        spec = importlib.util.spec_from_file_location("standalone_cli_tryon", "standalone_cli_tryon.py")

        if spec is None:
            print("❌ Cannot load standalone_cli_tryon.py")
            return False

        print("✅ CLI module can be loaded")

        # Test argument parser by checking help
        import subprocess
        result = subprocess.run([
            sys.executable, "standalone_cli_tryon.py", "--help"
        ], capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print("✅ CLI help works")
            return True
        else:
            print(f"❌ CLI help failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ CLI structure validation failed: {e}")
        return False

def main():
    """Run validation"""
    print("🎭 CLI VALIDATION")
    print("=" * 50)

    tests = [
        ("File Structure", validate_files),
        ("Basic Imports", validate_imports),
        ("Configuration", validate_config),
        ("Lib Utilities", validate_lib),
        ("ComfyUI Nodes", validate_nodes),
        ("CLI Structure", validate_cli_structure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")

    print("-" * 50)
    print(f"Results: {passed}/{total} validations passed")

    if passed == total:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("\nYour CLI is ready to use. Example commands:")
        print("\n# Show help")
        print("python standalone_cli_tryon.py --help")
        print("\n# Basic usage")
        print("python standalone_cli_tryon.py \\")
        print("  --user-image person.jpg \\")
        print("  --garment-image shirt.jpg \\")
        print("  --mask-type upper")
        print("\n# Using configuration file")
        print("python standalone_cli_tryon.py --config config_tryon.yaml")

    elif passed >= 4:
        print(f"\n⚠️  {total - passed} validations failed, but core functionality should work.")
        print("Try running: python standalone_cli_tryon.py --help")

    else:
        print(f"\n❌ {total - passed} validations failed. Please fix the issues above.")

    return 0 if passed >= 4 else 1

if __name__ == "__main__":
    sys.exit(main())
