#!/usr/bin/env python3
"""
Diagnostic script for ComfyUI Leffa import issues.
This script helps identify and troubleshoot import problems.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test importing a module and return result."""
    try:
        if package_name:
            __import__(package_name)
            print(f"✅ {module_name}: Successfully imported")
            return True
        else:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: Successfully imported")
            return True
    except ImportError as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: Unexpected error - {e}")
        return False

def test_file_import(file_path, module_name):
    """Test importing a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            print(f"❌ {module_name}: Could not create spec from {file_path}")
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"✅ {module_name}: Successfully imported from {file_path}")
        return True
    except Exception as e:
        print(f"❌ {module_name}: Failed to import from {file_path} - {e}")
        return False

def check_python_path():
    """Check Python path configuration."""
    print("🔍 Python Path Configuration:")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print("\nPython path entries:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")

def check_mask_modules():
    """Check mask module imports."""
    print("\n🔍 Testing mask module imports...")

    current_dir = Path(__file__).parent
    mask_dir = current_dir / "mask"

    print(f"Mask directory: {mask_dir}")
    print(f"Mask directory exists: {mask_dir.exists()}")

    if not mask_dir.exists():
        print("❌ Mask directory not found!")
        return False

    # Add mask directory to path
    mask_dir_str = str(mask_dir)
    if mask_dir_str not in sys.path:
        sys.path.insert(0, mask_dir_str)
        print(f"Added to Python path: {mask_dir_str}")

    # Test individual module imports
    modules_to_test = [
        ("preprocess.dwpose", mask_dir / "preprocess" / "dwpose" / "__init__.py"),
        ("preprocess.humanparsing.run_parsing", mask_dir / "preprocess" / "humanparsing" / "run_parsing.py"),
        ("src.utils_mask_contour", mask_dir / "src" / "utils_mask_contour.py"),
        ("src.utils_mask_contour_special", mask_dir / "src" / "utils_mask_contour_special.py"),
    ]

    success_count = 0
    for module_name, file_path in modules_to_test:
        print(f"\nTesting {module_name}:")
        print(f"  File path: {file_path}")
        print(f"  File exists: {file_path.exists()}")

        if file_path.exists():
            if test_import(module_name):
                success_count += 1
        else:
            print(f"❌ {module_name}: File not found")

    # Remove from path
    if mask_dir_str in sys.path:
        sys.path.remove(mask_dir_str)

    return success_count == len(modules_to_test)

def check_core_dependencies():
    """Check core Python dependencies."""
    print("\n🔍 Testing core dependencies...")

    dependencies = [
        ("numpy", None),
        ("cv2", None),
        ("PIL", None),
        ("torch", None),
        ("onnxruntime", None),
        ("diffusers", None),
        ("transformers", None),
        ("huggingface_hub", None),
        ("skimage", None),
    ]

    success_count = 0
    for module_name, package_name in dependencies:
        if test_import(module_name, package_name):
            success_count += 1

    print(f"\n📊 Core dependencies: {success_count}/{len(dependencies)} successful")
    return success_count == len(dependencies)

def check_model_files():
    """Check if model files exist."""
    print("\n🔍 Checking model files...")

    # FitDiT models
    fitdit_base = Path("C:/Users/chira/Downloads/ninja downlaod/ComfyUI_windows_portable_nvidia_1/ComfyUI_windows_portable/ComfyUI/models/FitDiT_models")

    fitdit_files = [
        "dwpose/dw-ll_ucoco_384.onnx",
        "dwpose/yolox_l.onnx",
        "humanparsing/parsing_atr.onnx",
        "humanparsing/parsing_lip.onnx"
    ]

    print(f"FitDiT base path: {fitdit_base}")
    print(f"FitDiT base exists: {fitdit_base.exists()}")

    fitdit_success = 0
    for file_path in fitdit_files:
        full_path = fitdit_base / file_path
        exists = full_path.exists()
        print(f"  {file_path}: {'✅' if exists else '❌'}")
        if exists:
            fitdit_success += 1

    # Leffa models
    leffa_base = Path("C:/Users/chira/Downloads/ninja downlaod/ComfyUI_windows_portable_nvidia_1/ComfyUI_windows_portable/ComfyUI/models/leffa")

    leffa_files = [
        "virtual_tryon.pth",
        "virtual_tryon_dc.pth"
    ]

    print(f"\nLeffa base path: {leffa_base}")
    print(f"Leffa base exists: {leffa_base.exists()}")

    leffa_success = 0
    for file_path in leffa_files:
        full_path = leffa_base / file_path
        exists = full_path.exists()
        print(f"  {file_path}: {'✅' if exists else '❌'}")
        if exists:
            leffa_success += 1

    print(f"\n📊 Model files: FitDiT {fitdit_success}/{len(fitdit_files)}, Leffa {leffa_success}/{len(leffa_files)}")

    return fitdit_success == len(fitdit_files) and leffa_success == len(leffa_files)

def main():
    """Main diagnostic function."""
    print("🔧 ComfyUI Leffa Import Diagnostics")
    print("=" * 50)

    # Check Python environment
    check_python_path()

    # Check core dependencies
    deps_ok = check_core_dependencies()

    # Check mask modules
    mask_ok = check_mask_modules()

    # Check model files
    models_ok = check_model_files()

    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    print(f"Core dependencies: {'✅ OK' if deps_ok else '❌ ISSUES'}")
    print(f"Mask modules: {'✅ OK' if mask_ok else '❌ ISSUES'}")
    print(f"Model files: {'✅ OK' if models_ok else '❌ ISSUES'}")

    if deps_ok and mask_ok and models_ok:
        print("\n🎉 All checks passed! The system should work correctly.")
    else:
        print("\n⚠️  Issues detected. Please address the problems above.")
        print("💡 Run setup_dependencies.py to install missing packages.")
        print("📥 See DOWNLOAD_INSTRUCTIONS.md for model download steps.")

if __name__ == "__main__":
    main()
