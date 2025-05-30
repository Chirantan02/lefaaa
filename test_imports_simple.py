#!/usr/bin/env python3
"""
Simple test script to verify import fixes.
"""

import os
import sys

print("=== ComfyUI Leffa Import Test ===")
print(f"Current directory: {os.getcwd()}")
print(f"Python version: {sys.version}")

# Test basic imports
try:
    import numpy as np
    print("✅ numpy imported successfully")
except ImportError as e:
    print(f"❌ numpy import failed: {e}")

try:
    import cv2
    print("✅ opencv imported successfully")
except ImportError as e:
    print(f"❌ opencv import failed: {e}")

try:
    from PIL import Image
    print("✅ PIL imported successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

# Test mask directory imports
mask_dir = os.path.join(os.getcwd(), "mask")
print(f"\nMask directory: {mask_dir}")
print(f"Mask directory exists: {os.path.exists(mask_dir)}")

if os.path.exists(mask_dir):
    sys.path.insert(0, mask_dir)
    
    try:
        from preprocess.dwpose import DWposeDetector
        print("✅ DWposeDetector imported successfully")
    except ImportError as e:
        print(f"❌ DWposeDetector import failed: {e}")
    
    try:
        from preprocess.humanparsing.run_parsing import Parsing
        print("✅ Parsing imported successfully")
    except ImportError as e:
        print(f"❌ Parsing import failed: {e}")
    
    try:
        from src.utils_mask_contour import get_mask_location
        print("✅ utils_mask_contour imported successfully")
    except ImportError as e:
        print(f"❌ utils_mask_contour import failed: {e}")
    
    sys.path.remove(mask_dir)

print("\n=== Test Complete ===")
