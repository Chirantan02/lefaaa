# Mask generation module for ComfyUI Leffa
# This module provides advanced mask generation capabilities for virtual try-on

import os
import sys

# Add the current directory to Python path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import the main mask generation functions
try:
    from .upper import generate_upper_body_mask
    from .lower import generate_lower_body_with_shoes_mask_improved
    from .full import generate_full_body_mask
    
    __all__ = [
        'generate_upper_body_mask',
        'generate_lower_body_with_shoes_mask_improved', 
        'generate_full_body_mask'
    ]
    
    print("✅ Mask generation modules imported successfully")
    
except ImportError as e:
    print(f"⚠️ Warning: Some mask generation modules could not be imported: {e}")
    __all__ = []
