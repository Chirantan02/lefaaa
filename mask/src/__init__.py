# Source utilities for mask generation

try:
    from .utils_mask_contour import get_mask_location
    from .utils_mask_contour_special import get_img_agnostic_tank_top_contour
    
    __all__ = [
        'get_mask_location',
        'get_img_agnostic_tank_top_contour'
    ]
    
except ImportError as e:
    print(f"Warning: Could not import mask contour utilities: {e}")
    __all__ = []
