import os
import sys
import numpy as np
from PIL import Image
import cv2
import time
import argparse
import traceback
import importlib.util
import torch
import onnxruntime as ort
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gc
from functools import lru_cache

# Check for GPU availability
GPU_AVAILABLE = torch.cuda.is_available()
CUDA_PROVIDER_AVAILABLE = 'CUDAExecutionProvider' in ort.get_available_providers()

if GPU_AVAILABLE and CUDA_PROVIDER_AVAILABLE:
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Available ONNX Runtime providers: {ort.get_available_providers()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    if not GPU_AVAILABLE:
        print("WARNING: No GPU detected by PyTorch. Using CPU.")
    elif not CUDA_PROVIDER_AVAILABLE:
        print("WARNING: CUDA provider not available for ONNX Runtime. Install onnxruntime-gpu package.")

# --- Import functions from upper.py and lower.py ---
def import_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not find module {module_name} at {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Global cache for model instances
model_cache = {}

# Cache for processed images
image_cache = {}

# Image caching functions
@lru_cache(maxsize=8)
def get_image_hash(image_path):
    """Generate a hash for an image file to use as a cache key"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def preload_image(image_path, target_size=768):
    """Preload and cache an image to avoid repeated loading"""
    img_hash = get_image_hash(image_path)
    if img_hash not in image_cache:
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")
            original_size = img.size
            # Resize for processing
            resized_img = img
            if min(img.size) != target_size:
                scale = target_size / min(img.size)
                new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
                resized_img = img.resize(new_size, Image.LANCZOS)

            # Cache both original and resized versions
            image_cache[img_hash] = {
                'original': img,
                'resized': resized_img,
                'original_size': original_size,
                'resized_size': resized_img.size
            }
            print(f"Image cached: {image_path}")
        except Exception as e:
            print(f"Error preloading image: {e}")
            return None
    return image_cache[img_hash]

# Import the upper.py and lower.py modules with proper error handling
try:
    upper_module = import_module_from_file("upper", os.path.join(current_dir, "upper.py"))
    lower_module = import_module_from_file("lower", os.path.join(current_dir, "lower.py"))
    print("Successfully imported upper.py and lower.py modules")
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    # Don't exit, let the calling function handle the error
    raise ImportError(f"Required modules not found: {e}")

def generate_upper_body_mask_wrapper(args):
    """Wrapper function for upper body mask generation to use with ThreadPoolExecutor"""
    return upper_module.generate_upper_body_mask(**args)

def generate_lower_body_mask_wrapper(args):
    """Wrapper function for lower body mask generation to use with ThreadPoolExecutor"""
    return lower_module.generate_lower_body_with_shoes_mask_improved(**args)

def generate_full_body_mask(
    image_path, model_root, output_dir, device='cuda', target_size=768,
    offset_top=0, offset_bottom=0, offset_left=0, offset_right=0,
    closing_kernel_size=9,
    closing_iterations=3,
    opening_kernel_size=5,
    opening_iterations=1,
    dilate_shoes=False,
    horizontal_expansion=15,
    vertical_expansion=0,
    gap_fill_kernel_size=5,  # Kernel size for filling gaps between upper and lower masks
    gap_fill_iterations=2,   # Iterations for filling gaps between upper and lower masks
    force_gpu=True,          # Force GPU usage if available
    parallel=True,           # Generate masks in parallel
    skip_overlay=False,      # Skip overlay generation for faster processing
    half_precision=False,    # Use half precision (FP16) for faster computation
    optimize_memory=True,    # Optimize memory usage
    mask_type='full'         # Type of mask to generate: 'full', 'upper', or 'lower'
    ):
    """
    Generates a full body mask by combining upper body and lower body masks.
    Uses parallel processing to generate both masks simultaneously for better performance.
    """
    print(f"\n--- Starting Full Body Mask Generation ---")
    start_time_total = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Configure GPU settings
    effective_device = 'cuda' if (force_gpu and GPU_AVAILABLE and CUDA_PROVIDER_AVAILABLE) else device
    if effective_device == 'cuda':
        # Clear GPU memory before running
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
            # Set memory optimization options
            if optimize_memory:
                # Set PyTorch to release memory more aggressively
                torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of available memory
                # Enable TensorFloat32 for faster computation on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Enable cuDNN benchmark mode for faster convolutions
                torch.backends.cudnn.benchmark = True

        print(f"Using GPU ({torch.cuda.get_device_name(0)}) for mask generation")
        if half_precision:
            print("Using half precision (FP16) for faster computation")
    else:
        print(f"Using {effective_device} for mask generation")

    # Preload and cache the image if needed
    try:
        preload_image(image_path, target_size)
    except Exception as e:
        print(f"Warning: Image preloading failed: {e}")

    # Initialize variables for results
    upper_mask = None
    lower_mask = None

    # Prepare arguments for mask generation functions
    upper_args = {
        'image_path': image_path,
        'model_root': model_root,
        'output_dir': output_dir,
        'device': effective_device,
        'target_size': target_size,
        'offset_top': offset_top,
        'offset_bottom': offset_bottom,
        'offset_left': offset_left,
        'offset_right': offset_right
    }

    lower_args = {
        'image_path': image_path,
        'model_root': model_root,
        'output_dir': output_dir,
        'device': effective_device,
        'target_size': target_size,
        'offset_top': offset_top,
        'offset_bottom': offset_bottom,
        'offset_left': offset_left,
        'offset_right': offset_right,
        'closing_kernel_size': closing_kernel_size,
        'closing_iterations': closing_iterations,
        'opening_kernel_size': opening_kernel_size,
        'opening_iterations': opening_iterations,
        'dilate_shoes': dilate_shoes,
        'horizontal_expansion': horizontal_expansion,
        'vertical_expansion': vertical_expansion
    }

    # Generate masks based on mask_type
    if mask_type == 'upper' or mask_type == 'full':
        print("Generating upper body mask...")
        upper_result = upper_module.generate_upper_body_mask(**upper_args)
        if upper_result['status'] != 'OK':
            print(f"Warning: Upper body mask generation failed with status: {upper_result['status']}")
            return {'status': 'Failed', 'mask': None}
        upper_mask = upper_result['mask']
        print(f"Upper body mask generated successfully.")

        # If only upper mask is requested, return it
        if mask_type == 'upper':
            return {'status': 'OK', 'mask': upper_mask, 'mask_type': 'upper'}

    if mask_type == 'lower' or mask_type == 'full':
        # Clear GPU memory before running lower body mask generation
        if effective_device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("Generating lower body mask...")
        lower_result = lower_module.generate_lower_body_with_shoes_mask_improved(**lower_args)
        if lower_result['status'] != 'OK':
            print(f"Warning: Lower body mask generation failed with status: {lower_result['status']}")
            return {'status': 'Failed', 'mask': None}
        lower_mask = lower_result['mask']
        print(f"Lower body mask generated successfully.")

        # If only lower mask is requested, return it
        if mask_type == 'lower':
            return {'status': 'OK', 'mask': lower_mask, 'mask_type': 'lower'}

    # For full mask, we need both upper and lower masks
    if mask_type == 'full':
        print(f"Both masks generated successfully.")

    # --- 3. Combine Masks ---
    print("Combining upper and lower body masks...")

    # Sanity check - we should have both masks at this point
    if upper_mask is None or lower_mask is None:
        print(f"Error: Missing masks for full body generation. Upper: {upper_mask is not None}, Lower: {lower_mask is not None}")
        return {'status': 'Failed', 'mask': None}

    # Check if OpenCV CUDA is available
    use_gpu_cv = False
    if effective_device == 'cuda':
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                use_gpu_cv = True
                print("Using GPU for mask processing")
            else:
                print("OpenCV CUDA support not available (no CUDA devices), using CPU for mask processing")
        except Exception as e:
            print(f"OpenCV CUDA support not available: {e}, using CPU for mask processing")
            print("To enable OpenCV CUDA support, reinstall OpenCV with CUDA: pip install opencv-python-headless-cuda")

    # Convert masks to numpy arrays
    upper_mask_np = np.array(upper_mask)
    lower_mask_np = np.array(lower_mask)

    # Ensure masks have the same dimensions
    if upper_mask_np.shape != lower_mask_np.shape:
        print(f"Warning: Mask dimensions don't match. Upper: {upper_mask_np.shape}, Lower: {lower_mask_np.shape}")
        # Get original image dimensions
        original_img = Image.open(image_path)
        original_size = original_img.size  # (width, height)

        # Resize both masks to the original image dimensions
        upper_mask = upper_mask.resize(original_size, Image.NEAREST)
        lower_mask = lower_mask.resize(original_size, Image.NEAREST)

        # Update numpy arrays
        upper_mask_np = np.array(upper_mask)
        lower_mask_np = np.array(lower_mask)

        print(f"Resized both masks to original image dimensions: {original_size}")

    # Combine masks using bitwise OR
    combined_mask_np = cv2.bitwise_or(upper_mask_np, lower_mask_np)

    # Fill any gaps between upper and lower body
    if gap_fill_kernel_size > 0 and gap_fill_iterations > 0:
        print(f"Filling gaps between upper and lower body (Kernel: {gap_fill_kernel_size}, Iterations: {gap_fill_iterations})...")
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_fill_kernel_size, gap_fill_kernel_size))

        # Use GPU for morphological operations if available
        if use_gpu_cv:
            try:
                # Upload to GPU
                gpu_mask = cv2.cuda_GpuMat()
                gpu_mask.upload(combined_mask_np)

                # Perform morphological operation on GPU
                gpu_mask = cv2.cuda.morphologyEx(gpu_mask, cv2.MORPH_CLOSE, gap_kernel, iterations=gap_fill_iterations)

                # Download result back to CPU
                combined_mask_np = gpu_mask.download()
                print("Gap filling performed on GPU")
            except Exception as e:
                print(f"GPU morphology failed: {e}, falling back to CPU")
                combined_mask_np = cv2.morphologyEx(combined_mask_np, cv2.MORPH_CLOSE, gap_kernel, iterations=gap_fill_iterations)
        else:
            # CPU fallback
            combined_mask_np = cv2.morphologyEx(combined_mask_np, cv2.MORPH_CLOSE, gap_kernel, iterations=gap_fill_iterations)

    # Convert back to PIL Image
    combined_mask = Image.fromarray(combined_mask_np)

    # --- 4. Save Mask ---
    # Create a descriptive filename based on mask type
    if mask_type == 'upper':
        save_path = os.path.join(output_dir, f'{base_name}_upper_body_mask.png')
    elif mask_type == 'lower':
        params_suffix = f"h{horizontal_expansion}v{vertical_expansion}_c{closing_kernel_size}i{closing_iterations}"
        save_path = os.path.join(output_dir, f'{base_name}_lower_body_{params_suffix}_mask.png')
    else:  # full
        params_suffix = f"h{horizontal_expansion}v{vertical_expansion}_c{closing_kernel_size}i{closing_iterations}_g{gap_fill_kernel_size}i{gap_fill_iterations}"
        save_path = os.path.join(output_dir, f'{base_name}_full_body_{params_suffix}_mask.png')

    combined_mask.save(save_path)
    print(f"Successfully saved {mask_type.capitalize()} Body mask to: {save_path}")

    # --- 5. Create and Save Overlay (for visualization) ---
    if not skip_overlay:
        try:
            # Use cached image if available
            try:
                img_hash = get_image_hash(image_path)
                if img_hash in image_cache and 'original' in image_cache[img_hash]:
                    original_img = image_cache[img_hash]['original']
                else:
                    original_img = Image.open(image_path).convert('RGB')
            except Exception:
                # Fallback if caching fails
                original_img = Image.open(image_path).convert('RGB')

            # Ensure mask is the same size as the original image
            if original_img.size != combined_mask.size:
                combined_mask = combined_mask.resize(original_img.size, Image.NEAREST)

            overlay_img = create_overlay(original_img, combined_mask)

            # Create overlay filename based on mask type
            if mask_type == 'upper':
                overlay_path = os.path.join(output_dir, f'{base_name}_upper_body_overlay.png')
            elif mask_type == 'lower':
                overlay_path = os.path.join(output_dir, f'{base_name}_lower_body_{params_suffix}_overlay.png')
            else:  # full
                overlay_path = os.path.join(output_dir, f'{base_name}_full_body_{params_suffix}_overlay.png')

            overlay_img.save(overlay_path)
            print(f"Successfully saved {mask_type.capitalize()} Body overlay visualization to: {overlay_path}")
        except Exception as e:
            print(f"Warning: Failed to create overlay: {e}")
            traceback.print_exc()
    else:
        print("Skipping overlay generation for faster processing")

    print(f"--- Total Time: {time.time() - start_time_total:.2f} seconds ---")
    return {'status': 'OK', 'mask': combined_mask}

def create_overlay(image, mask):
    """Create a visualization overlay of the mask on the original image."""
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')

    # Convert to numpy arrays
    img_np = np.array(image)
    mask_np = np.array(mask)

    # Ensure mask is binary
    if len(mask_np.shape) > 2:
        mask_np = mask_np[:, :, 0]  # Take first channel if it's RGB

    # Create red overlay
    red_overlay = np.zeros_like(img_np)
    red_overlay[:, :, 0] = 255  # Red channel

    # Apply overlay where mask is non-zero
    mask_bool = mask_np > 0
    mask_3ch = np.stack([mask_bool] * 3, axis=-1)
    result = np.where(mask_3ch, cv2.addWeighted(img_np, 0.7, red_overlay, 0.3, 0), img_np)

    return Image.fromarray(result)

# === Main Execution Block ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Full Body mask by combining upper and lower body masks.')
    # Standard Arguments
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image file.')
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save the generated mask.')
    parser.add_argument('--model_root', required=True, type=str, help='Root directory containing DWpose and Parsing model weights.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run models on.')
    parser.add_argument('--force_gpu', action='store_true', help='Force GPU usage if available, regardless of device setting.')
    parser.add_argument('--parallel', action='store_true', help='Generate masks in parallel for better performance.')
    parser.add_argument('--sequential', action='store_true', help='Generate masks sequentially (disables parallel processing).')
    parser.add_argument('--target_size', type=int, default=768, help='Target size for detection/parsing.')
    parser.add_argument('--skip_overlay', action='store_true', help='Skip overlay generation for faster processing.')
    parser.add_argument('--half_precision', action='store_true', help='Use half precision (FP16) for faster computation.')
    parser.add_argument('--optimize_memory', action='store_true', help='Optimize memory usage for better performance.')
    parser.add_argument('--mask_type', type=str, default='full', choices=['full', 'upper', 'lower'],
                        help='Type of mask to generate: full body, upper body only, or lower body only.')

    # Upper Body Mask Arguments (none needed anymore)

    # Lower Body Mask Arguments
    parser.add_argument('--offset_top', type=int, default=0, help='Offset Top for mask positioning.')
    parser.add_argument('--offset_bottom', type=int, default=0, help='Offset Bottom for mask positioning.')
    parser.add_argument('--offset_left', type=int, default=0, help='Offset Left for mask positioning.')
    parser.add_argument('--offset_right', type=int, default=0, help='Offset Right for mask positioning.')
    parser.add_argument('--closing_kernel_size', type=int, default=9, help='Kernel size for morphological closing.')
    parser.add_argument('--closing_iterations', type=int, default=3, help='Iterations for morphological closing.')
    parser.add_argument('--opening_kernel_size', type=int, default=5, help='Kernel size for morphological opening.')
    parser.add_argument('--opening_iterations', type=int, default=1, help='Iterations for morphological opening.')
    parser.add_argument('--dilate_shoes', action='store_true', help='Dilate shoe mask for better coverage (default: False).')
    parser.add_argument('--horizontal_expansion', type=int, default=15, help='Horizontal expansion in pixels (sides only).')
    parser.add_argument('--vertical_expansion', type=int, default=0, help='Vertical expansion in pixels (0 to avoid masking shoes).')

    # Gap Filling Arguments
    parser.add_argument('--gap_fill_kernel_size', type=int, default=5, help='Kernel size for filling gaps between upper and lower masks.')
    parser.add_argument('--gap_fill_iterations', type=int, default=2, help='Iterations for filling gaps between upper and lower masks.')

    args = parser.parse_args()

    # Path validation
    if not os.path.exists(args.image_path):
        print(f"ERROR: Input image not found: '{args.image_path}'")
        sys.exit(1)
    if not os.path.isdir(args.model_root):
        print(f"ERROR: Model root not found: '{args.model_root}'")
        sys.exit(1)

    try:
        result = generate_full_body_mask(
            image_path=args.image_path,
            model_root=args.model_root,
            output_dir=args.output_dir,
            device=args.device,
            target_size=args.target_size,
            offset_top=args.offset_top,
            offset_bottom=args.offset_bottom,
            offset_left=args.offset_left,
            offset_right=args.offset_right,

            closing_kernel_size=args.closing_kernel_size,
            closing_iterations=args.closing_iterations,
            opening_kernel_size=args.opening_kernel_size,
            opening_iterations=args.opening_iterations,
            dilate_shoes=args.dilate_shoes,
            horizontal_expansion=args.horizontal_expansion,
            vertical_expansion=args.vertical_expansion,
            gap_fill_kernel_size=args.gap_fill_kernel_size,
            gap_fill_iterations=args.gap_fill_iterations,
            force_gpu=args.force_gpu,
            parallel=args.parallel,
            skip_overlay=args.skip_overlay,
            half_precision=args.half_precision,
            optimize_memory=args.optimize_memory,
            mask_type=args.mask_type
        )
        print(f"\nResult Status: {result['status']}")
    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        traceback.print_exc()
        sys.exit(1)

    print("\nScript finished.")
