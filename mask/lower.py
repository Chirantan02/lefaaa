import os
import sys
import numpy as np
from PIL import Image, ImageDraw
import cv2 # Make sure OpenCV is installed: pip install opencv-python
import time
import argparse
import traceback
# --- Attempt to import required modules with proper path handling ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add the current directory to sys.path to enable relative imports
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from preprocess.dwpose import DWposeDetector
    from preprocess.humanparsing.run_parsing import Parsing
    from src.utils_mask_contour import get_mask_location # Core strategy
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Check paths for 'preprocess.dwpose', 'preprocess.humanparsing', 'src.utils_mask_contour'.")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    # Don't exit, let the calling function handle the error
    raise ImportError(f"Required modules not found: {e}")

# --- CONFIRMED Constants ---
LABEL_PANTS = 6; LABEL_SKIRT = 5; LABEL_DRESS = 7
LABEL_LEFT_SHOE = 9; LABEL_RIGHT_SHOE = 10
LABEL_LEFT_LEG = 12; LABEL_RIGHT_LEG = 13

CONFIDENCE_THRESHOLD = 0.1

# === Helper Functions (resize_image, get_keypoint - same as before) ===
def resize_image(img, target_size=768):
    width, height = img.size;
    if width == 0 or height == 0: return img
    scale = target_size / min(width, height);
    nw, nh = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    return img.resize((nw, nh), Image.LANCZOS)

def get_keypoint(keypoints_data, index, img_width, img_height, name=""):
    point = None; reason = "Not checked"
    if keypoints_data is None: reason = "Keypoints data is None"
    elif index >= keypoints_data.shape[0] or keypoints_data.shape[1] < 3: reason = f"Invalid shape {keypoints_data.shape}"
    else: kp = keypoints_data[index]; confidence = kp[2];
    if confidence > CONFIDENCE_THRESHOLD: x = int(np.clip(kp[0], 0, img_width - 1)); y = int(np.clip(kp[1], 0, img_height - 1)); point = (x, y); reason = f"OK (Conf: {confidence:.2f})"
    else: reason = f"Low confidence ({confidence:.2f})"
    return point

# === Main Generation Function (Improved Lower Body) ===

def generate_lower_body_with_shoes_mask_improved(
    image_path, model_root, output_dir, device='cuda', target_size=768,
    offset_top=0, offset_bottom=0, offset_left=0, offset_right=0,
    closing_kernel_size=9, # Proven effective size from backup
    closing_iterations=3, # Reduced iterations from backup
    opening_kernel_size=5, # Smaller kernel from backup
    opening_iterations=1,  # Maintain light smoothing
    dilate_shoes=False,    # Option to dilate shoes for better coverage
    horizontal_expansion=15, # Horizontal expansion (sides only)
    vertical_expansion=0    # Vertical expansion (set to 0 to avoid masking shoes)
    ):
    """
    Generates an IMPROVED lower body mask (waist to ankle, INCLUDING shoes),
    with stronger morphology for better smoothing and gap filling.
    """
    print(f"\n--- Starting IMPROVED Lower Body (With Shoes) Mask Generation ---")
    start_time_total = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # --- 1. Initialization ---
    print("Initializing models..."); t0=time.time()
    dwprocessor = None; parsing_model = None
    init_device = device
    gpu_error = False # Initialize gpu_error
    try: # (Robust initialization with CPU fallback)
        # Check if model_root already points to FitDiT models or if we need to append the path
        if os.path.exists(os.path.join(model_root, 'dwpose')):
            # model_root already points to the correct directory
            dwprocessor = DWposeDetector(model_root=model_root, device=device)
            parsing_model = Parsing(model_root=model_root, device=device)
        else:
            # Try the nested path structure
            nested_model_root = os.path.join(model_root, 'mask', 'model', 'FitDiT')
            if os.path.exists(os.path.join(nested_model_root, 'dwpose')):
                dwprocessor = DWposeDetector(model_root=nested_model_root, device=device)
                parsing_model = Parsing(model_root=nested_model_root, device=device)
            else:
                # Fallback to original model_root
                dwprocessor = DWposeDetector(model_root=model_root, device=device)
                parsing_model = Parsing(model_root=model_root, device=device)
    except Exception as e:
        error_msg = str(e).lower()
        gpu_error = ('cuda' in device and ('out of memory' in error_msg or 'bad allocation' in error_msg))
        if gpu_error and init_device != 'cpu':
            print(f"WARNING: GPU Error ({e}). Falling back to CPU.")
            device = 'cpu'
            try:
                dwprocessor = DWposeDetector(model_root=model_root, device=device)
                parsing_model = Parsing(model_root=model_root, device=device)
            except Exception as cpu_e:
                print(f"FATAL: Failed init on CPU: {cpu_e}")
                raise cpu_e
        else:
            print(f"FATAL: Model init failed: {e}")
            raise e
    if dwprocessor is None or parsing_model is None: raise RuntimeError("Model init failed.")
    print(f"Models initialized ({time.time()-t0:.2f}s).")

    # --- 2. Load/Prep Image ---
    print("Loading image..."); t0=time.time()
    try: vton_img = Image.open(image_path).convert("RGB"); original_size = vton_img.size; vton_img_det = resize_image(vton_img, target_size=target_size); det_width, det_height = vton_img_det.size
    except Exception as e: raise RuntimeError(f"Image load/resize failed: {e}")
    print(f"Image loaded ({time.time()-t0:.2f}s). Size: {original_size} -> {det_width}x{det_height}")

    # --- 3. Pose Detection ---
    print("Detecting pose..."); t0=time.time()
    candidate = None
    try: pose_input_np = np.array(vton_img_det)[:, :, ::-1].copy(); _, _, _, candidate = dwprocessor(pose_input_np);
    except Exception as e: print(f"Pose Detection Warning: {e}.")
    if candidate is None or candidate.shape[0] == 0: print("Pose Detection Warning: No candidates found.")
    else: candidate = candidate[0]; candidate[:, 0] = np.clip(candidate[:, 0], 0, det_width - 1); candidate[:, 1] = np.clip(candidate[:, 1], 0, det_height - 1); print(f"Pose OK ({time.time()-t0:.2f}s).")

    # --- 4. Human Parsing ---
    print("Parsing image..."); t0=time.time()
    model_parse_pil = None; model_parse_np = None
    try: model_parse_pil, _ = parsing_model(vton_img_det); model_parse_np = np.array(model_parse_pil)
    except Exception as e: raise RuntimeError(f"Human Parsing Error: {e}")
    print(f"Parsing OK ({time.time()-t0:.2f}s).")

    # --- 5. Generate Base Lower Body Mask (Strategy 0) ---
    print("Generating base lower body mask (Strategy 0)..."); t0=time.time()
    lower_body_mask_det_pil = None
    try:
        if model_parse_pil is None or candidate is None: raise RuntimeError("Parsing/Pose result missing.")
        offsets = {'top': offset_top, 'bottom': offset_bottom, 'left': offset_left, 'right': offset_right}
        lower_body_mask_det_pil, _ = get_mask_location(
             'Lower-body', model_parse_pil, candidate, det_width, det_height,
             offsets['top'], offsets['bottom'], offsets['left'], offsets['right']
         )
        if lower_body_mask_det_pil: print(f"Base 'Lower-body' mask OK ({time.time()-t0:.2f}s).")
        else: print("Warning: get_mask_location returned None."); lower_body_mask_det_pil = Image.new('L', (det_width, det_height), 0)
    except Exception as e: print(f"ERROR base lower mask: {e}"); lower_body_mask_det_pil = Image.new('L', (det_width, det_height), 0)
    lower_body_mask_det_np = np.array(lower_body_mask_det_pil.convert('L'))

    # --- 6. Extract Leg and Shoe Masks from Parsing ---
    print("Extracting leg and shoe segments...")
    left_leg_mask_np = (model_parse_np == LABEL_LEFT_LEG).astype(np.uint8) * 255
    right_leg_mask_np = (model_parse_np == LABEL_RIGHT_LEG).astype(np.uint8) * 255
    legs_mask_np = cv2.bitwise_or(left_leg_mask_np, right_leg_mask_np)

    left_shoe_mask_np = (model_parse_np == LABEL_LEFT_SHOE).astype(np.uint8) * 255
    right_shoe_mask_np = (model_parse_np == LABEL_RIGHT_SHOE).astype(np.uint8) * 255
    shoes_mask_np = cv2.bitwise_or(left_shoe_mask_np, right_shoe_mask_np)

    # --- IMPROVEMENT: Optional Shoe Dilation ---
    if dilate_shoes:
        shoe_kernel_dilate = np.ones((5,5), np.uint8) # Dilation kernel (if used)
        shoes_mask_np = cv2.dilate(shoes_mask_np, shoe_kernel_dilate, iterations=1)
        print(f"   Dilated shoe mask. Found {np.sum(shoes_mask_np > 0)} shoe pixels.")
    else:
        print(f"   Using exact shoe mask. Found {np.sum(shoes_mask_np > 0)} shoe pixels.")

    # --- 7. Combine Lower Body + Legs, REMOVING Shoes ---
    print("Combining lower body/legs and removing shoes...")
    # Combine lower body and legs then remove shoes
    combined_lower_leg_np = cv2.bitwise_or(lower_body_mask_det_np, legs_mask_np)
    shoes_mask_inv_np = cv2.bitwise_not(shoes_mask_np)
    mask_before_morph_np = cv2.bitwise_and(combined_lower_leg_np, shoes_mask_inv_np)

    # --- IMPROVEMENT: Stronger Morphology with Directional Dilation ---
    print(f"Applying Morphology: Directional Dilation -> Closing (Kernel:{closing_kernel_size}, Iter:{closing_iterations}), Opening (Kernel:{opening_kernel_size}, Iter:{opening_iterations})...")

    # 1. Initial Dilation: Create a minimal mask fit with horizontal-only expansion
    # Use a horizontal-only kernel for initial dilation to avoid expanding downward
    print(f"   Step 1: Horizontal-only Initial Dilation (H-expansion: {horizontal_expansion}, V-expansion: {vertical_expansion})")

    # Start with basic dilation for connectivity
    initial_dilation_kernel_size = 5 # Smaller initial kernel for basic connectivity
    initial_dilation_iterations = 2
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (initial_dilation_kernel_size, initial_dilation_kernel_size))
    dilated_mask_np = cv2.dilate(mask_before_morph_np, dilation_kernel, iterations=initial_dilation_iterations)

    # Apply horizontal-only expansion if specified
    if horizontal_expansion > 0:
        print(f"   Applying horizontal-only expansion: {horizontal_expansion} pixels")
        h_kernel = np.ones((1, horizontal_expansion * 2 + 1), np.uint8)
        dilated_mask_np = cv2.dilate(dilated_mask_np, h_kernel, iterations=1)

    # Apply vertical expansion only if explicitly requested (should be 0 to avoid masking shoes)
    if vertical_expansion > 0:
        print(f"   Applying vertical expansion: {vertical_expansion} pixels")
        v_kernel = np.ones((vertical_expansion * 2 + 1, 1), np.uint8)
        dilated_mask_np = cv2.dilate(dilated_mask_np, v_kernel, iterations=1)

    # 2. Closing: Fill gaps between shorts/pants and legs, smooth internal holes
    print(f"   Step 2: Closing (Kernel: {closing_kernel_size}, Iter: {closing_iterations})")
    # Ensure kernel size is odd
    actual_closing_kernel_size = closing_kernel_size
    if actual_closing_kernel_size % 2 == 0:
        actual_closing_kernel_size += 1
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (actual_closing_kernel_size, actual_closing_kernel_size))
    closed_mask_np = cv2.morphologyEx(dilated_mask_np, cv2.MORPH_CLOSE, closing_kernel, iterations=closing_iterations)

    # 3. Opening: Remove small noise/protrusions introduced by dilation/closing
    #    Crucially, ensure the opening kernel is smaller than the closing kernel
    #    to avoid undoing the gap filling.
    actual_opening_kernel_size = min(opening_kernel_size, actual_closing_kernel_size - 2) # Ensure smaller than closing
    actual_opening_kernel_size = max(3, actual_opening_kernel_size) # Ensure at least 3x3
    if actual_opening_kernel_size % 2 == 0: actual_opening_kernel_size += 1 # Ensure odd
    print(f"   Step 3: Opening (Adjusted Kernel: {actual_opening_kernel_size}, Iter: {opening_iterations})")
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (actual_opening_kernel_size, actual_opening_kernel_size))
    final_mask_np = cv2.morphologyEx(closed_mask_np, cv2.MORPH_OPEN, opening_kernel, iterations=opening_iterations)

    # Additional horizontal-only dilation for final refinement
    if horizontal_expansion > 0:
        post_h_kernel = np.ones((1, 5), np.uint8)  # Horizontal-only kernel
        final_mask_np = cv2.dilate(final_mask_np, post_h_kernel, iterations=1)
        print("Final horizontal-only dilation applied.")
    else:
        # Minimal uniform dilation if no horizontal expansion specified
        post_dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask_np = cv2.dilate(final_mask_np, post_dilation_kernel, iterations=1)
        print("Minimal uniform dilation applied.")

    # --- 8. Resize and Save Final Mask ---
    final_mask_pil = Image.fromarray(final_mask_np).resize(original_size, Image.NEAREST)

    if np.sum(np.array(final_mask_pil)) > 0:
        # Include morphology parameters in filename for comparison
        morph_suffix = f"h{horizontal_expansion}v{vertical_expansion}_c{closing_kernel_size}i{closing_iterations}_o{actual_opening_kernel_size}i{opening_iterations}"
        save_path = os.path.join(output_dir, f'{base_name}_lower_leg_no_shoes_improved_{morph_suffix}_mask.png')
        final_mask_pil.save(save_path)
        print(f"Successfully saved Improved Lower Body mask to: {save_path}")

        # Create and save overlay visualization
        try:
            original_img = Image.open(image_path).convert('RGB')
            overlay_img = original_img.copy()
            overlay_r = Image.new('RGB', overlay_img.size, (255, 0, 0))
            mask_overlay = final_mask_pil.resize(overlay_img.size)
            overlay_img.paste(overlay_r, (0, 0), mask_overlay)
            overlay_path = os.path.join(output_dir, f'{base_name}_lower_leg_no_shoes_improved_{morph_suffix}_overlay.png')
            overlay_img.save(overlay_path)
            print(f"Successfully saved overlay visualization to: {overlay_path}")
        except Exception as e:
            print(f"Warning: Failed to create overlay: {e}")

        status = "OK"
    else:
        print("Warning: Final improved mask is empty. Not saving.")
        status = "Empty"

    print(f"--- Total Time: {time.time() - start_time_total:.2f} seconds ---")
    return {'status': status, 'mask': final_mask_pil if status=='OK' else None}


# === Main Execution Block ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate IMPROVED Lower Body mask (waist to ankle, including shoes).')
    # (Standard Arguments)
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image file.')
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save the generated mask.')
    parser.add_argument('--model_root', required=True, type=str, help='Root directory containing DWpose and Parsing model weights.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run models on.')
    parser.add_argument('--target_size', type=int, default=768, help='Target size for detection/parsing.')
    # (Offsets for Strategy 0)
    parser.add_argument('--offset_top', type=int, default=0); parser.add_argument('--offset_bottom', type=int, default=0)
    parser.add_argument('--offset_left', type=int, default=0); parser.add_argument('--offset_right', type=int, default=0)
    # (Morphology Tuning Arguments)
    parser.add_argument('--closing_kernel_size', type=int, default=9, help='Kernel size (odd) for morphological closing.')
    parser.add_argument('--closing_iterations', type=int, default=3, help='Iterations for morphological closing.')
    parser.add_argument('--opening_kernel_size', type=int, default=5, help='Kernel size (odd) for morphological opening.')
    parser.add_argument('--opening_iterations', type=int, default=1, help='Iterations for morphological opening.')
    parser.add_argument('--dilate_shoes', action='store_true', help='Dilate shoe mask for better coverage (default: False).')
    parser.add_argument('--horizontal_expansion', type=int, default=15, help='Horizontal expansion in pixels (sides only).')
    parser.add_argument('--vertical_expansion', type=int, default=0, help='Vertical expansion in pixels (0 to avoid masking shoes).')

    args = parser.parse_args()

    # (Path validation)
    if not os.path.exists(args.image_path): print(f"ERROR: Input image not found: '{args.image_path}'"); sys.exit(1)
    if not os.path.isdir(args.model_root): print(f"ERROR: Model root not found: '{args.model_root}'"); sys.exit(1)

    try:
        result = generate_lower_body_with_shoes_mask_improved( # Call the improved function
            image_path=args.image_path, model_root=args.model_root, output_dir=args.output_dir,
            device=args.device, target_size=args.target_size,
            offset_top=args.offset_top, offset_bottom=args.offset_bottom,
            offset_left=args.offset_left, offset_right=args.offset_right,
            closing_kernel_size=args.closing_kernel_size, closing_iterations=args.closing_iterations,
            opening_kernel_size=args.opening_kernel_size, opening_iterations=args.opening_iterations,
            dilate_shoes=args.dilate_shoes,
            horizontal_expansion=args.horizontal_expansion,
            vertical_expansion=args.vertical_expansion
        )
        print(f"\nResult Status: {result['status']}")
    except Exception as e: print(f"\n--- SCRIPT FAILED ---"); traceback.print_exc(); sys.exit(1)

    print("\nScript finished.")

    #RUNNNING INSTRUCTIONS:

    ## JUST RUN : python mask/lower.py --image_path "C:\Users\chira\Downloads\image (50).png" --output_dir output --model_root . --closing_kernel_size 9 --closing_iterations 3 --opening_kernel_size 5 --opening_iterations 1 --horizontal_expansion 20 --vertical_expansion 0