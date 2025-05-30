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
    # This function is the core strategy
    from src.utils_mask_contour import get_mask_location
except ImportError as e:
    print(f"ERROR: Failed to import required modules: {e}")
    print("Check paths for 'preprocess.dwpose', 'preprocess.humanparsing', 'src.utils_mask_contour'.")
    print(f"Current directory: {current_dir}")
    print(f"Python path: {sys.path}")
    # Don't exit, let the calling function handle the error
    raise ImportError(f"Required modules not found: {e}")

# --- CONFIRMED Constants ---
# Keypoint indices (COCO format) - Neck and Shoulders are important here
KP_NECK = 1
KP_R_SHOULDER = 2
KP_L_SHOULDER = 5

# Parsing labels (ATR dataset standard)
LABEL_FACE = 1
LABEL_UPPER_CLOTHES = 4
LABEL_ARMS = 7  # Arms label - useful for sleeveless tops
# --- End of CONFIRMED Constants ---

CONFIDENCE_THRESHOLD = 0.15 # Slightly higher threshold for critical points like neck/shoulders

# === Helper Functions ===

def resize_image(img, target_size=768):
    """Resizes an image while maintaining aspect ratio."""
    width, height = img.size
    if width == 0 or height == 0: return img
    scale = target_size / min(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), Image.LANCZOS)

def get_keypoint(keypoints_data, index, img_width, img_height, name=""):
    """Safely retrieves and validates keypoint coordinates. Handles shapes (N, 2) and (N, 3)."""
    point = None; reason = "Not checked"
    if keypoints_data is None: reason = "Keypoints data is None"
    # Allow shape (N, 2) or (N, 3)
    elif index >= keypoints_data.shape[0] or keypoints_data.shape[1] < 2: reason = f"Invalid index or shape {keypoints_data.shape}"
    else:
        kp = keypoints_data[index]
        # Check if confidence data is available (shape[1] >= 3)
        if keypoints_data.shape[1] >= 3:
            confidence = kp[2]
            if confidence > CONFIDENCE_THRESHOLD:
                x = int(np.clip(kp[0], 0, img_width - 1)); y = int(np.clip(kp[1], 0, img_height - 1))
                point = (x, y); reason = f"OK (Conf: {confidence:.2f})"
            else: reason = f"Low confidence ({confidence:.2f})"
        else: # Assume valid if shape is (N, 2) and index is valid
            x = int(np.clip(kp[0], 0, img_width - 1)); y = int(np.clip(kp[1], 0, img_height - 1))
            point = (x, y); reason = "OK (No confidence data)"

    if point is None and name: print(f"      Point '{name}' (Idx {index}): Failed ({reason})")
    return point

# === Main Generation Function ===

def generate_upper_body_mask(
    image_path, model_root, output_dir, device='cuda', target_size=768,
    offset_top=0, offset_bottom=0, offset_left=0, offset_right=0
    ):
    """
    Generates an upper body mask using the tank top specific approach with enhanced neck coverage.
    This is the only strategy available as it has been proven to work best.
    """
    print(f"\n--- Starting Upper Body Mask Generation for: {os.path.basename(image_path)} ---")
    start_time_total = time.time()
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # --- 1. Initialization ---
    print("Initializing models..."); t0=time.time()
    dwprocessor = None; parsing_model = None
    init_device = device
    try:
        dwprocessor = DWposeDetector(model_root, device=init_device)
        parsing_model = Parsing(model_root, device=init_device)
        print(f"Models initialized ({time.time()-t0:.2f}s).")
    except Exception as e:
        print(f"ERROR during model initialization: {e}")
        if 'CUDA' in str(e) and init_device != 'cpu':
            print("Attempting to fall back to CPU...")
            init_device = 'cpu'
            try:
                dwprocessor = DWposeDetector(model_root, device=init_device)
                parsing_model = Parsing(model_root, device=init_device)
                print(f"Models initialized on CPU ({time.time()-t0:.2f}s).")
            except Exception as cpu_e:
                raise RuntimeError(f"Model init failed on both GPU and CPU: {cpu_e}") from cpu_e
        else:
            raise RuntimeError(f"Model init failed: {e}") from e

    # --- 2. Load/Prep Image ---
    print("Loading image..."); t0=time.time()
    try:
        vton_img = Image.open(image_path).convert("RGB")
        original_size = vton_img.size
        vton_img_det = resize_image(vton_img, target_size=target_size)
        det_width, det_height = vton_img_det.size
    except Exception as e:
        raise RuntimeError(f"Image load/resize failed: {e}")
    print(f"Image loaded ({time.time()-t0:.2f}s). Size: {original_size} -> {det_width}x{det_height}")

    # --- 3. Pose Detection ---
    print("Detecting pose..."); t0=time.time()
    candidate = None
    try:
        pose_input_np = np.array(vton_img_det)[:, :, ::-1].copy()
        _, _, _, candidate = dwprocessor(pose_input_np)
    except Exception as e:
        print(f"Pose Detection Warning: {e}.")
    if candidate is None or candidate.shape[0] == 0:
        print("Pose Detection Warning: No candidates found.")
    else:
        candidate = candidate[0]
        candidate[:, 0] = np.clip(candidate[:, 0], 0, det_width - 1)
        candidate[:, 1] = np.clip(candidate[:, 1], 0, det_height - 1)
        print(f"Pose OK ({time.time()-t0:.2f}s).")

    # --- 4. Human Parsing ---
    print("Parsing image..."); t0=time.time()
    model_parse_pil = None
    try:
        model_parse_pil, _ = parsing_model(vton_img_det)
    except Exception as e:
        raise RuntimeError(f"Human Parsing Error: {e}")
    print(f"Parsing OK ({time.time()-t0:.2f}s).")

    # Convert PIL Image to numpy array for contour functions
    image_np = np.array(vton_img_det)
    parsing_np = np.array(model_parse_pil)

    print("Generating mask using tank top approach with enhanced neck coverage")

    # Tank top specific approach with enhanced neck coverage
    from src.utils_mask_contour_special import get_img_agnostic_tank_top_contour
    mask, mask_gray = get_img_agnostic_tank_top_contour(
        model_parse_pil, candidate, offset_top, offset_bottom, offset_left, offset_right
    )

    # Add explicit neck region handling with improved coverage
    if candidate is not None:
        # Convert mask to numpy array for processing
        mask_np = np.array(mask)

        # Get key points
        neck_kp = get_keypoint(candidate, KP_NECK, det_width, det_height, "Neck")
        l_shoulder_kp = get_keypoint(candidate, KP_L_SHOULDER, det_width, det_height, "L Shoulder")
        r_shoulder_kp = get_keypoint(candidate, KP_R_SHOULDER, det_width, det_height, "R Shoulder")

        if neck_kp and l_shoulder_kp and r_shoulder_kp:
            # Create a neck region mask
            neck_mask = np.zeros_like(mask_np)

            # Calculate neck width based on shoulder distance - FURTHER INCREASED WIDTH
            shoulder_width = abs(r_shoulder_kp[0] - l_shoulder_kp[0])
            neck_width = int(shoulder_width * 0.7)  # 70% of shoulder width (increased from 60%)

            # Calculate neck height - FURTHER INCREASED HEIGHT
            shoulder_y = (l_shoulder_kp[1] + r_shoulder_kp[1]) / 2
            neck_height = int(abs(neck_kp[1] - shoulder_y) * 2.0)  # 200% of distance (increased from 180%)

            # Define neck polygon points - EXPANDED AREA
            neck_left = max(0, int(neck_kp[0] - neck_width/2))
            neck_right = min(det_width-1, int(neck_kp[0] + neck_width/2))
            # Extend further above neck keypoint to ensure full coverage
            neck_top = max(0, int(neck_kp[1] - neck_height * 0.7))  # Extend more above neck keypoint (increased from 0.6)
            neck_bottom = min(det_height-1, int(neck_kp[1] + neck_height * 0.9))  # Extend more below (increased from 0.8)

            # Draw neck region
            neck_points = np.array([
                [neck_left, neck_bottom],
                [neck_left, neck_top],
                [neck_right, neck_top],
                [neck_right, neck_bottom]
            ], dtype=np.int32)
            cv2.fillPoly(neck_mask, [neck_points], 255)

            # Add a connecting region between neck and shoulders - FURTHER IMPROVED CONNECTION
            connect_mask = np.zeros_like(mask_np)

            # Create a wider trapezoid to ensure better connection
            shoulder_mid_y = (l_shoulder_kp[1] + r_shoulder_kp[1]) / 2
            connect_points = np.array([
                [max(0, l_shoulder_kp[0] - int(shoulder_width * 0.15)), l_shoulder_kp[1]],  # Extend left shoulder more
                [neck_left - int(shoulder_width * 0.15), neck_bottom],  # Widen left neck connection more
                [neck_right + int(shoulder_width * 0.15), neck_bottom],  # Widen right neck connection more
                [min(det_width-1, r_shoulder_kp[0] + int(shoulder_width * 0.15)), r_shoulder_kp[1]]  # Extend right shoulder more
            ], dtype=np.int32)
            cv2.fillPoly(connect_mask, [connect_points], 255)

            # Add face mask to ensure complete coverage of neck-to-face transition - ENHANCED
            face_mask = np.zeros_like(mask_np)
            face_y = max(0, neck_kp[1] - int(neck_height * 0.9))  # Position higher above neck (increased from 0.8)
            face_radius = int(neck_width * 0.8)  # Larger size based on neck width (increased from 0.7)
            cv2.circle(face_mask, (neck_kp[0], face_y), face_radius, 255, -1)

            # Add an additional elliptical mask to better cover the transition area
            transition_mask = np.zeros_like(mask_np)
            transition_y = int((face_y + neck_top) / 2)  # Midpoint between face and neck top
            axes_length = (int(neck_width * 0.9), int(neck_height * 0.3))  # Width and height of ellipse
            cv2.ellipse(transition_mask, (neck_kp[0], transition_y), axes_length, 0, 0, 360, 255, -1)

            # Combine all masks
            combined_mask = cv2.bitwise_or(mask_np, neck_mask)
            combined_mask = cv2.bitwise_or(combined_mask, connect_mask)
            combined_mask = cv2.bitwise_or(combined_mask, face_mask)
            combined_mask = cv2.bitwise_or(combined_mask, transition_mask)  # Add the new transition mask

            # Clean up with stronger morphological operations
            # Use larger kernel and more iterations for better blending
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))  # Increased from (15,15)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=4)  # Increased from 3

            # Apply Gaussian blur to smooth the edges
            combined_mask = cv2.GaussianBlur(combined_mask, (9, 9), 3)
            # Threshold back to binary after blurring
            _, combined_mask = cv2.threshold(combined_mask, 127, 255, cv2.THRESH_BINARY)

            # Additional dilation to ensure full coverage
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)  # Increased from 1

            # Convert back to PIL Image
            mask = Image.fromarray(combined_mask)
            mask_gray = Image.fromarray(combined_mask)

    # --- 6. Save Results ---
    if np.sum(np.array(mask)) > 0:
        # Save mask
        mask_path = os.path.join(output_dir, f'{base_name}_upper_body_mask.png')
        mask.save(mask_path)
        print(f"Successfully saved upper body mask to: {mask_path}")

        # Create and save overlay visualization
        overlay = Image.fromarray(image_np)
        overlay_r = Image.new('RGB', overlay.size, (255, 0, 0))
        mask_overlay = mask.convert('L').resize(overlay.size)
        overlay.paste(overlay_r, (0, 0), mask_overlay)
        overlay_path = os.path.join(output_dir, f'{base_name}_upper_body_overlay.png')
        overlay.save(overlay_path)
        print(f"Successfully saved overlay visualization to: {overlay_path}")
        status = "OK"
    else:
        print("Warning: Generated mask is empty, creating basic rectangular mask.")
        # Get image dimensions from numpy array
        vton_img_np = np.array(vton_img_det)
        h, w = vton_img_np.shape[:2]
        # Create a basic rectangular mask
        mask = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask)
        # Draw a rectangle that covers roughly the upper body area
        width = w
        height = h
        top = int(height * 0.15)  # Start from 15% down
        bottom = int(height * 0.45)  # End at 45% down
        left = int(width * 0.2)  # Start from 20% in
        right = int(width * 0.8)  # End at 80% in
        draw.rectangle([left, top, right, bottom], fill=255)

        # Save the basic mask
        mask_path = os.path.join(output_dir, f'{base_name}_upper_body_mask.png')
        mask.save(mask_path)
        print(f"Saved basic rectangular mask to: {mask_path}")

        # Create and save overlay visualization
        overlay = Image.fromarray(vton_img_np)
        overlay_r = Image.new('RGB', (w, h), (255, 0, 0))
        mask_overlay = mask.resize((w, h))
        overlay.paste(overlay_r, (0, 0), mask_overlay)
        overlay_path = os.path.join(output_dir, f'{base_name}_upper_body_overlay.png')
        overlay.save(overlay_path)
        print(f"Saved basic rectangular overlay to: {overlay_path}")
        status = "BASIC"

    print(f"--- Total Time: {time.time() - start_time_total:.2f} seconds ---")
    return {'status': status, 'mask': mask if status=='OK' else None}


# === Main Execution Block ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Upper Body mask using tank top approach with enhanced neck coverage.')
    parser.add_argument('--image_path', required=True, type=str, help='Path to the input image file.')
    parser.add_argument('--output_dir', required=True, type=str, help='Directory to save the generated mask.')
    parser.add_argument('--model_root', required=True, type=str, help='Root directory containing DWpose and Parsing model weights.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to run models on (cuda or cpu).')
    parser.add_argument('--target_size', type=int, default=768, help='Target size for resizing image during detection/parsing.')
    parser.add_argument('--offset_top', type=int, default=0, help='Offset Top for mask generation.')
    parser.add_argument('--offset_bottom', type=int, default=0, help='Offset Bottom for mask generation.')
    parser.add_argument('--offset_left', type=int, default=0, help='Offset Left for mask generation.')
    parser.add_argument('--offset_right', type=int, default=0, help='Offset Right for mask generation.')
    args = parser.parse_args()

    # Path validation
    if not os.path.exists(args.image_path): print(f"ERROR: Input image not found: '{args.image_path}'"); sys.exit(1)
    if not os.path.isdir(args.model_root): print(f"ERROR: Model root not found: '{args.model_root}'"); sys.exit(1)

    try:
        result = generate_upper_body_mask(
            image_path=args.image_path, model_root=args.model_root, output_dir=args.output_dir,
            device=args.device, target_size=args.target_size,
            offset_top=args.offset_top, offset_bottom=args.offset_bottom,
            offset_left=args.offset_left, offset_right=args.offset_right
        )
        print(f"\nResult Status: {result['status']}")
    except Exception as e: print(f"\n--- SCRIPT FAILED ---"); traceback.print_exc(); sys.exit(1)

    print("\nScript finished.")