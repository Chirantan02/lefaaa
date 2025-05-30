import numpy as np
import os
import sys
import tempfile
import importlib.util
import torch

from PIL import Image

from .lib.ximg import *

current_folder = os.path.dirname(os.path.abspath(__file__))


class CXH_NEW_Advanced_Mask_Generator:
    """
    🚀 COMPLETELY NEW Advanced Mask Generator using the latest masking logic from mask/ folder.
    Supports upper, lower, and full body mask generation with advanced features.
    REPLACES: CXH_Leffa_Mask_Generator (old/deprecated)
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_type": (["full", "upper", "lower"], {"default": "full"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "target_size": ("INT", {"default": 768, "min": 256, "max": 1024, "step": 32}),
                "horizontal_expansion": ("INT", {"default": 15, "min": 0, "max": 50, "step": 1}),
                "vertical_expansion": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "closing_kernel_size": ("INT", {"default": 9, "min": 3, "max": 21, "step": 2}),
                "closing_iterations": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "opening_kernel_size": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2}),
                "opening_iterations": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "gap_fill_kernel_size": ("INT", {"default": 5, "min": 3, "max": 15, "step": 2}),
                "gap_fill_iterations": ("INT", {"default": 2, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "offset_top": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "offset_bottom": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "offset_left": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "offset_right": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "dilate_shoes": ("BOOLEAN", {"default": False}),
                "force_gpu": ("BOOLEAN", {"default": True}),
                "half_precision": ("BOOLEAN", {"default": False}),
                "optimize_memory": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "generate_mask"
    OUTPUT_NODE = False
    CATEGORY = "🚀 CXH/NEW_MASKING"

    def import_module_from_file(self, module_name, file_path):
        """Import a module from a file path"""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not find module {module_name} at {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def generate_mask(self, image, mask_type="full", device="cuda", target_size=768,
                     horizontal_expansion=15, vertical_expansion=0, closing_kernel_size=9,
                     closing_iterations=3, opening_kernel_size=5, opening_iterations=1,
                     gap_fill_kernel_size=5, gap_fill_iterations=2, offset_top=0,
                     offset_bottom=0, offset_left=0, offset_right=0, dilate_shoes=False,
                     force_gpu=True, half_precision=False, optimize_memory=True):

        try:
            print(f"🚀 NEW ADVANCED MASK GENERATOR - Generating {mask_type} mask with enhanced features!")

            # Convert tensor to PIL Image
            pil_image = tensor2pil(image)

            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                pil_image.save(temp_file.name)
                temp_image_path = temp_file.name

            # Create temporary output directory
            temp_output_dir = tempfile.mkdtemp()

            # Set model root path to the FitDiT models location
            model_root = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\FitDiT_models"

            # Add the mask directory to sys.path temporarily to allow imports
            mask_dir = os.path.join(current_folder, "mask")
            if mask_dir not in sys.path:
                sys.path.insert(0, mask_dir)

            try:
                # Import the specific mask generation module based on mask_type
                if mask_type == "upper":
                    try:
                        upper_module = self.import_module_from_file("upper", os.path.join(mask_dir, "upper.py"))
                        result = upper_module.generate_upper_body_mask(
                            image_path=temp_image_path,
                            model_root=model_root,
                            output_dir=temp_output_dir,
                            device=device,
                            target_size=target_size,
                            offset_top=offset_top,
                            offset_bottom=offset_bottom,
                            offset_left=offset_left,
                            offset_right=offset_right
                        )
                    except ImportError as ie:
                        print(f"🚨 Import error in upper mask generation: {ie}")
                        result = {'status': 'IMPORT_ERROR', 'mask': None, 'error': str(ie)}

                elif mask_type == "lower":
                    try:
                        lower_module = self.import_module_from_file("lower", os.path.join(mask_dir, "lower.py"))
                        result = lower_module.generate_lower_body_with_shoes_mask_improved(
                            image_path=temp_image_path,
                            model_root=model_root,
                            output_dir=temp_output_dir,
                            device=device,
                            target_size=target_size,
                            offset_top=offset_top,
                            offset_bottom=offset_bottom,
                            offset_left=offset_left,
                            offset_right=offset_right,
                            closing_kernel_size=closing_kernel_size,
                            closing_iterations=closing_iterations,
                            opening_kernel_size=opening_kernel_size,
                            opening_iterations=opening_iterations,
                            dilate_shoes=dilate_shoes,
                            horizontal_expansion=horizontal_expansion,
                            vertical_expansion=vertical_expansion
                        )
                    except ImportError as ie:
                        print(f"🚨 Import error in lower mask generation: {ie}")
                        result = {'status': 'IMPORT_ERROR', 'mask': None, 'error': str(ie)}

                else:  # full
                    try:
                        full_module = self.import_module_from_file("full", os.path.join(mask_dir, "full.py"))
                        result = full_module.generate_full_body_mask(
                            image_path=temp_image_path,
                            model_root=model_root,
                            output_dir=temp_output_dir,
                            device=device,
                            target_size=target_size,
                            offset_top=offset_top,
                            offset_bottom=offset_bottom,
                            offset_left=offset_left,
                            offset_right=offset_right,
                            closing_kernel_size=closing_kernel_size,
                            closing_iterations=closing_iterations,
                            opening_kernel_size=opening_kernel_size,
                            opening_iterations=opening_iterations,
                            dilate_shoes=dilate_shoes,
                            horizontal_expansion=horizontal_expansion,
                            vertical_expansion=vertical_expansion,
                            gap_fill_kernel_size=gap_fill_kernel_size,
                            gap_fill_iterations=gap_fill_iterations,
                            force_gpu=force_gpu,
                            parallel=True,
                            skip_overlay=True,  # Skip overlay for faster processing
                            half_precision=half_precision,
                            optimize_memory=optimize_memory,
                            mask_type=mask_type
                        )
                    except ImportError as ie:
                        print(f"🚨 Import error in full mask generation: {ie}")
                        result = {'status': 'IMPORT_ERROR', 'mask': None, 'error': str(ie)}

            finally:
                # Remove the mask directory from sys.path
                if mask_dir in sys.path:
                    sys.path.remove(mask_dir)

            # Clean up temporary files
            os.unlink(temp_image_path)

            if result['status'] == 'OK' and result['mask'] is not None:
                # Convert PIL mask to tensor
                mask_array = np.array(result['mask'].convert('L'))
                # Normalize to 0-1 range
                mask_array = mask_array.astype(np.float32) / 255.0
                # Convert to tensor with proper dimensions
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)  # Add batch dimension

                print(f"✅ Successfully generated {mask_type} mask using NEW ADVANCED GENERATOR!")
                return (mask_tensor,)
            else:
                print(f"❌ Mask generation failed with status: {result.get('status', 'Unknown')}")
                # Return empty mask
                mask_array = np.zeros((pil_image.height, pil_image.width), dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
                return (mask_tensor,)

        except Exception as e:
            print(f"❌ Error in NEW ADVANCED mask generation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty mask on error
            pil_image = tensor2pil(image)
            mask_array = np.zeros((pil_image.height, pil_image.width), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            return (mask_tensor,)


class CXH_NEW_Advanced_Pose_Preprocessor:
    """
    🚀 COMPLETELY NEW Advanced Pose Preprocessor using the latest DWPose detection from mask/ folder.
    REPLACES: CXH_Leffa_Pose_Preprocessor (old/deprecated)
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "target_size": ("INT", {"default": 512, "min": 256, "max": 1024, "step": 32}),
                "include_hands": ("BOOLEAN", {"default": True}),
                "include_face": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "process_pose"
    OUTPUT_NODE = False
    CATEGORY = "🚀 CXH/NEW_MASKING"

    def import_module_from_file(self, module_name, file_path):
        """Import a module from a file path"""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not find module {module_name} at {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def process_pose(self, image, device="cuda", target_size=512, include_hands=True, include_face=False):
        try:
            print(f"🚀 NEW ADVANCED POSE PREPROCESSOR - Processing pose with enhanced features!")

            # Convert tensor to PIL Image
            pil_image = tensor2pil(image)

            # Create temporary file for the image
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                pil_image.save(temp_file.name)
                temp_image_path = temp_file.name

            # Set model root path to the FitDiT models location
            model_root = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\FitDiT_models"

            # Add the mask directory to sys.path temporarily to allow imports
            mask_dir = os.path.join(current_folder, "mask")
            if mask_dir not in sys.path:
                sys.path.insert(0, mask_dir)

            try:
                # Import DWPose detector
                from preprocess.dwpose import DWposeDetector

                # Initialize DWPose detector
                dwprocessor = DWposeDetector(model_root, device=device)

                # Resize image for processing
                original_size = pil_image.size
                if min(pil_image.size) != target_size:
                    scale = target_size / min(pil_image.size)
                    new_size = (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale))
                    resized_image = pil_image.resize(new_size, Image.LANCZOS)
                else:
                    resized_image = pil_image

                # Convert to numpy array for processing
                pose_input_np = np.array(resized_image)[:, :, ::-1].copy()  # RGB to BGR

                # Process pose
                pose_result = dwprocessor(pose_input_np)

                # Extract pose image (the first element is typically the rendered pose)
                if isinstance(pose_result, tuple) and len(pose_result) > 0:
                    pose_image_np = pose_result[0]  # Get the first result which should be the pose image

                    # Convert back to RGB if needed
                    if len(pose_image_np.shape) == 3 and pose_image_np.shape[2] == 3:
                        pose_image_np = pose_image_np[:, :, ::-1]  # BGR to RGB

                    # Convert to PIL Image
                    pose_image = Image.fromarray(pose_image_np.astype(np.uint8))

                    # Resize back to original size
                    pose_image = pose_image.resize(original_size, Image.LANCZOS)

                    # Convert to tensor
                    pose_tensor = pil2tensor(pose_image)

                    print("✅ Successfully processed pose using NEW ADVANCED PREPROCESSOR!")
                    return (pose_tensor,)
                else:
                    print("❌ Pose processing failed, returning original image")
                    return (image,)

            finally:
                # Remove the mask directory from sys.path
                if mask_dir in sys.path:
                    sys.path.remove(mask_dir)
                # Clean up temporary files
                os.unlink(temp_image_path)

        except Exception as e:
            print(f"❌ Error in NEW ADVANCED pose processing: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return (image,)
