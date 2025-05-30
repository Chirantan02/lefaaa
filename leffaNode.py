
import numpy as np
import os
import sys
import tempfile
import importlib.util
import torch

from PIL import Image

from .lib.xmodel import download_hg_model_safe

from .leffa.transform import LeffaTransform
from .leffa.model import LeffaModel
from .leffa.inference import LeffaInference

from .lib.ximg import *

current_folder = os.path.dirname(os.path.abspath(__file__))

class CXH_Leffa_Viton_Load:

    def __init__(self):
        self.vt_inference = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (['franciszzj/Leffa'],),
                "viton_type": (['hd',"dc"],),
            }
        }

    RETURN_TYPES = ("CXH_Leffa_Viton_Load",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH/IDM"

    def gen(self, model, viton_type):
        try:
            # Try to download or find the model - use 'leffa' as exDir to match the expected path structure
            model_path = download_hg_model_safe(model, exDir='leffa', offline_mode=False, max_retries=3)

            # Check required files exist with multiple possible locations
            inpainting_paths = [
                os.path.join(model_path, "stable-diffusion-inpainting"),
                os.path.join(model_path, "inpainting"),
                os.path.join(model_path, "sd-inpainting"),
                model_path  # Sometimes the model_path itself contains the inpainting model
            ]

            inpainting = None
            for path in inpainting_paths:
                if os.path.exists(path):
                    # Check if it contains model files
                    if os.path.isdir(path):
                        # Look for diffusers model structure (scheduler, unet, vae subdirectories)
                        diffusers_subdirs = ['scheduler', 'unet', 'vae']
                        if all(os.path.exists(os.path.join(path, subdir)) for subdir in diffusers_subdirs):
                            inpainting = path
                            break
                        # Fallback: Look for common model files
                        model_files = ['model_index.json', 'config.json', 'pytorch_model.bin', 'diffusion_pytorch_model.bin']
                        if any(os.path.exists(os.path.join(path, f)) for f in model_files):
                            inpainting = path
                            break
                    else:
                        # If it's a file, check if it's a model file
                        if path.endswith(('.bin', '.safetensors', '.ckpt')):
                            inpainting = path
                            break

            if inpainting is None:
                # Create a fallback message with download instructions
                error_msg = f"""
Required 'stable-diffusion-inpainting' model not found in any of these locations:
{chr(10).join(f'  - {path}' for path in inpainting_paths)}

Please download the Leffa model files manually:
1. Go to: https://huggingface.co/franciszzj/Leffa
2. Download the 'stable-diffusion-inpainting' folder
3. Place it in: {model_path}

See DOWNLOAD_INSTRUCTIONS.md for detailed instructions.
"""
                raise FileNotFoundError(error_msg)

            if viton_type == 'hd':
                virtual_tryon = os.path.join(model_path, "virtual_tryon.pth")
            else:
                virtual_tryon = os.path.join(model_path, "virtual_tryon_dc.pth")

            if not os.path.exists(virtual_tryon):
                error_msg = f"""
Required virtual try-on model not found at: {virtual_tryon}

Please download the Leffa model files manually:
1. Go to: https://huggingface.co/franciszzj/Leffa
2. Download 'virtual_tryon.pth' and 'virtual_tryon_dc.pth'
3. Place them in: {model_path}

See DOWNLOAD_INSTRUCTIONS.md for detailed instructions.
"""
                raise FileNotFoundError(error_msg)

            print(f"Loading Leffa model from: {model_path}")
            print(f"Inpainting model: {inpainting}")
            print(f"Virtual try-on model: {virtual_tryon}")

            vt_model = LeffaModel(
                pretrained_model_name_or_path=inpainting,
                pretrained_model=virtual_tryon,
            )
            self.vt_inference = LeffaInference(model=vt_model)

            print("Leffa model loaded successfully!")
            return (self,)

        except Exception as e:
            error_msg = f"Failed to load Leffa model: {str(e)}"
            print(error_msg)
            # Instead of raising, we could return a dummy object or handle gracefully
            # For now, let's raise with a more informative message
            raise RuntimeError(error_msg)



class CXH_Leffa_Viton_Run:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("CXH_Leffa_Viton_Load",),
                "model":("IMAGE",),
                "cloth":("IMAGE",),
                "pose":("IMAGE",),
                "mask":("MASK",),
                "steps":("INT", {"default": 20, "min": 1, "max": 100, "step": 0.01}),
                "cfg":("FLOAT", {"default": 2.5, "min": 1, "max": 50, "step": 0.01}),
                "seed": ("INT", {"default": 656545, "min": 0, "max": 1000000}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "gen"
    OUTPUT_NODE = False
    CATEGORY = "CXH/IDM"

    def gen(self, pipe,model,cloth,pose,mask,steps,cfg,seed):

        src_image = tensor2pil(model)
        ref_image = tensor2pil(cloth)
        pose_image = tensor2pil(pose)
        original_size = src_image.size

        # src_image = resize_and_center(src_image, 768, 1024)
        # ref_image = resize_and_center(ref_image, 768, 1024)

        src_image = src_image.convert("RGB")
        transform = LeffaTransform()
        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [pose_image],

        }
        data = transform(data)
        output = pipe.vt_inference(data,num_inference_steps=steps,guidance_scale = cfg,seed = seed)
        gen_image = output["generated_image"][0]
        gen_image = gen_image.resize(original_size, Image.NEAREST)
        img = pil2tensor(gen_image)

        return (img,)


class CXH_Leffa_Mask_Generator:
    """
    ComfyUI node for generating masks using the new masking logic from mask/ folder.
    Supports upper, lower, and full body mask generation.
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
    CATEGORY = "CXH/IDM"

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
                        print(f"Import error in upper mask generation: {ie}")
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
                        print(f"Import error in lower mask generation: {ie}")
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
                        print(f"Import error in full mask generation: {ie}")
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

                print(f"Successfully generated {mask_type} mask")
                return (mask_tensor,)
            elif result['status'] == 'IMPORT_ERROR':
                error_msg = f"""
Mask generation failed due to missing dependencies:
{result.get('error', 'Unknown import error')}

This usually means one of the following is missing:
1. DWpose model files in: {model_root}/dwpose/
2. Human parsing model files in: {model_root}/humanparsing/
3. Required Python packages (opencv-python, onnxruntime, etc.)

Please check the model files and dependencies.
"""
                print(error_msg)
                # Return empty mask
                mask_array = np.zeros((pil_image.height, pil_image.width), dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
                return (mask_tensor,)
            else:
                print(f"Mask generation failed with status: {result.get('status', 'Unknown')}")
                if 'error' in result:
                    print(f"Error details: {result['error']}")
                # Return empty mask
                mask_array = np.zeros((pil_image.height, pil_image.width), dtype=np.float32)
                mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
                return (mask_tensor,)

        except Exception as e:
            print(f"Error in mask generation: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty mask on error
            pil_image = tensor2pil(image)
            mask_array = np.zeros((pil_image.height, pil_image.width), dtype=np.float32)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)
            return (mask_tensor,)


class CXH_Leffa_Pose_Preprocessor:
    """
    ComfyUI node for preprocessing pose detection using DWpose.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "target_size": ("INT", {"default": 768, "min": 256, "max": 1024, "step": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION = "process_pose"
    OUTPUT_NODE = False
    CATEGORY = "CXH/IDM"

    def import_module_from_file(self, module_name, file_path):
        """Import a module from a file path"""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None:
            raise ImportError(f"Could not find module {module_name} at {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def process_pose(self, image, device="cuda", target_size=768):
        try:
            # Convert tensor to PIL Image
            pil_image = tensor2pil(image)

            # Set model root path to the FitDiT models location
            model_root = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\FitDiT_models"

            # Add the mask directory to sys.path temporarily to allow imports
            mask_dir = os.path.join(current_folder, "mask")
            if mask_dir not in sys.path:
                sys.path.insert(0, mask_dir)

            try:
                # Import DWpose detector
                from preprocess.dwpose import DWposeDetector

                # Initialize pose detector
                dwprocessor = DWposeDetector(model_root, device=device)

                # Resize image for processing
                def resize_image(img, target_size=768):
                    width, height = img.size
                    if width == 0 or height == 0:
                        return img
                    scale = target_size / min(width, height)
                    new_width = max(1, int(round(width * scale)))
                    new_height = max(1, int(round(height * scale)))
                    return img.resize((new_width, new_height), Image.LANCZOS)

                resized_image = resize_image(pil_image, target_size)

                # Convert to numpy for pose detection
                pose_input_np = np.array(resized_image)[:, :, ::-1].copy()  # RGB to BGR

                # Detect pose
                pose_image, body, ori_score, candidate = dwprocessor(pose_input_np)

                # Convert back to RGB and PIL
                pose_image_rgb = pose_image[:, :, ::-1]  # BGR to RGB
                pose_pil = Image.fromarray(pose_image_rgb)

                # Resize back to original size
                pose_pil = pose_pil.resize(pil_image.size, Image.LANCZOS)

                # Convert to tensor
                pose_tensor = pil2tensor(pose_pil)

                print("Successfully processed pose")
                return (pose_tensor,)

            except ImportError as ie:
                print(f"Import error in pose processing: {ie}")
                # Return original image if pose processing fails
                return (image,)

            finally:
                # Remove the mask directory from sys.path
                if mask_dir in sys.path:
                    sys.path.remove(mask_dir)

        except Exception as e:
            print(f"Error in pose processing: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return (image,)