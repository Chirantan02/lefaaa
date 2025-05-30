import os
import time
import logging

# Try to import folder_paths, but don't fail if it's not available
try:
    import folder_paths
    FOLDER_PATHS_AVAILABLE = True
except ImportError:
    FOLDER_PATHS_AVAILABLE = False
    folder_paths = None

# Set up logging
logger = logging.getLogger(__name__)

def download_hg_model(model_id: str, exDir: str = '', offline_mode: bool = False, max_retries: int = 3):
    """
    Download Hugging Face model to local directory with robust error handling.

    Args:
        model_id: Hugging Face model ID (e.g., 'franciszzj/Leffa')
        exDir: Extra directory path within models_dir
        offline_mode: If True, only use local models
        max_retries: Maximum number of download attempts

    Returns:
        str: Path to the local model directory

    Raises:
        FileNotFoundError: If model not found locally and offline_mode=True
        ConnectionError: If download fails after all retries
    """
    # Construct local model path
    if FOLDER_PATHS_AVAILABLE and folder_paths:
        model_checkpoint = os.path.join(folder_paths.models_dir, exDir, os.path.basename(model_id))
    else:
        # Fallback to a default models directory
        default_models_dir = r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models"
        model_checkpoint = os.path.join(default_models_dir, exDir, os.path.basename(model_id))

    print(f"Model checkpoint path: {model_checkpoint}")

    # Check if model already exists locally
    if os.path.exists(model_checkpoint):
        print(f"Model found locally at: {model_checkpoint}")
        return model_checkpoint

    # If offline mode and model doesn't exist locally, raise error
    if offline_mode:
        raise FileNotFoundError(
            f"Model '{model_id}' not found locally at '{model_checkpoint}' and offline_mode=True. "
            f"Please download the model manually or set offline_mode=False."
        )

    # Try to download the model with retries
    print(f"Model not found locally. Attempting to download '{model_id}'...")

    for attempt in range(max_retries):
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError, RepositoryNotFoundError

            print(f"Download attempt {attempt + 1}/{max_retries}")

            # Create directory if it doesn't exist
            os.makedirs(model_checkpoint, exist_ok=True)

            # Download with timeout and error handling
            snapshot_download(
                repo_id=model_id,
                local_dir=model_checkpoint,
                local_dir_use_symlinks=False,
                resume_download=True,  # Resume partial downloads
                local_files_only=False
            )

            print(f"Successfully downloaded model to: {model_checkpoint}")
            return model_checkpoint

        except (LocalEntryNotFoundError, RepositoryNotFoundError) as e:
            error_msg = f"Model '{model_id}' not found on Hugging Face Hub: {str(e)}"
            print(error_msg)
            raise FileNotFoundError(error_msg)

        except Exception as e:
            print(f"Download attempt {attempt + 1} failed: {str(e)}")

            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                error_msg = (
                    f"Failed to download model '{model_id}' after {max_retries} attempts. "
                    f"Last error: {str(e)}\n"
                    f"Please check your internet connection or download the model manually to: {model_checkpoint}"
                )
                print(error_msg)
                raise ConnectionError(error_msg)

    return model_checkpoint


def validate_model_directory(model_path: str) -> bool:
    """
    Validate that a directory contains the required Leffa model files.

    Args:
        model_path: Path to check

    Returns:
        bool: True if valid model directory, False otherwise
    """
    if not os.path.exists(model_path):
        return False

    # Check for required files/directories
    required_items = [
        "stable-diffusion-inpainting",  # Directory
        "virtual_tryon.pth",           # File
        "virtual_tryon_dc.pth"         # File
    ]

    for item in required_items:
        item_path = os.path.join(model_path, item)
        if not os.path.exists(item_path):
            print(f"Missing required item: {item_path}")
            return False

    # Additional check: stable-diffusion-inpainting should be a directory
    inpainting_path = os.path.join(model_path, "stable-diffusion-inpainting")
    if not os.path.isdir(inpainting_path):
        print(f"stable-diffusion-inpainting should be a directory: {inpainting_path}")
        return False

    # Check that stable-diffusion-inpainting has the expected subdirectories
    required_subdirs = ["scheduler", "unet", "vae"]
    for subdir in required_subdirs:
        subdir_path = os.path.join(inpainting_path, subdir)
        if not os.path.exists(subdir_path):
            print(f"Missing required subdirectory: {subdir_path}")
            return False
        if not os.path.isdir(subdir_path):
            print(f"Expected directory but found file: {subdir_path}")
            return False

        # Check for config.json in each subdirectory
        config_path = os.path.join(subdir_path, "config.json")
        if subdir in ["unet", "vae"] and not os.path.exists(config_path):
            print(f"Missing config.json in: {subdir_path}")
            return False

        # For scheduler, check for scheduler_config.json
        if subdir == "scheduler":
            scheduler_config_path = os.path.join(subdir_path, "scheduler_config.json")
            if not os.path.exists(scheduler_config_path):
                print(f"Missing scheduler_config.json in: {subdir_path}")
                return False

    return True


def get_manual_model_path(model_id: str, exDir: str = ''):
    """
    Check for manually placed model in various locations.
    Only returns paths that contain valid model files.

    Args:
        model_id: Hugging Face model ID
        exDir: Extra directory path

    Returns:
        str or None: Path to manually placed model if found, None otherwise
    """
    model_name = os.path.basename(model_id)

    # List of potential locations to check
    potential_paths = [
        # Current directory
        os.path.join(os.getcwd(), model_name),
        # Custom nodes directory
        os.path.join(os.path.dirname(os.path.dirname(__file__)), model_name),
        # Local models subdirectory
        os.path.join(os.path.dirname(__file__), "..", "models", model_name),
        # Alternative names
        os.path.join(os.getcwd(), "Leffa_model"),
        os.path.join(os.getcwd(), "leffa_model"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "Leffa_model"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "leffa_model"),
        # ComfyUI models directory paths (both uppercase and lowercase)
        r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\Leffa",
        r"C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\models\leffa",
        # Try with folder_paths if available
    ]

    # Add folder_paths based locations if available
    if FOLDER_PATHS_AVAILABLE and folder_paths:
        try:
            models_dir = folder_paths.models_dir
            potential_paths.extend([
                os.path.join(models_dir, "Leffa"),
                os.path.join(models_dir, "leffa"),
                os.path.join(models_dir, exDir, model_name) if exDir else os.path.join(models_dir, model_name),
            ])
        except:
            pass

    for path in potential_paths:
        if validate_model_directory(path):
            print(f"Found valid manual model at: {path}")
            return path
        elif os.path.exists(path):
            print(f"Directory exists but missing required files: {path}")

    return None


def download_hg_model_safe(model_id: str, exDir: str = '', offline_mode: bool = False, max_retries: int = 3):
    """
    Safe wrapper for download_hg_model that tries manual paths first.

    Args:
        model_id: Hugging Face model ID
        exDir: Extra directory path within models_dir
        offline_mode: If True, only use local models
        max_retries: Maximum number of download attempts

    Returns:
        str: Path to the local model directory

    Raises:
        FileNotFoundError: If model cannot be found or downloaded
    """
    # First, try to find manually placed model
    manual_path = get_manual_model_path(model_id, exDir)
    if manual_path:
        return manual_path

    # If not found manually, try the original download function
    try:
        return download_hg_model(model_id, exDir, offline_mode, max_retries)
    except Exception as e:
        # Provide helpful error message with manual installation instructions
        model_name = os.path.basename(model_id)
        error_msg = (
            f"Failed to download model '{model_id}': {str(e)}\n\n"
            f"MANUAL INSTALLATION INSTRUCTIONS:\n"
            f"1. Download the model manually from: https://huggingface.co/{model_id}\n"
            f"   - Click 'Files and versions' tab\n"
            f"   - Download these required files:\n"
            f"     * stable-diffusion-inpainting/ (entire folder)\n"
            f"     * virtual_tryon.pth\n"
            f"     * virtual_tryon_dc.pth\n\n"
            f"2. Create a folder named '{model_name}' in one of these locations:\n"
            f"   - {os.path.join(os.getcwd(), model_name)}\n"
            f"   - {os.path.join(os.path.dirname(os.path.dirname(__file__)), model_name)}\n"
            f"   - {os.path.join(os.path.dirname(__file__), '..', 'models', model_name)}\n\n"
            f"3. Place the downloaded files in the '{model_name}' folder\n"
            f"4. Restart ComfyUI and try again.\n\n"
            f"IMPORTANT: Do NOT use the source code folder 'Leffa' that contains Python files.\n"
            f"You need the actual model files from Hugging Face.\n\n"
            f"Alternative: Run 'python download_leffa_model.py' for automatic download."
        )
        raise FileNotFoundError(error_msg)


# clip_model = AutoModelForCausalLM.from_pretrained(
#                 CLIP_PATH,
#                 device_map="cuda",
#                 trust_remote_code=True,
#                 torch_dtype="auto"
#             )

#         clip_processor = AutoProcessor.from_pretrained(CLIP_PATH, trust_remote_code=True)