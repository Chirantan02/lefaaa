# Leffa Model Download Instructions

## The Problem
You're seeing this error because the required model files are missing:
```
Failed to load Leffa model: Required 'stable-diffusion-inpainting' directory not found
```

## Quick Fix - Manual Download

### Step 1: Go to Hugging Face
Open this link in your browser: https://huggingface.co/franciszzj/Leffa

### Step 2: Download Required Files
Click on "Files and versions" tab and download these 3 items:

1. **stable-diffusion-inpainting** (folder) - Click the download icon next to it
2. **virtual_tryon.pth** (file) - Click to download
3. **virtual_tryon_dc.pth** (file) - Click to download

### Step 3: Create Model Directory
In your current location (`custom_nodes\Comfyui_leffa`), create a new folder called `Leffa`

### Step 4: Place Files
Put the downloaded files in the `Leffa` folder so it looks like this:
```
Comfyui_leffa/
├── Leffa/                          ← Create this folder
│   ├── stable-diffusion-inpainting/  ← Downloaded folder
│   ├── virtual_tryon.pth            ← Downloaded file
│   └── virtual_tryon_dc.pth         ← Downloaded file
├── leffa/                          ← This is source code (already exists)
├── lib/
├── leffaNode.py
└── ...
```

### Step 5: Restart ComfyUI
Restart ComfyUI and try the Leffa nodes again.

## Alternative Methods

### Method 1: Use Git Clone
```bash
git clone https://huggingface.co/franciszzj/Leffa
```

### Method 2: Use Download Script
```bash
python download_leffa_model.py
```

### Method 3: Use Hugging Face Hub
```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('franciszzj/Leffa', local_dir='Leffa')"
```

## Verification
After downloading, your `Leffa` folder should contain:
- ✅ `stable-diffusion-inpainting/` (directory with model files)
- ✅ `virtual_tryon.pth` (PyTorch model file)
- ✅ `virtual_tryon_dc.pth` (PyTorch model file)

## Common Mistakes to Avoid
- ❌ Don't use the `leffa/` folder (that's source code)
- ❌ Don't rename the files
- ❌ Don't put files in the wrong location
- ✅ Make sure `stable-diffusion-inpainting` is a folder, not a file

## Still Having Issues?
1. Check your internet connection
2. Make sure you have enough disk space (model is ~5GB)
3. Try downloading one file at a time
4. Check the file sizes match what's shown on Hugging Face

## File Sizes (approximate)
- `stable-diffusion-inpainting/`: ~4.5GB
- `virtual_tryon.pth`: ~500MB
- `virtual_tryon_dc.pth`: ~500MB

## Troubleshooting Import Errors

If you're getting import errors like "No module named 'utils.transforms'" or "No module named 'preprocess.dwpose'", follow these steps:

### Step 1: Install Missing Dependencies
```bash
# Run the setup script to check and install dependencies
python setup_dependencies.py

# Or install manually:
pip install onnxruntime-gpu>=1.15.0  # For CUDA systems
pip install onnxruntime>=1.15.0      # For CPU-only systems
pip install opencv-python>=4.8.0
pip install diffusers>=0.21.0
pip install scikit-image>=0.20.0
```

### Step 2: Check Model Files
The masking functionality requires additional model files in the FitDiT_models directory:

**Required FitDiT Model Files:**
- `dwpose/dw-ll_ucoco_384.onnx`
- `dwpose/yolox_l.onnx`
- `humanparsing/parsing_atr.onnx`
- `humanparsing/parsing_lip.onnx`

These should be located at:
```
ComfyUI/models/FitDiT_models/
├── dwpose/
│   ├── dw-ll_ucoco_384.onnx
│   └── yolox_l.onnx
└── humanparsing/
    ├── parsing_atr.onnx
    └── parsing_lip.onnx
```

### Step 3: Run Diagnostics
```bash
# Run the diagnostic script to identify issues
python diagnose_imports.py
```

This will check all dependencies and model files and provide specific guidance on what's missing.

### Step 4: Restart ComfyUI
After installing dependencies and downloading model files, restart ComfyUI completely.
