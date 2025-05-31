# 🎭 Standalone Virtual Try-On CLI

## ✅ **WORKING CLI - READY TO USE**

The **`standalone_cli_tryon.py`** is a fully functional CLI for virtual try-on using Leffa models. It works independently without ComfyUI dependencies and supports external mask URLs.

## 🚀 Quick Start

### Basic Usage
```bash
# Show help and all options
python standalone_cli_tryon.py --help

# Basic virtual try-on with predefined masks
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image shirt.jpg \
  --mask-type upper

# Using URLs (recommended for testing)
python standalone_cli_tryon.py \
  --user-image https://storage.googleapis.com/mask_images/f0518e08_Subliminator%20Printed%20_model.jpeg \
  --garment-image https://storage.googleapis.com/mask_images/image_prompt.jpeg \
  --mask-type full
```

### Configuration File Usage
```bash
# Create config_tryon.yaml with your settings
python standalone_cli_tryon.py --config config_tryon.yaml

# Override specific parameters
python standalone_cli_tryon.py \
  --config config_tryon.yaml \
  --steps 25 \
  --cfg 3.0 \
  --output my_result.png
```

## 📋 Command Line Options

### Required Arguments
- `--user-image` / `-u`: Path or URL to person/model image
- `--garment-image` / `-g`: Path or URL to garment/clothing image
- `--mask-image` / `-m`: Path or URL to mask image (OR use `--mask-type`)

### Mask Options
- `--mask-type`: Use predefined masks (`upper`, `lower`, `full`)
  - `upper`: For shirts, jackets, tops
  - `lower`: For pants, skirts, shorts  
  - `full`: For dresses, full-body garments

### Model & Performance
- `--model-path`: Leffa model path (default: `franciszzj/Leffa`)
- `--viton-type`: Virtual try-on type (`hd` for upper, `dc` for lower)
- `--device`: Device to use (`cuda`, `cpu`, `auto`)
- `--steps`: Inference steps (default: 20, range: 10-50)
- `--cfg`: Guidance scale (default: 2.5, range: 1.5-5.0)
- `--seed`: Random seed for reproducible results

### Output & Config
- `--output` / `-o`: Output image path (default: `standalone_output_tryon.png`)
- `--config` / `-c`: YAML configuration file path
- `--verbose` / `-v`: Enable verbose output

## 🎯 Predefined Mask URLs

The CLI includes these predefined mask URLs for quick testing:

- **Upper body**: `https://storage.googleapis.com/mask_images/upper_mask.png`
- **Lower body**: `https://storage.googleapis.com/mask_images/lower_mask.png`
- **Full body**: `https://storage.googleapis.com/mask_images/upper_mask.png`

Use with `--mask-type upper|lower|full` instead of providing custom mask images.

## ⚙️ Configuration File

Create `config_tryon.yaml`:

```yaml
# Model Configuration
model:
  path: "franciszzj/Leffa"
  viton_type: "hd"  # 'hd' for upper body, 'dc' for lower body
  device: "auto"    # 'cuda', 'cpu', or 'auto'

# Inference Parameters
inference:
  steps: 25         # Higher = better quality (10-50)
  cfg: 3.0          # Higher = more adherence to garment (1.5-5.0)
  seed: 42          # For reproducible results

# Image Paths
paths:
  user_image: "person.jpg"
  garment_image: "shirt.jpg"
  mask_image: "https://storage.googleapis.com/mask_images/upper_mask.png"
  output: "result.png"
```

## 🔧 Technical Details

### Architecture
- **Direct Diffusers Integration**: Uses `StableDiffusionInpaintPipeline`
- **No ComfyUI Dependencies**: Standalone execution
- **External Mask Support**: Bypasses mask generation entirely
- **Memory Optimization**: Includes attention slicing and CPU offload
- **URL Support**: Downloads images and masks from URLs

### Model Loading
- Automatically downloads Leffa models from HuggingFace
- Falls back to Stable Diffusion Inpainting as base model
- Supports both `hd` (upper body) and `dc` (lower body) variants
- Includes pose detection fallback (dummy pose if unavailable)

### Performance Tips
- Use `--mask-type` shortcuts for faster execution
- Start with lower `--steps` values (15-20) for testing
- Use `--device cpu` if GPU memory is limited
- Test with smaller images first

## 🎭 Example Workflows

### 1. Upper Body Try-On (Shirts, Jackets)
```bash
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image shirt.jpg \
  --mask-type upper \
  --viton-type hd \
  --steps 25 \
  --cfg 3.0
```

### 2. Lower Body Try-On (Pants, Skirts)
```bash
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image pants.jpg \
  --mask-type lower \
  --viton-type dc \
  --steps 20 \
  --cfg 2.5
```

### 3. Full Body Try-On (Dresses)
```bash
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image dress.jpg \
  --mask-type full \
  --viton-type hd \
  --steps 30 \
  --cfg 3.5
```

### 4. High Quality Settings
```bash
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image garment.jpg \
  --mask-type upper \
  --steps 40 \
  --cfg 4.0 \
  --seed 12345 \
  --output high_quality_result.png
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python standalone_cli_tryon.py --device cpu [other args]
   
   # Or reduce steps
   python standalone_cli_tryon.py --steps 15 [other args]
   ```

2. **Model Download Issues**
   ```bash
   # Check internet connection
   # Models download automatically from HuggingFace
   ```

3. **Image Loading Errors**
   ```bash
   # Test with verbose output
   python standalone_cli_tryon.py --verbose [other args]
   
   # Verify image URLs are accessible
   ```

### Performance Issues
- **Slow inference**: Reduce `--steps` or use `--device cpu`
- **Memory errors**: Use CPU mode or smaller images
- **Network timeouts**: Use local images instead of URLs

## ✅ Validation

Test your setup:
```bash
# Validate CLI structure
python validate_cli.py

# Test help command
python standalone_cli_tryon.py --help

# Quick test with predefined images
python standalone_cli_tryon.py \
  --user-image https://storage.googleapis.com/mask_images/f0518e08_Subliminator%20Printed%20_model.jpeg \
  --garment-image https://storage.googleapis.com/mask_images/image_prompt.jpeg \
  --mask-type full \
  --steps 15
```

## 📁 File Structure

```
├── standalone_cli_tryon.py      # ✅ Main CLI (WORKING)
├── config_loader.py             # ✅ Configuration system
├── config_tryon.yaml            # ✅ Default config
├── lib/ximg.py                  # ✅ Image utilities
├── CLI_README.md                # ✅ This documentation
├── USAGE_EXAMPLES.md            # ✅ Usage examples
├── CLI_IMPLEMENTATION_SUMMARY.md # ✅ Implementation summary
└── validate_cli.py              # ✅ Validation script
```

## 🎉 Success!

Your CLI is ready to use! The `standalone_cli_tryon.py` file provides a complete virtual try-on solution that works independently of ComfyUI and supports all the features you requested.
