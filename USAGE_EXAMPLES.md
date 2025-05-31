# Virtual Try-On CLI Usage Examples

## Quick Start Commands

### 1. Basic Usage with Predefined Masks

```bash
# Upper body try-on (shirts, jackets)
python standalone_cli_tryon.py \
  --user-image C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_leffa\output\user_image.jpeg\
  --garment-image C:\Users\chira\Downloads\ninja downlaod\ComfyUI_windows_portable_nvidia_1\ComfyUI_windows_portable\ComfyUI\custom_nodes\Comfyui_leffa\output\garment_image.jpeg \
  --mask-type upper \
  --viton-type hd

# Lower body try-on (pants, skirts)
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image pants.jpg \
  --mask-type lower \
  --viton-type dc
```

### 2. Using URLs

```bash
# With external image URLs
python standalone_cli_tryon.py \
  --user-image https://example.com/person.jpg \
  --garment-image https://example.com/shirt.jpg \
  --mask-type upper \
  --output result.png
```

### 3. Custom Parameters

```bash
# High quality settings
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image dress.jpg \
  --mask-type full \
  --steps 30 \
  --cfg 3.5 \
  --seed 12345

# Fast mode
python standalone_cli_tryon.py \
  --user-image person.jpg \
  --garment-image shirt.jpg \
  --mask-type upper \
  --steps 15 \
  --cfg 2.0
```

### 4. Using Configuration Files

```bash
# Create and edit config_tryon.yaml with your settings
python standalone_cli_tryon.py --config config_tryon.yaml

# Override specific parameters
python standalone_cli_tryon.py \
  --config config_tryon.yaml \
  --steps 25 \
  --output custom_result.png
```

## Configuration File Example

Create `my_config.yaml`:

```yaml
# Model Configuration
model:
  path: "franciszzj/Leffa"
  viton_type: "hd"  # 'hd' for upper body, 'dc' for lower body
  device: "auto"

# Inference Parameters
inference:
  steps: 25         # Higher = better quality
  cfg: 3.0          # Higher = more adherence to garment
  seed: 42          # For reproducible results

# Image Paths
paths:
  user_image: "my_person.jpg"
  garment_image: "my_shirt.jpg"
  mask_image: "https://storage.googleapis.com/mask_images/upper_mask.png"
  output: "my_result.png"
```

Then run:
```bash
python standalone_cli_tryon.py --config my_config.yaml
```

## Predefined Mask URLs

The CLI includes these predefined mask URLs:

- **Upper body**: `https://storage.googleapis.com/mask_images/upper_mask.png`
- **Lower body**: `https://storage.googleapis.com/mask_images/lower_mask.png`
- **Full body**: `https://storage.googleapis.com/mask_images/upper_mask.png`

Use with `--mask-type upper|lower|full`

## Parameter Guidelines

### Steps (--steps)
- **10-15**: Fast mode, lower quality
- **20-25**: Balanced quality/speed (recommended)
- **30-50**: High quality, slower

### CFG Scale (--cfg)
- **1.5-2.0**: More creative, less adherence to garment
- **2.5-3.0**: Balanced (recommended)
- **3.5-5.0**: Strong adherence to garment details

### Viton Type (--viton-type)
- **hd**: Upper body garments (shirts, jackets, tops)
- **dc**: Lower body garments (pants, skirts, shorts)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Use CPU instead
   python standalone_cli_tryon.py --device cpu [other args]

   # Or reduce steps
   python standalone_cli_tryon.py --steps 15 [other args]
   ```

2. **Image Loading Errors**
   ```bash
   # Check if files exist
   ls -la your_image.jpg

   # Test with verbose output
   python standalone_cli_tryon.py --verbose [other args]
   ```

3. **Model Loading Issues**
   ```bash
   # Check internet connection for HuggingFace models
   # Or use local model path
   python standalone_cli_tryon.py --model-path /path/to/local/model [other args]
   ```

### Performance Tips

- Use `--mask-type` instead of custom masks for faster loading
- Start with lower `--steps` values for testing
- Use `--device cpu` if GPU memory is limited
- Test with smaller images first

## Advanced Usage

### Batch Processing Script

Create `batch_tryon.py`:

```python
import subprocess
import sys

# List of configurations
configs = [
    {
        "user": "person1.jpg",
        "garment": "shirt1.jpg",
        "output": "result1.png",
        "mask_type": "upper"
    },
    {
        "user": "person2.jpg",
        "garment": "dress1.jpg",
        "output": "result2.png",
        "mask_type": "full"
    }
]

for i, config in enumerate(configs):
    print(f"Processing {i+1}/{len(configs)}: {config['output']}")

    cmd = [
        sys.executable, "standalone_cli_tryon.py",
        "--user-image", config["user"],
        "--garment-image", config["garment"],
        "--mask-type", config["mask_type"],
        "--output", config["output"]
    ]

    subprocess.run(cmd)
    print(f"Completed: {config['output']}")
```

### Custom Mask Creation

If you need custom masks:

1. Create a grayscale image where:
   - White (255) = areas to replace
   - Black (0) = areas to keep
2. Save as PNG format
3. Use `--mask-image path/to/your/mask.png`

## Getting Help

```bash
# Show all available options
python standalone_cli_tryon.py --help

# Test configuration
python config_loader.py

# Run examples
python example_usage.py
```
