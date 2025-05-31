# Modal CPU-Only Mask Generation

> **⚠️ CRITICAL WARNING: FALLBACKS AND PLACEHOLDERS ARE STRICTLY PROHIBITED ⚠️**
> All outfit combinations MUST be pre-generated. Do NOT implement any fallback mechanisms or placeholder images.
> The system must ONLY display actual generated VTON images. If a VTON image is not available, show an error message
> instructing the user that pre-generation is required.
>
> **⚠️ CRITICAL: COMPREHENSIVE LOGGING IS MANDATORY ⚠️**
> Every step of the VTON process MUST be logged in detail to both the Flutter terminal and Firestore.
> No VTON-related process should execute without detailed logs. All logs must include timestamps, user IDs,
> and component information.

## Overview

This script provides an optimized solution for generating upper and lower body masks from user images using CPU-only processing. It's designed to run on Modal's cloud infrastructure, providing fast and cost-effective mask generation without requiring GPU resources.

## Key Features

- **CPU-Only Processing**: Optimized to run efficiently on CPU, eliminating the need for expensive GPU resources
- **Memory Snapshots**: Implements Modal Memory Snapshots for significantly faster startup times (1.5-3x reduction in cold start time)
- **Cloudinary Integration**: Automatically uploads generated masks to Cloudinary for easy access
- **Robust Error Handling**: Comprehensive error handling and logging for reliable operation
- **Performance Optimizations**: Includes various optimizations for faster processing and reduced resource usage

## How It Works

1. **Input**: Reads user image URL from data.json or as a parameter
2. **Processing**:
   - Generates upper body mask (covering torso, arms, and neck)
   - Generates lower body mask (covering legs but excluding shoes)
3. **Output**: Returns Cloudinary URLs for the generated masks

## Technical Details

### Dependencies

- Python 3.10+
- PyTorch (CPU version)
- torchvision
- OpenCV
- ONNX Runtime
- Cloudinary
- tqdm
- Various other Python libraries (numpy, PIL, etc.)

### Model Architecture

The script uses two main components for mask generation:

1. **Pose Detection**: Uses DWPose (ONNX models) to detect human pose keypoints
2. **Human Parsing**: Uses specialized parsing models to segment the human body into different parts

### Performance

- **Average Processing Time**: ~25-30 seconds per image
- **Memory Usage**: ~4GB
- **Cost**: Extremely low (less than $0.001 per run on Modal)

## Usage

### Deployment

The CPU-only mask generation API is already deployed at:
```
https://zebels-main--cpu-mask-generation-optimized-main.modal.run
```

You can use this endpoint directly in your application.

### API Usage

To call the deployed API:

```javascript
// Using axios in JavaScript/Node.js
const response = await axios.post(
  'https://zebels-main--cpu-mask-generation-optimized-main.modal.run',
  { user_image_url: 'https://example.com/user_image.jpg' }
);

// Using fetch in JavaScript
const response = await fetch(
  'https://zebels-main--cpu-mask-generation-optimized-main.modal.run',
  {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_image_url: 'https://example.com/user_image.jpg' })
  }
);

// Using http package in Dart/Flutter
final response = await http.post(
  Uri.parse('https://zebels-main--cpu-mask-generation-optimized-main.modal.run'),
  headers: {'Content-Type': 'application/json'},
  body: jsonEncode({'user_image_url': profileImageUrl})
);
```

### Example Output

```json
{
  "status": "success",
  "masks": {
    "upper": "https://res.cloudinary.com/example/image/upload/v1234567890/upper_mask.png",
    "lower": "https://res.cloudinary.com/example/image/upload/v1234567890/lower_mask.png"
  },
  "total_time": 26.57,
  "timing": {
    "setup_model_symlinks": 0.00,
    "download_image": 3.70,
    "generate_masks": 19.77,
    "upload_masks": 1.32
  }
}
```

### Integration with data.json

The generated mask URLs should be stored in the user's profile in Firestore and included in the data.json file used for virtual try-on generation. The VTONTestDataExtractor class in the Flutter app should include these URLs in the following format:

```json
{
  "userImageUrl": "https://firebasestorage.googleapis.com/...",
  "tops": [...],
  "bottoms": [...],
  "apiRequest": {...},
  "Mask": {
    "Status": "success",
    "Upper mask URL": "https://res.cloudinary.com/example/image/upload/v1234567890/upper_mask.png",
    "Lower mask URL": "https://res.cloudinary.com/example/image/upload/v1234567890/lower_mask.png"
  }
}
```

This allows the Modal API to use the pre-generated masks instead of generating them on-the-fly, improving performance and reducing costs.

## Cost Analysis

Based on Modal's pricing (https://modal.com/pricing):

- **Free Tier**: 100 CPU-hours/month, 100 GB-hours of memory
- **Cost per run**: Approximately $0.000218 (less than a penny)
- **Monthly usage (100 runs)**: ~1.48 CPU-hours and 2.95 GB-hours

This script can be run hundreds of times per month while staying within Modal's Free Tier limits.

## Troubleshooting

Common issues and solutions:

1. **Missing dependencies**: Ensure all required packages are installed
2. **Model loading errors**: Check that model paths are correct and models are accessible
3. **Python path issues**: Make sure the cleaned_mask directory is in the Python path

## Future Improvements

Potential enhancements for future versions:

1. Further optimize processing time with parallel execution
2. Add support for batch processing multiple images
3. Implement additional mask types (e.g., full body, face only)
4. Add options for different mask resolutions and formats
