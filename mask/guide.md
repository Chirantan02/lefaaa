# Mask Generation Guide

## Full Body Mask
To generate a full body mask with your RTX 4060 GPU, use the following command:

```
python mask/full.py --image_path "path/to/your/image.png" --output_dir output --model_root . --horizontal_expansion 20 --vertical_expansion 0 --gap_fill_kernel_size 7 --gap_fill_iterations 3 --force_gpu --half_precision --optimize_memory --mask_type full
```

This will generate both a mask file and an overlay visualization showing the mask on the original image.

## Upper Body Mask Only
To generate only the upper body mask:

```
python mask/full.py --image_path "path/to/your/image.png" --output_dir output --model_root . --force_gpu --half_precision --optimize_memory --mask_type upper
```

You can also directly use the upper.py script:

```
python mask/upper.py --image_path "path/to/your/image.png" --output_dir output --model_root .
```

This will generate both a mask file and an overlay visualization showing the mask on the original image.

## Lower Body Mask Only (No Shoes)
To generate only the lower body mask (excluding shoes):

```
python mask/full.py --image_path "path/to/your/image.png" --output_dir output --model_root . --horizontal_expansion 20 --vertical_expansion 0 --closing_kernel_size 9 --closing_iterations 3 --opening_kernel_size 5 --opening_iterations 1 --force_gpu --half_precision --optimize_memory --mask_type lower
```

You can also directly use the lower-no-shoes.py script:

```
python mask/lower-no-shoes.py --image_path "path/to/your/image.png" --output_dir output --model_root . --horizontal_expansion 20 --vertical_expansion 0 --closing_kernel_size 9 --closing_iterations 3 --opening_kernel_size 5 --opening_iterations 1
```

This will generate both a mask file and an overlay visualization showing the mask on the original image.

## Lower Body Mask With Shoes
To generate a lower body mask that includes shoes:

```
python mask/lower_with_shoes.py --image_path "path/to/your/image.png" --output_dir output --model_root . --horizontal_expansion 20 --vertical_expansion 10 --closing_kernel_size 9 --closing_iterations 3 --opening_kernel_size 5 --opening_iterations 1
```

This will generate both a mask file and an overlay visualization showing the mask on the original image.

Note: The vertical_expansion parameter should be set to a positive value (e.g., 10) to ensure the shoes are fully included in the mask.

## Batch Processing
To process multiple images at once, use one of the batch processing scripts:

### Upper Body Masks
To generate upper body masks for all images in a folder:

```
python mask/process_batch.py --input_folder "path/to/your/images" --output_folder "path/to/output" --model_root .
```

### Lower Body Masks (Without Shoes)
To generate lower body masks (excluding shoes) for all images in a folder:

```
python mask/process_batch_lower.py --input_folder "path/to/your/images" --output_folder "path/to/output" --model_root . --horizontal_expansion 20 --vertical_expansion 0
```

Note: This script uses lower-no-shoes.py internally to generate masks that exclude shoes.

### Lower Body Masks (With Shoes)
To generate lower body masks that include shoes for all images in a folder:

```
python mask/process_batch_lower.py --input_folder "path/to/your/images" --output_folder "path/to/output" --model_root . --horizontal_expansion 20 --vertical_expansion 10
```

Note: Use the same script but with a positive vertical_expansion value to include shoes.

### Full Body Masks (Both Upper and Lower)
To generate both upper and lower body masks for all images in a folder:

```
python mask/process_batch_full.py --input_folder "path/to/your/images" --output_folder "path/to/output" --model_root . --mask_type full
```

You can specify `--mask_type upper` or `--mask_type lower` to generate only upper or lower body masks respectively.

## Performance Options

- `--force_gpu`: Force GPU usage for all operations that support it
- `--half_precision`: Use half precision (FP16) for faster computation
- `--optimize_memory`: Optimize memory usage for better performance
- `--skip_overlay`: Skip overlay generation for faster processing
- `--device cpu`: Use CPU instead of GPU (useful if you encounter GPU memory issues)

## Mask Type Options

- `--mask_type full`: Generate and combine both upper and lower body masks (default)
- `--mask_type upper`: Generate only the upper body mask
- `--mask_type lower`: Generate only the lower body mask

## Notes

- The upper.py script has been simplified to use only the tank top approach with enhanced neck coverage, which has proven to be the most effective method.
- The lower-no-shoes.py script (formerly lower.py) excludes shoes from the mask by default (vertical_expansion=0).
- The lower_with_shoes.py script includes shoes in the mask by using a positive vertical_expansion value.
- All mask generation scripts now produce both mask files and overlay visualizations showing the mask on the original image.
- For batch processing, you can use the same process_batch_lower.py script with different vertical_expansion values to include or exclude shoes.
- The scripts run significantly faster with GPU acceleration and make better use of your RTX 4060's resources.