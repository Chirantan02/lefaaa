# ComfyUI Leffa - Virtual Try-On Node

This is a ComfyUI custom node implementation of [Leffa](https://github.com/franciszzj/Leffa) for virtual try-on applications.

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model

The node will automatically try to download the model from Hugging Face. If this fails due to network issues, you have several options:

#### Option A: Automatic Download (Recommended)
The node will automatically attempt to download the model on first use. If you encounter network issues, try the manual methods below.

#### Option B: Manual Download Script
```bash
python download_leffa_model.py
```

#### Option C: Manual Download
1. Go to: https://huggingface.co/franciszzj/Leffa
2. Download the following required files:
   - `stable-diffusion-inpainting/` (entire directory)
   - `virtual_tryon.pth`
   - `virtual_tryon_dc.pth`
3. Place them in one of these locations:
   - `./Leffa/` (in the custom node directory)
   - `./models/Leffa/` (in a models subdirectory)
   - In your ComfyUI models directory under `Leffa/`

#### Option D: Git Clone
```bash
git clone https://huggingface.co/franciszzj/Leffa
```

## Troubleshooting

### Network/Download Issues
If you see errors like `LocalEntryNotFoundError` or connection timeouts:

1. **Check your internet connection**
2. **Use the manual download script**: `python download_leffa_model.py`
3. **Download manually** from the Hugging Face page
4. **Use offline mode** by placing the model files locally

### Model Structure
The downloaded model should have this structure:
```
Leffa/
├── stable-diffusion-inpainting/
│   ├── model_index.json
│   ├── scheduler/
│   ├── unet/
│   ├── vae/
│   └── ...
├── virtual_tryon.pth
└── virtual_tryon_dc.pth
```

### Error Messages
- **"Model not found locally"**: Download the model manually
- **"Required files missing"**: Ensure all required files are present
- **"Connection failed"**: Check internet connection or use manual download

## Usage

1. **Load Model**: Use the `CXH_Leffa_Viton_Load` node to load the model
   - Select model: `franciszzj/Leffa`
   - Choose type: `hd` (high definition) or `dc` (dress code)

2. **Run Inference**: Use the `CXH_Leffa_Viton_Run` node
   - Connect the loaded model
   - Provide input images: model, cloth, pose, mask
   - Adjust parameters: steps, CFG, seed

## Workflow
![workflow](https://github.com/user-attachments/assets/4efc53fe-b78a-451d-b684-736203914190)

## Model Requirements
![Model Files](https://github.com/user-attachments/assets/c7f1d38a-64ae-42d6-95ca-31960768712b)


# lefaaa
# lefaaa
# lefaaa
