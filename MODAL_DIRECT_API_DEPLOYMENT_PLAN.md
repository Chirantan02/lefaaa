# Modal Direct API Deployment Plan - Leffa Virtual Try-On (SIMPLIFIED)

## 🎯 **END GOAL (SIMPLIFIED)**
Create a Modal-hosted API that:
1. Takes user image URL and garment image URL
2. Takes mask type option (upper/lower/full)
3. Generates mask automatically based on mask type
4. Generates virtual try-on image
5. Uploads result to GCP bucket
6. Returns GCP bucket URL of generated image

**SIMPLE FLOW**: Image URLs → Generate Mask → Virtual Try-On → Upload to GCP → Return URL

## 📋 **Current State Analysis**
- ✅ Custom Leffa ComfyUI nodes working locally
- ✅ Mask generation working (your masking nodes)
- ✅ Modal deployment structure exists (`modal_comfyui_app.py`)
- ✅ Test client ready (`test_client.py`)
- ✅ GCP bucket setup for image storage

## 🚀 **Deployment Strategy: Direct API (Simplified)**

### **Why Direct API:**
- **Simplicity**: Single endpoint, simple flow
- **Performance**: Fast mask generation + inference
- **Cost**: Lower compute costs
- **Integration**: Direct GCP bucket upload

---

## 📝 **IMPLEMENTATION STAGES (SIMPLIFIED)**

### **Stage 1: Basic Modal Setup**
**Objective**: Get Modal app running with core components

#### 1.1 Create Modal App
- [ ] Create `modal_leffa_simple_api.py`
- [ ] Set up Modal image with basic dependencies
- [ ] Configure GPU (L40S)
- [ ] Set up model volume

#### 1.2 Install Dependencies
- [ ] Core: torch, diffusers, transformers
- [ ] Image: PIL, opencv, requests
- [ ] Masking: onnxruntime-gpu (for pose detection)
- [ ] GCP: google-cloud-storage
- [ ] API: fastapi

#### 1.3 Model Setup
- [ ] Download Leffa models to Modal volume ONLY
- [ ] Use memory snapshots for faster cold starts
- [ ] Load models on container start
- [ ] Basic model validation
- [ ] **ALWAYS** provide detailed time and cost breakdown

### **Stage 2: Core Pipeline (Simplified)**
**Objective**: Implement the simple flow

#### 2.1 Image Processing
- [ ] Download images from URLs
- [ ] Basic image validation and resizing
- [ ] Image preprocessing for Leffa

#### 2.2 Mask Generation
- [ ] Use your existing masking nodes directly
- [ ] Support upper/lower/full mask types
- [ ] Generate pose detection
- [ ] Output mask for virtual try-on

#### 2.3 Virtual Try-On
- [ ] Use Leffa inference pipeline
- [ ] Process: user_image + garment_image + mask
- [ ] Generate virtual try-on result
- [ ] Basic error handling

### **Stage 3: GCP Integration & API**
**Objective**: Add GCP upload and create simple API

#### 3.1 GCP Storage Setup
- [ ] Set up GCP storage client
- [ ] Configure bucket access
- [ ] Implement image upload function
- [ ] Generate public URLs

#### 3.2 Simple API Endpoint
- [ ] Create single POST endpoint `/generate-tryon`
- [ ] Simple request schema:
  ```json
  {
    "user_image_url": "https://...",
    "garment_image_url": "https://...",
    "mask_type": "upper|lower|full"
  }
  ```
- [ ] Simple response:
  ```json
  {
    "status": "success",
    "result_image_url": "https://storage.googleapis.com/..."
  }
  ```

#### 3.3 Complete Pipeline
- [ ] Download images → Generate mask → Virtual try-on → Upload to GCP → Return URL
- [ ] Basic error handling
- [ ] Simple logging

### **Stage 4: Testing & Deployment**
**Objective**: Test and deploy the simple version

#### 4.1 Basic Testing
- [ ] Test with sample images
- [ ] Test mask generation
- [ ] Test GCP upload
- [ ] Test complete flow

#### 4.2 Deploy to Modal
- [ ] Deploy simple version
- [ ] Test with real URLs
- [ ] Basic performance check
- [ ] Fix any issues

#### 4.3 Create Test Client
- [ ] Update test_client.py for simple API
- [ ] Test from CLI
- [ ] Validate results

---

## 🔧 **Technical Specifications (SIMPLIFIED)**

### **Simple API Endpoint**
```
POST /generate-tryon
{
  "user_image_url": "https://firebasestorage.googleapis.com/...",
  "garment_image_url": "https://firebasestorage.googleapis.com/...",
  "mask_type": "upper|lower|full",
  "steps": 20,        // optional, default 20
  "cfg": 2.5,         // optional, default 2.5
  "seed": 656545      // optional, random if not provided
}
```

### **Simple Response**
```
{
  "status": "success|error",
  "result_image_url": "https://storage.googleapis.com/your-bucket/generated_image.png",
  "processing_time": 45.2,
  "error_message": "..." // only if status is error
}
```

### **Pipeline Flow**
1. **Download** user_image and garment_image from URLs
2. **Generate Mask** using your masking nodes (upper/lower/full)
3. **Generate Pose** using your pose detection
4. **Virtual Try-On** using Leffa inference
5. **Upload** result to GCP bucket
6. **Return** GCP bucket URL

### **Resource Requirements**
- **GPU**: L40S (24GB VRAM)
- **Memory**: 16GB RAM (simplified)
- **Storage**: 30GB for models
- **Timeout**: 180 seconds

### **GCP Bucket Structure**
```
your-bucket/
├── generated-images/
│   ├── tryon_20250123_001.png
│   ├── tryon_20250123_002.png
│   └── ...
```

---

## 📊 **Success Metrics (SIMPLIFIED)**
- [ ] API response time < 90 seconds
- [ ] Working end-to-end pipeline
- [ ] Successful GCP upload
- [ ] Quality virtual try-on results

## 🔄 **Next Steps**
1. **Start Stage 1**: Create basic Modal app
2. **Test locally**: Validate masking and inference work
3. **Deploy**: Get it working on Modal
4. **Iterate**: Improve and optimize

---

**SIMPLIFIED APPROACH**: Get the basic flow working first, then optimize later!
