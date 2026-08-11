# MegaLoc ONNX

Global descriptor extractor for image retrieval. Uses ONNX Runtime with CUDA acceleration.

---

## Overview

**Purpose:** Extract global image descriptors for place recognition and image retrieval.

**Input:** RGB image (any size, resized internally to 322×322).

**Output:** 8448-dimensional L2-normalized descriptor.

**Performance:** ~12ms per image with CUDA (vs ~45ms PyTorch).

---

## Class Structure

```
MegaLocONNX(BaseONNXModel)
│
├── Class Attributes
│   ├── MODEL_NAME = "megaloc_onnx.onnx"
│   ├── MODELS_DIR = ./models/
│   ├── MEAN = [0.485, 0.456, 0.406]  (ImageNet)
│   └── STD = [0.229, 0.224, 0.225]   (ImageNet)
│
├── Constructor Args
│   ├── model_path  : Path to ONNX file (auto-detects if None)
│   ├── device      : "cuda" | "cpu"
│   ├── resize      : Target size (default: 322)
│   └── verbose     : Print loading messages
│
├── Core Methods
│   ├── extract()    : Extract descriptor (convenience alias)
│   ├── preprocess() : Resize, normalize, transpose
│   ├── postprocess(): Return descriptor array
│   └── warmup()     : CUDA kernel warmup
│
└── Inherited from BaseONNXModel
    ├── forward()    : Run ONNX inference
    └── __call__()   : preprocess → forward → postprocess
```

---

## Input/Output Specification

### Input

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| images | [1, 3, 322, 322] | float32 | RGB image, ImageNet normalized |

**Accepted input formats:**
- `[H, W, 3]` uint8 (0-255) — auto-resized and normalized
- `[1, 3, H, W]` float32 (0-1) — only normalized

### Output

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| descriptor | [1, 8448] | float32 | L2-normalized global descriptor |

---

## Configuration

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| device | "cuda" | Execution device. Use "cpu" if no GPU. |
| resize | 322 | Target image size. Must be multiple of 14. |
| verbose | True | Print model loading information. |

### Valid Resize Values

The ViT-B/14 backbone uses 14×14 pixel patches. Image size must be divisible by 14.

| Size | Patches | Notes |
|------|---------|-------|
| 322 | 23×23 | Default. Matches training config. |
| 518 | 37×37 | Higher resolution. More detail. |
| 1078 | 77×77 | Very high resolution. Slow. |

**Note:** The ONNX model may be exported with fixed 322×322 shape. Check dynamic shape support before changing.

---

## Model File

### Export

MegaLoc requires export from PyTorch weights.

```bash
# Full setup (downloads weights, exports ONNX)
python -m onnx_models.setup

# Manual export
python -m onnx_models.megaloc_onnx.export
```

### File Location

```
megaloc_onnx/models/megaloc_onnx.onnx
```

---

## Usage Examples

### Basic Usage

```python
from onnx_models import MegaLocONNX
import cv2

# Load model
megaloc = MegaLocONNX()
megaloc.warmup()

# Load image (RGB format)
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extract descriptor
descriptor = megaloc.extract(img)

print(f"Shape: {descriptor.shape}")  # [1, 8448]
print(f"Norm: {np.linalg.norm(descriptor):.4f}")  # ~1.0
```

### Image Retrieval

```python
import numpy as np
from onnx_models import MegaLocONNX

megaloc = MegaLocONNX()
megaloc.warmup()

# Build database
db_descriptors = []
for img_path in database_images:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    desc = megaloc.extract(img)
    db_descriptors.append(desc)

db_descriptors = np.vstack(db_descriptors)  # [N, 8448]

# Query
query_img = cv2.cvtColor(cv2.imread("query.jpg"), cv2.COLOR_BGR2RGB)
query_desc = megaloc.extract(query_img)  # [1, 8448]

# Find similar images (dot product = cosine similarity for L2-normalized)
similarities = db_descriptors @ query_desc.T
top_k = np.argsort(similarities.flatten())[::-1][:10]
```

---

## Technical Notes

### Why No TensorRT?

MegaLoc uses DINOv2 ViT-B/14 backbone. TensorRT FP16 causes numerical instability.

| Provider | Status | Reason |
|----------|--------|--------|
| TensorRT FP16 | ❌ Disabled | Attention softmax overflow |
| TensorRT FP32 | ⚠️ No speedup | No benefit over CUDA |
| CUDA | ✅ Used | Stable and fast |

The model always uses CUDA provider, regardless of `use_tensorrt` parameter.

### Why 322×322?

ViT-B/14 processes images in 14×14 patches.

```
322 ÷ 14 = 23 patches per dimension
23 × 23 = 529 total patches
```

This matches MegaLoc's training configuration. Other sizes work but may affect accuracy.

### Preprocessing Pipeline

```
Input: RGB [H, W, 3] uint8 (0-255)
    ↓
Resize: [322, 322, 3] (bilinear)
    ↓
Scale: [322, 322, 3] float32 (0-1)
    ↓
Transpose: [3, 322, 322]
    ↓
Batch: [1, 3, 322, 322]
    ↓
ImageNet Normalize: (x - mean) / std
    ↓
Output: [1, 3, 322, 322] float32
```

### Descriptor Properties

- **Dimension:** 8448 (DINOv2 features + GeM pooling)
- **Normalization:** L2-normalized (magnitude = 1.0)
- **Similarity:** Use dot product (equals cosine similarity)

### Architecture

```
Input Image [1, 3, 322, 322]
    ↓
DINOv2 ViT-B/14 (Vision Transformer)
    ↓
Patch Embeddings [1, 529, 768]
    ↓
GeM Pooling (Generalized Mean)
    ↓
Global Descriptor [1, 8448]
    ↓
L2 Normalize
    ↓
Output [1, 8448]
```

---

## Limitations

| Limitation | Reason | Workaround |
|------------|--------|------------|
| No TensorRT | DINOv2 FP16 instability | Use CUDA provider |
| Fixed input size | ONNX export constraint | Re-export for different size |
| Large descriptor | 8448 dims × 4 bytes = 33KB | PCA reduction if needed |

---

## Troubleshooting

### "Model not found"

```bash
# Export the model first
python -m onnx_models.setup
```

### "Empty outputs"

- Check image format is RGB, not BGR
- Verify image has 3 channels
- Check CUDA provider is available

### "Wrong descriptor values"

- Ensure ImageNet normalization is applied
- Check input is float32 in [0, 1] range
- Verify image is RGB (not BGR from cv2.imread)

---

## File Structure

```
megaloc_onnx/
├── __init__.py          # Exports MegaLocONNX
├── model.py             # MegaLocONNX class
├── export.py            # PyTorch → ONNX export script
├── README.md            # This file
├── setup.py             # Download/export script
└── models/
    └── megaloc_onnx.onnx
```
