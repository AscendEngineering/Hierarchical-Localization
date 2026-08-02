# MegaLoc ONNX

ONNX Runtime accelerated MegaLoc for global image descriptors with TensorRT support.

## Overview

MegaLoc extracts global descriptors for image retrieval and place recognition.
Built on DINOv2 ViT-B/14 with GeM pooling, it outputs 8448-dimensional descriptors
that can match images across viewpoint and illumination changes.

This ONNX version supports:
- TensorRT execution (fastest, ~8x speedup over PyTorch)
- CUDA execution (reliable fallback)

## Usage

```python
from onnx_models import MegaLocONNX
import cv2

# Load model (run setup.py --export-megaloc first)
ml = MegaLocONNX(use_tensorrt=True)
ml.warmup()  # Important for TensorRT

# Load image (RGB format)
img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Extract descriptor
descriptor = ml.extract(img, resize=322)

print(f"Descriptor shape: {descriptor.shape}")  # [1, 8448]
print(f"Descriptor norm: {np.linalg.norm(descriptor):.4f}")  # ~1.0 (L2 normalized)
```

## Image Retrieval Example

```python
import numpy as np
from onnx_models import MegaLocONNX

ml = MegaLocONNX()
ml.warmup()

# Extract descriptors for database images
db_descriptors = []
for img_path in database_images:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    desc = ml.extract(img)
    db_descriptors.append(desc)

db_descriptors = np.vstack(db_descriptors)  # [N, 8448]

# Query
query_img = cv2.cvtColor(cv2.imread("query.jpg"), cv2.COLOR_BGR2RGB)
query_desc = ml.extract(query_img)  # [1, 8448]

# Find similar images (cosine similarity = dot product for L2-normalized vectors)
similarities = db_descriptors @ query_desc.T
top_k_indices = np.argsort(similarities.flatten())[::-1][:10]
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| device | "cuda" | Execution device |
| use_tensorrt | True | Enable TensorRT acceleration |
| resize | 322 | Image size (must be multiple of 14) |

## Model File

Setup all models (including MegaLoc export):

```bash
python -m onnx_models.setup
```

Or export manually:

```bash
python -m onnx_models.megaloc_onnx.export --output megaloc_onnx/models/megaloc_onnx.onnx
```

## Technical Notes

Input preprocessing:
1. Resize to 322x322 (or other multiple of 14)
2. Convert to float32 and scale to [0, 1]
3. Normalize with ImageNet mean/std

The 322x322 size comes from ViT-B/14 which processes 14x14 patches. 322 = 23 * 14
gives 23x23 = 529 patches, matching MegaLoc's training configuration.

Output is L2-normalized so cosine similarity equals dot product.

TensorRT engine is cached in `megaloc_onnx/models/.trt_cache/`. First inference
is slow (~30s) while TensorRT builds the engine, but subsequent runs reuse the cache.
