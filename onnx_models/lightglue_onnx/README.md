# LightGlue ONNX

ONNX Runtime accelerated LightGlue for feature matching with TensorRT support.

## Overview

LightGlue is a lightweight attention-based feature matcher. Given keypoints and 
descriptors from two images, it finds correspondences using learned attention 
mechanisms.

This ONNX version supports:
- TensorRT execution (fastest, ~3x speedup)
- CUDA execution (reliable fallback)
- Dynamic keypoint counts (1 to 8192 per image)

## Usage

```python
from onnx_models import SuperPointONNX, LightGlueONNX
import cv2

# Load models
sp = SuperPointONNX()
lg = LightGlueONNX(features="superpoint")

# Extract features from both images
img0 = cv2.imread("image0.jpg", cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0

kp0, scores0, desc0 = sp.extract(img0)
kp1, scores1, desc1 = sp.extract(img1)

# Match features
matches, match_scores = lg.match(
    kp0, kp1, desc0, desc1,
    image_size0=img0.shape[:2],
    image_size1=img1.shape[:2],
)

print(f"Found {len(matches)} matches")
# matches[i] = [idx0, idx1] means kp0[idx0] matches kp1[idx1]
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| features | "superpoint" | Feature type (superpoint, disk, aliked) |
| device | "cuda" | Execution device |
| trt_cache_dir | None | TensorRT engine cache directory |

## Model Files

Two model variants are available:

1. `{features}_lightglue.trt.onnx` - TensorRT optimized (recommended)
2. `{features}_lightglue_fused.onnx` - CUDA fallback

Download from: https://github.com/fabio-sim/LightGlue-ONNX/releases

Run `python -m onnx_models.setup` to download the model.

## Technical Notes

Keypoint coordinates are normalized to [-1, 1] internally using the provided 
image sizes. If image sizes are not provided, keypoints are assumed to already 
be normalized.

The TensorRT provider uses dynamic shape profiles to handle variable keypoint 
counts. First inference may be slow while TensorRT builds the engine (cached 
for subsequent runs).

Match output format:
- matches: [K, 2] array where matches[i] = [idx_in_image0, idx_in_image1]
- scores: [K] array of match confidence (0 to 1)
