# SuperPoint ONNX

ONNX Runtime accelerated SuperPoint for keypoint detection and description.

## Overview

SuperPoint is a self-supervised CNN that detects interest points and computes 
256-dimensional descriptors. This ONNX version runs inference through ONNX Runtime,
using CUDA for GPU acceleration.

The model takes a grayscale image and outputs:
- Keypoint locations (x, y pixel coordinates)
- Detection scores (higher = more confident)
- 256-dim descriptors (L2 normalized)

## Usage

```python
from onnx_models import SuperPointONNX
import cv2

# Load model (downloads automatically if needed)
sp = SuperPointONNX(max_num_keypoints=4096)

# Load and preprocess image
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype("float32") / 255.0

# Extract features
keypoints, scores, descriptors = sp.extract(img)

print(f"Found {len(keypoints)} keypoints")
print(f"Descriptors shape: {descriptors.shape}")  # [N, 256]
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_num_keypoints | 2048 | Maximum keypoints to return (keeps highest scores) |
| detection_threshold | 0.0005 | Minimum score threshold |
| device | "cuda" | Execution device ("cuda" or "cpu") |

## Model File

The ONNX model is downloaded from the LightGlue-ONNX releases:
https://github.com/fabio-sim/LightGlue-ONNX

Run `python -m onnx_models.setup` to download the model.

## Technical Notes

The ONNX model expects input shape [1, 1, H, W] with float32 values in [0, 1].
Output keypoints are in pixel coordinates (not normalized). Descriptors are 
already L2 normalized by the model.
