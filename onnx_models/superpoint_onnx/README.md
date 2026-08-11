# SuperPoint ONNX

Keypoint detector and descriptor extractor. Uses ONNX Runtime with CUDA acceleration.

---

## Overview

**Purpose:** Detect keypoints and compute local descriptors for feature matching.

**Input:** Grayscale image (any size).

**Output:** Keypoints (x, y), detection scores, and 256-dim descriptors.

**Performance:** ~4ms per image with CUDA (vs ~8ms PyTorch).

---

## Class Structure

```
SuperPointONNX(BaseONNXModel)
│
├── Class Attributes
│   ├── MODEL_NAME = "superpoint_onnx.onnx"
│   └── MODELS_DIR = ./models/
│
├── Constructor Args
│   ├── model_path           : Path to ONNX file (auto-detects if None)
│   ├── max_num_keypoints    : Maximum keypoints to return (default: 2048)
│   ├── detection_threshold  : Minimum score (default: 0.0005)
│   ├── device               : "cuda" | "cpu"
│   ├── force_num_keypoints  : Always return exactly max_num_keypoints
│   └── verbose              : Print loading messages
│
├── Core Methods
│   ├── extract()          : Extract features (numpy array)
│   ├── extract_gpu()      : Extract features (GPU tensor, zero-copy)
│   ├── preprocess()       : Ensure [1, 1, H, W] shape
│   ├── postprocess()      : Filter keypoints by score/threshold
│   └── _filter_keypoints(): Apply threshold and top-K selection
│
└── Inherited from BaseONNXModel
    ├── forward()          : Run ONNX inference
    ├── warmup()           : CUDA kernel warmup
    └── __call__()         : preprocess → forward → postprocess
```

---

## Input/Output Specification

### Input

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| image | [1, 1, H, W] | float32 | Grayscale image, values in [0, 1] |

**Accepted input formats:**
- `[H, W]` — auto-expanded to [1, 1, H, W]
- `[1, H, W]` — auto-expanded to [1, 1, H, W]
- `[1, 1, H, W]` — used directly

**Note:** Any image size works. SuperPoint is fully convolutional.

### Outputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| keypoints | [N, 2] | float32 | (x, y) pixel coordinates |
| scores | [N] | float32 | Detection confidence (0 to 1) |
| descriptors | [N, 256] | float32 | L2-normalized descriptors |

---

## Configuration

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| max_num_keypoints | 2048 | Maximum keypoints returned. Keeps highest scores. |
| detection_threshold | 0.0005 | Minimum score to keep a keypoint. |
| device | "cuda" | Execution device. Use "cpu" if no GPU. |
| force_num_keypoints | False | If True, always return exactly max_num_keypoints. |
| verbose | True | Print model loading information. |

### Keypoint Filtering Modes

**Default mode** (`force_num_keypoints=False`):
1. Remove keypoints with score < `detection_threshold`
2. If count > `max_num_keypoints`, keep top-K by score

**Forced mode** (`force_num_keypoints=True`):
1. Skip threshold filtering
2. Always return exactly `max_num_keypoints` (top-K by score)

Use forced mode for batched LightGlue inference (requires fixed counts).

---

## Model File

### Download

```bash
# Automatic download
python -m onnx_models.setup

# Manual download
# From: https://github.com/fabio-sim/LightGlue-ONNX/releases
# File: superpoint.onnx → rename to superpoint_onnx.onnx
```

### File Location

```
superpoint_onnx/models/superpoint_onnx.onnx
```

---

## Usage Examples

### Basic Usage

```python
from onnx_models import SuperPointONNX
import cv2

# Load model
superpoint = SuperPointONNX(max_num_keypoints=4096)

# Load image (grayscale, float32, 0-1 range)
img = cv2.imread("image.jpg", cv2.IMREAD_GRAYSCALE)
img = img.astype("float32") / 255.0

# Extract features
keypoints, scores, descriptors = superpoint.extract(img)

print(f"Found {len(keypoints)} keypoints")
print(f"Keypoints shape: {keypoints.shape}")    # [N, 2]
print(f"Descriptors shape: {descriptors.shape}")  # [N, 256]
```

### GPU Usage (Zero-Copy)

```python
import torch

# Image already on GPU
img_gpu = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).cuda()

# Extract with IO binding (avoids GPU→CPU→GPU copies)
kp_gpu, scores_gpu, desc_gpu = superpoint.extract_gpu(img_gpu)

# Results are torch tensors on GPU
print(kp_gpu.device)  # cuda:0
```

### Fixed Keypoint Count

```python
# For batched LightGlue inference
superpoint = SuperPointONNX(
    max_num_keypoints=1024,
    force_num_keypoints=True,  # Always returns exactly 1024
)

kp, scores, desc = superpoint.extract(img)
assert len(kp) == 1024  # Guaranteed
```

---

## Technical Notes

### Why No TensorRT?

SuperPoint uses the `ScatterND` ONNX operation with reduction. TensorRT does not support this.

| Provider | Status | Reason |
|----------|--------|--------|
| TensorRT | ❌ Disabled | ScatterND with reduction unsupported |
| CUDA | ✅ Used | Full op support |
| CPU | ✅ Fallback | Works but slower |

The model always uses CUDA provider on GPU systems.

### Architecture

```
Input Image [1, 1, H, W]
    ↓
VGG-style Encoder (conv layers)
    ↓
Shared Features [1, 256, H/8, W/8]
    ↓
    ├── Detector Head → Heatmap [1, 65, H/8, W/8]
    │       ↓
    │   Softmax + Reshape → Scores [H, W]
    │       ↓
    │   NMS + Threshold → Keypoints [N, 2]
    │
    └── Descriptor Head → Dense Descriptors [1, 256, H/8, W/8]
            ↓
        Bilinear Sample at Keypoints → Descriptors [N, 256]
            ↓
        L2 Normalize
            ↓
        Output [N, 256]
```

### Fully Convolutional

SuperPoint accepts any input size because:
- No fixed positional embeddings (unlike ViT)
- All operations are convolutions or interpolations
- Output scales proportionally with input

Larger images → more keypoints (same density).

### Descriptor Properties

- **Dimension:** 256
- **Normalization:** L2-normalized (magnitude = 1.0)
- **Matching:** Use dot product or L2 distance

### Preprocessing

```
Input: Grayscale [H, W] uint8 (0-255)
    ↓
Scale: float32 (0-1)
    ↓
Expand: [1, 1, H, W]
    ↓
Output: [1, 1, H, W] float32
```

**Important:** Input must be in [0, 1] range, not [0, 255].

---

## Limitations

| Limitation | Reason | Workaround |
|------------|--------|------------|
| No TensorRT | ScatterND op unsupported | Use CUDA provider |
| Variable output size | Keypoint count depends on image | Use `force_num_keypoints` if needed |
| Grayscale only | Model training | Convert RGB: `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` |

---

## Troubleshooting

### "No keypoints detected"

- Check image is grayscale (not RGB)
- Check values are in [0, 1] range (not [0, 255])
- Lower `detection_threshold` (default 0.0005 is very low)
- Check image has enough texture

### "Too many keypoints"

- Reduce `max_num_keypoints`
- Increase `detection_threshold`

### "extract_gpu returns CPU tensors"

- IO Binding requires CUDA provider active
- Check `superpoint.use_io_binding` is True
- Check `superpoint.provider` is "CUDAExecutionProvider"

---

## File Structure

```
superpoint_onnx/
├── __init__.py              # Exports SuperPointONNX
├── model.py                 # SuperPointONNX class
├── README.md                # This file
├── setup.py                 # Download script
└── models/
    └── superpoint_onnx.onnx
```
