# LightGlue ONNX

Feature matcher for keypoint correspondences. Uses ONNX Runtime with TensorRT acceleration.

---

## Overview

**Purpose:** Match keypoints between two images using learned attention.

**Input:** Keypoints and descriptors from two images (from SuperPoint, DISK, or ALIKED).

**Output:** Matched keypoint pairs with confidence scores.

**Performance:** ~8ms per match pair with TensorRT (vs ~25ms PyTorch).

---

## Class Structure

```
LightGlueONNX(BaseONNXModel)
│
├── Class Attributes
│   ├── MODEL_NAME = "lightglue_onnx.trt.onnx"
│   ├── MODELS_DIR = ./models/
│   └── DESCRIPTOR_DIMS = {superpoint: 256, disk: 128, aliked: 128}
│
├── Constructor Args
│   ├── model_path     : Path to ONNX file (auto-detects if None)
│   ├── features       : "superpoint" | "disk" | "aliked"
│   ├── device         : "cuda" | "cpu"
│   ├── trt_cache_dir  : TensorRT engine cache location
│   └── verbose        : Print loading messages
│
├── Core Methods
│   ├── match()        : Match features (numpy arrays)
│   ├── match_gpu()    : Match features (GPU tensors, zero-copy)
│   ├── preprocess()   : Normalize keypoints, add batch dim
│   └── postprocess()  : Filter valid matches
│
└── Inherited from BaseONNXModel
    ├── forward()      : Run ONNX inference
    ├── warmup()       : Pre-build TensorRT engines
    └── IO binding helpers
```

---

## Input/Output Specification

### Inputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| kpts0 | [1, N, 2] | float32 | Keypoints from image 0, normalized to [-1, 1] |
| kpts1 | [1, M, 2] | float32 | Keypoints from image 1, normalized to [-1, 1] |
| desc0 | [1, N, D] | float32 | Descriptors from image 0 (D=256 for SuperPoint) |
| desc1 | [1, M, D] | float32 | Descriptors from image 1 |

**Note:** The `match()` method accepts pixel coordinates. It normalizes them internally.

### Outputs

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| matches | [K, 2] | int64 | Match pairs: [index_in_image0, index_in_image1] |
| scores | [K] | float32 | Match confidence (0 to 1) |

---

## Configuration

### Constructor Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| features | "superpoint" | Feature extractor type. Determines descriptor dimension. |
| device | "cuda" | Execution device. Use "cpu" if no GPU available. |
| trt_cache_dir | ./models/.trt_cache/ | Cache for TensorRT engines. Speeds up subsequent loads. |
| verbose | True | Print model loading information. |

### TensorRT Shape Profiles

LightGlue uses dynamic shapes. TensorRT requires min/opt/max profiles.

| Profile | Shape | Description |
|---------|-------|-------------|
| min | 1×1×2 | Minimum 1 keypoint |
| opt | 1×4096×2 | Optimized for 4096 keypoints |
| max | 1×4096×2 | Maximum 4096 keypoints |

**Warning:** If keypoint count exceeds max, TensorRT falls back to CUDA (slower).

---

## Model Files

### Available Variants

| File | Provider | Description |
|------|----------|-------------|
| `lightglue_onnx.trt.onnx` | TensorRT | Optimized for TRT. Best performance. |
| `lightglue_onnx_fused.onnx` | CUDA | Fused attention ops. Fallback option. |

The class auto-selects the best available model.

### Download

```bash
# Option 1: Use setup script
python -m onnx_models.setup

# Option 2: Manual download
# From: https://github.com/fabio-sim/LightGlue-ONNX/releases
```

---

## Usage Examples

### Basic Usage (NumPy)

```python
from onnx_models import SuperPointONNX, LightGlueONNX
import cv2

# Load models
superpoint = SuperPointONNX()
lightglue = LightGlueONNX(features="superpoint")

# Load images
img0 = cv2.imread("image0.jpg", cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
img1 = cv2.imread("image1.jpg", cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0

# Extract features
kp0, scores0, desc0 = superpoint.extract(img0)
kp1, scores1, desc1 = superpoint.extract(img1)

# Match features
matches, match_scores = lightglue.match(
    kp0, kp1, desc0, desc1,
    image_size0=img0.shape[:2],  # (H, W)
    image_size1=img1.shape[:2],
)

print(f"Found {len(matches)} matches")
# matches[i] = [idx0, idx1] means kp0[idx0] ↔ kp1[idx1]
```

### GPU Usage (Zero-Copy)

```python
import torch

# Features already on GPU
kp0_gpu = torch.from_numpy(kp0).cuda()
desc0_gpu = torch.from_numpy(desc0).cuda()
kp1_gpu = torch.from_numpy(kp1).cuda()
desc1_gpu = torch.from_numpy(desc1).cuda()

# Match with IO binding (avoids GPU→CPU→GPU copies)
matches, scores = lightglue.match_gpu(
    kp0_gpu, kp1_gpu, desc0_gpu, desc1_gpu,
    image_size0=(H0, W0),
    image_size1=(H1, W1),
)
```

---

## Technical Notes

### Why TensorRT?

| Provider | Latency | Notes |
|----------|---------|-------|
| PyTorch | ~25ms | Baseline |
| CUDA EP | ~15ms | 1.7x faster |
| TensorRT EP | ~8ms | 3x faster |

TensorRT fuses operations and optimizes memory access patterns.

### Why FP16?

LightGlue enables FP16 in TensorRT (`trt_fp16=True`).

- Reduces memory bandwidth
- Uses Tensor Cores on RTX GPUs
- No accuracy loss for this model

### Why Optimization Level 3?

```python
"trt_builder_optimization_level": 3
```

Level 5 (max) causes "Myelin stream capture" conflicts with dynamic shapes.
Level 3 avoids this issue while maintaining good performance.

### Keypoint Normalization

The model expects keypoints in [-1, 1] range.

```
normalized = 2.0 * (pixel_coords / image_size) - 1.0
```

The `match()` method does this automatically when you provide `image_size0/1`.

### IO Binding

`match_gpu()` uses ONNX Runtime IO Binding.

**Without IO Binding:**
```
GPU tensor → CPU numpy → ONNX → CPU numpy → GPU tensor
```

**With IO Binding:**
```
GPU tensor → ONNX (reads directly) → CPU numpy (outputs only)
```

Outputs stay on CPU because match count varies (can't pre-allocate GPU buffer).

---

## Limitations

| Limitation | Reason | Workaround |
|------------|--------|------------|
| Max 4096 keypoints | TensorRT profile limit | Increase max_shapes, rebuild engine |
| First inference slow | TensorRT engine build | Use warmup(), cache engines |
| Outputs on CPU | Variable output size | None (acceptable overhead) |

---

## Troubleshooting

### "TensorRT engine build failed"

- Check CUDA/TensorRT versions match
- Delete `.trt_cache/` and retry
- Fall back to CUDA provider (rename .trt.onnx file)

### "Out of memory"

- Reduce max keypoints in TRT profile
- Use smaller images
- Set `trt_context_memory_sharing_enable=True` (default)

### "Slow first inference"

- Normal for TensorRT. Engine builds on first run.
- Call `warmup()` after loading.
- Engine is cached for subsequent runs.

---

## File Structure

```
lightglue_onnx/
├── __init__.py          # Exports LightGlueONNX
├── model.py             # LightGlueONNX class
├── README.md            # This file
├── setup.py             # Download script
└── models/
    ├── lightglue_onnx.trt.onnx
    └── .trt_cache/      # TensorRT engine cache
```
