# ONNX Models

Accelerated visual localization models using ONNX Runtime with TensorRT support.

## Models

| Model | Purpose | Input | Output | TensorRT | Precision |
|-------|---------|-------|--------|----------|-----------|
| SuperPoint | Keypoint detection | Grayscale image | Keypoints, scores, 256-dim descriptors | ✓ | FP16 |
| LightGlue | Feature matching | Two sets of keypoints + descriptors | Match indices and scores | ✓ | FP16 |
| MegaLoc | Global descriptors | RGB image | 8448-dim descriptor | ✓ | FP32 |

## Quick Start

```python
from onnx_models import SuperPointONNX, LightGlueONNX, MegaLocONNX

# Feature extraction (with TensorRT)
sp = SuperPointONNX(use_tensorrt=True)
keypoints, scores, descriptors = sp.extract(grayscale_image)

# Feature matching
lg = LightGlueONNX()
matches, scores = lg.match(kp0, kp1, desc0, desc1)

# Global descriptor
ml = MegaLocONNX(use_tensorrt=True)
descriptor = ml.extract(rgb_image)
```

## Setup

Download and setup all models:

```bash
python -m onnx_models.setup
```

This downloads SuperPoint and LightGlue, and exports MegaLoc from PyTorch.

### Pre-build TensorRT Engines

TensorRT engine compilation requires ~6GB GPU memory. Pre-build them once to avoid OOM during inference:

```bash
# Inside Docker container (after running ./scripts/launch_docker.sh):
python3 onnx_models/build_trt_engines.py

# Or as a one-liner from the host:
docker run --gpus all --rm -v "$(pwd)":/app -w /app hloc:latest python3 onnx_models/build_trt_engines.py
```

This caches engines in each model's `models/.trt_cache/` directory. Subsequent runs load instantly.

If you don't have PyTorch/CUDA:

```bash
python -m onnx_models.setup --skip-megaloc
```

Setup individual models:

```bash
python -m onnx_models.superpoint_onnx.setup
python -m onnx_models.lightglue_onnx.setup
python -m onnx_models.megaloc_onnx.setup
```

## Directory Structure

```
onnx_models/
├── __init__.py          # Public API
├── utils.py             # Shared utilities
├── hloc_wrappers.py     # hloc integration
├── setup.py             # Download/export all models
├── build_trt_engines.py # Pre-build TensorRT engines
│
├── superpoint_onnx/
│   ├── __init__.py
│   ├── model.py         # SuperPointONNX class
│   ├── setup.py         # Download model
│   ├── models/          # Model files
│   └── README.md
│
├── lightglue_onnx/
│   ├── __init__.py
│   ├── model.py         # LightGlueONNX class
│   ├── setup.py         # Download model
│   ├── models/          # Model files + TRT cache
│   └── README.md
│
└── megaloc_onnx/
    ├── __init__.py
    ├── model.py         # MegaLocONNX class
    ├── setup.py         # Export from PyTorch
    ├── export.py        # Advanced export options
    ├── models/          # Model files + TRT cache
    └── README.md
```

## hloc Integration

These models integrate with hloc through wrappers in `hloc/extractors/` and `hloc/matchers/`:

```python
# In extract_features.py
confs["superpoint_onnx"] = {
    "output": "feats-superpoint-onnx",
    "model": {"name": "superpoint_onnx", "max_num_keypoints": 4096, "use_tensorrt": True},
    "preprocessing": {"grayscale": True, "resize_max": 1024},
}

confs["megaloc_onnx"] = {
    "output": "global-feats-megaloc",
    "model": {"name": "megaloc_onnx", "resize": 322, "use_tensorrt": True},
    "preprocessing": {"resize_max": 1024},
}

# In match_features.py
confs["superpoint_onnx+lightglue_onnx"] = {
    "output": "matches-sp-lg-onnx",
    "model": {"name": "lightglue_onnx", "features": "superpoint"},
}
```

## Requirements

```
onnxruntime-gpu>=1.16.0
numpy
opencv-python
torch  # For MegaLoc export only
```

TensorRT support requires `onnxruntime-gpu` built with TensorRT. Check with:

```python
import onnxruntime
print(onnxruntime.get_available_providers())
# Should include 'TensorrtExecutionProvider'
```

## Performance Notes

**TensorRT Acceleration:**
- Provides 2-8x speedup over CUDA execution provider
- First run compiles the engine (~30s), subsequent runs load from cache
- LightGlue requires ~6GB GPU memory during engine compilation
- Run `build_trt_engines.py` with an empty GPU to avoid OOM

**Precision:**
- SuperPoint and LightGlue use FP16 for speed
- MegaLoc uses FP32 (DINOv2 produces NaN with FP16)

**Memory:**
- Engine compilation is memory-intensive but transient
- Runtime usage is much lower than compilation
- Delete `.trt_cache/` directories to force engine rebuild
