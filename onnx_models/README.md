# ONNX Models

Accelerated visual localization models using ONNX Runtime with TensorRT support.

## Models

| Model | Purpose | Input | Output | Speedup |
|-------|---------|-------|--------|---------|
| SuperPoint | Keypoint detection | Grayscale image | Keypoints, scores, 256-dim descriptors | ~2x |
| LightGlue | Feature matching | Two sets of keypoints + descriptors | Match indices and scores | ~3x |
| MegaLoc | Global descriptors | RGB image | 8448-dim descriptor | ~8x |

## Quick Start

```python
from onnx_models import SuperPointONNX, LightGlueONNX, MegaLocONNX

# Feature extraction
sp = SuperPointONNX()
keypoints, scores, descriptors = sp.extract(grayscale_image)

# Feature matching
lg = LightGlueONNX()
matches, scores = lg.match(kp0, kp1, desc0, desc1)

# Global descriptor
ml = MegaLocONNX()
descriptor = ml.extract(rgb_image)
```

## Setup

Download and setup all models:

```bash
python -m onnx_models.setup
```

This downloads SuperPoint and LightGlue, and exports MegaLoc from PyTorch.

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
├── setup.py             # Root setup
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
    "model": {"name": "superpoint_onnx", "max_num_keypoints": 4096},
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

TensorRT provides the best performance but has caveats:
- First inference is slow (~30s) while building the engine
- Engine is cached in `.trt_cache/` for subsequent runs
- Some models need FP32 (FP16 can cause numerical issues)

For MegaLoc specifically, TensorRT FP16 is disabled due to NaN outputs in the
DINOv2 backbone. This still provides ~8x speedup over PyTorch.
