# ONNX Accelerated Visual Localization - Knowledge Base

**Last Updated:** 2026-08-09

This document captures all knowledge gained during the development of ONNX-accelerated visual localization models, including TensorRT optimization, debugging sessions, and architectural decisions.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Models](#models)
4. [TensorRT Integration](#tensorrt-integration)
5. [Critical Bugs and Solutions](#critical-bugs-and-solutions)
6. [Performance](#performance)
7. [Reference Feature Caching](#reference-feature-caching)
8. [Code Structure](#code-structure)
9. [hloc Integration](#hloc-integration)
10. [Docker Setup](#docker-setup)
11. [ONNX Runtime Configuration](#onnx-runtime-configuration)
12. [Lessons Learned](#lessons-learned)

---

## Overview

### What We Built

An ONNX Runtime / TensorRT accelerated inference pipeline for visual localization, consisting of:

1. **SuperPoint ONNX** - Local feature detection and description
2. **LightGlue ONNX** - Feature matching with attention
3. **MegaLoc ONNX** - Global descriptor extraction for image retrieval

### Why ONNX?

| Aspect | PyTorch | ONNX (CUDA) | ONNX (TensorRT) |
|--------|---------|-------------|-----------------|
| SuperPoint | ~8ms | ~4ms | ~4ms |
| LightGlue | ~25ms | ~15ms | ~8ms |
| MegaLoc | ~45ms | ~12ms | ~6ms |
| **Total Pipeline** | ~2.3s/image | ~1.8s/image | ~1.57s/image |

TensorRT provides the best speedup but requires careful configuration.

---

## Architecture

### Visual Localization Pipeline

```
Query Image
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │ (PARALLEL)
    ▼                  ▼                  │
MegaLoc          SuperPoint               │
    │                  │                  │
    ▼                  │                  │
Global Descriptor      │                  │
(8448-dim)            │                  │
    │                  │                  │
    ▼                  │                  │
Image Retrieval        │                  │
(find similar         │                  │
 map images)          │                  │
    │                  │                  │
    └──────────────────┼──────────────────┘
                       │
                       ▼
               Keypoints + Descriptors
                       │
                       ▼
               LightGlue Matching
               (match against retrieved images)
                       │
                       ▼
               PnP + RANSAC
               (estimate camera pose)
                       │
                       ▼
               6-DoF Camera Pose
```

### Model Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        ONNX Models                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  SuperPoint  │    │  LightGlue   │    │   MegaLoc    │       │
│  │    ONNX      │    │    ONNX      │    │    ONNX      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │    CUDA      │    │  TensorRT    │    │  TensorRT    │       │
│  │   Provider   │    │   Provider   │    │   Provider   │       │
│  │              │    │  (FP32+FP16) │    │  (FP32 only) │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Models

### SuperPoint ONNX

**Source:** [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) releases

**Architecture:**
- Encoder: VGG-style CNN
- Output: Keypoints (x, y), scores, 256-dim descriptors

**Input:**
- Grayscale image: `[1, 1, H, W]` float32, values in `[0, 1]`

**Output:**
- `keypoints`: `[N, 2]` pixel coordinates
- `scores`: `[N]` detection confidence
- `descriptors`: `[N, 256]` L2-normalized

**Configuration:**
```python
SuperPointONNX(
    model_path=None,           # Auto-detects in models/
    max_num_keypoints=2048,    # Top-K keypoints
    detection_threshold=0.0005,
    device="cuda",
)
```

**Provider:** CUDAExecutionProvider (TensorRT not beneficial for this model)

---

### LightGlue ONNX

**Source:** [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX) releases

**Architecture:**
- Transformer-based attention matcher
- Self-attention + cross-attention layers
- Optimal transport for final assignment

**Input:**
- `kpts0`, `kpts1`: `[1, N, 2]` keypoint coordinates (normalized to [-1, 1])
- `desc0`, `desc1`: `[1, N, 256]` descriptors
- `image0_shape`, `image1_shape`: `[1, 2]` (H, W)

**Output:**
- `matches0`: `[M, 2]` match indices
- `matching_scores0`: `[M]` confidence scores

**Configuration:**
```python
LightGlueONNX(
    model_path=None,
    features="superpoint",  # or "disk", "aliked"
    device="cuda",
    trt_cache_dir=None,     # For TensorRT engine cache
)
```

**Provider:** TensorrtExecutionProvider with dynamic shape profiles

**TensorRT Configuration:**
```python
{
    "device_id": 0,
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": ".trt_cache/",
    "trt_timing_cache_enable": True,
    "trt_builder_optimization_level": 5,
    "trt_profile_min_shapes": "kpts0:1x1x2,kpts1:1x1x2,desc0:1x1x256,desc1:1x1x256",
    "trt_profile_opt_shapes": "kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x256,desc1:1x512x256",
    "trt_profile_max_shapes": "kpts0:1x8192x2,kpts1:1x8192x2,desc0:1x8192x256,desc1:1x8192x256",
}
```

---

### MegaLoc ONNX

**Source:** Exported from PyTorch via `torch.onnx.export`

**Architecture:**
- Backbone: DINOv2 ViT-B/14 (Vision Transformer)
- Pooling: GeM (Generalized Mean Pooling)
- Output: 8448-dimensional global descriptor

**Input:**
- RGB image: `[1, 3, 322, 322]` float32
- Normalized with ImageNet mean/std

**Output:**
- `descriptors`: `[1, 8448]` L2-normalized

**Why 322x322?**
- ViT-B/14 uses 14x14 patches
- 322 = 23 × 14 = 529 patches (23x23 grid)
- Matches MegaLoc training configuration

**Configuration:**
```python
MegaLocONNX(
    model_path=None,
    device="cuda",
    use_tensorrt=True,  # FP32 mode forced
)
```

**CRITICAL: FP16 is DISABLED for MegaLoc!** See [Critical Bugs](#critical-bugs-and-solutions).

---

## TensorRT Integration

### How TensorRT Works with ONNX Runtime

1. ONNX Runtime loads the `.onnx` model
2. TensorRT EP parses the graph and builds an optimized engine
3. Engine is cached to `.trt_cache/` for subsequent runs
4. First inference is slow (~30s) while building; subsequent runs are fast

### TensorRT Provider Configuration

```python
providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
provider_options = [
    {
        "device_id": 0,
        "trt_fp16_enable": True,           # Enable FP16 (NOT for MegaLoc!)
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir),
        "trt_builder_optimization_level": 5,  # Max optimization
    },
    {"device_id": 0},  # CUDA fallback
    {},                # CPU fallback
]
```

### Dynamic Shapes

LightGlue requires dynamic shapes because keypoint counts vary per image:

```python
"trt_profile_min_shapes": "kpts0:1x1x2,kpts1:1x1x2,desc0:1x1x256,desc1:1x1x256",
"trt_profile_opt_shapes": "kpts0:1x512x2,kpts1:1x512x2,desc0:1x512x256,desc1:1x512x256",
"trt_profile_max_shapes": "kpts0:1x8192x2,kpts1:1x8192x2,desc0:1x8192x256,desc1:1x8192x256",
```

- **min**: Minimum expected shape (1 keypoint)
- **opt**: Most common shape (512 keypoints) - TensorRT optimizes for this
- **max**: Maximum shape (8192 keypoints)

### Cache Files

TensorRT generates several cache files:
```
.trt_cache/
├── TensorrtExecutionProvider_cache_sm89.timing     # Timing cache
├── TensorrtExecutionProvider_TRTKernel_*.engine   # Compiled engine
└── TensorrtExecutionProvider_TRTKernel_*.profile  # Profile data
```

**Important:** Cache is GPU-specific! An engine built for RTX 4090 (sm89) won't work on other GPUs.

### TensorRT Operator Limitations

**Not all ONNX models can use TensorRT.** Some operators are unsupported:

| Model | TensorRT Support | Issue |
|-------|------------------|-------|
| **LightGlue** | ✅ YES | Specifically exported for TRT (`.trt.onnx` models from LightGlue-ONNX) |
| **SuperPoint** | ❌ NO | Uses `ScatterND` with `reduction` attribute - unsupported by TRT |
| **MegaLoc (DINOv2)** | ❌ NO | Same `ScatterND` with `reduction` issue |

**Error message:**
```
[ERROR] In node 87 with name: /ScatterND and operator: ScatterND (importScatterND): 
UNSUPPORTED_NODE_ATTR: Assertion failed: !attrs.count("reduction"): 
Attribute reduction is not supported.
```

**Root cause:** TensorRT's ONNX parser doesn't support `ScatterND` with the `reduction` attribute (even `reduction='none'`). This is a limitation in TensorRT 10.x and earlier.

**Workarounds:**
1. Re-export models avoiding `ScatterND` with reduction (requires modifying source PyTorch code)
2. Use a custom TensorRT plugin
3. Accept CUDA EP for these models (current approach - still fast)

**Current configuration (optimal):**
- SuperPoint: CUDA EP (~4ms)
- MegaLoc: CUDA EP (~12ms)
- LightGlue: TensorRT EP (~8ms) ← Only model using TRT

See also: [NVIDIA TensorRT GitHub Issue #4425](https://github.com/NVIDIA/TensorRT/issues/4425)

---

## Critical Bugs and Solutions

### Bug #1: MegaLoc TensorRT FP16 Produces NaN

**Symptom:**
```python
descriptor = megaloc.extract(image)
print(np.linalg.norm(descriptor))  # Returns nan
print(np.isnan(descriptor).any())  # True
```

**Root Cause:**
DINOv2 ViT-B/14 backbone has numerical instability in FP16 mode. The attention computations involve very small values that underflow to zero or overflow to infinity when using half precision.

**Investigation Process:**
1. Verified ONNX model works with CUDA provider (no NaN)
2. Verified TensorRT FP32 works (no NaN)
3. TensorRT FP16 produces NaN consistently
4. Issue is in the DINOv2 backbone, not the ONNX export

**Solution:**
Force FP32 for MegaLoc TensorRT:
```python
providers, provider_options = get_onnx_providers(
    device="cuda",
    use_tensorrt=True,
    trt_fp16=False,  # CRITICAL: Must be False for MegaLoc
)
```

**Performance Impact:**
- FP32 TensorRT: ~6ms/image
- PyTorch: ~45ms/image
- Still provides ~8x speedup, just not the additional ~2x from FP16

---

### Bug #2: TensorRT Provider Format Error

**Symptom:**
```
TypeError: 'tuple' object cannot be interpreted as an integer
```

**Root Cause:**
ONNX Runtime 1.28+ changed how provider options are passed. The old format:
```python
providers = [
    ("TensorrtExecutionProvider", {...options...}),
    ("CUDAExecutionProvider", {...}),
]
```

Doesn't work. Must use:
```python
providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
provider_options = [{...trt options...}, {...cuda options...}]

session = ort.InferenceSession(
    model_path,
    providers=providers,
    provider_options=provider_options,  # Separate parameter!
)
```

---

### Bug #3: Invalid TensorRT Option `trt_log_level`

**Symptom:**
```
Invalid TensorRT EP option: trt_log_level
```

**Root Cause:**
ONNX Runtime 1.28+ removed `trt_log_level` option. Use session options instead:
```python
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 4  # 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
```

---

### Bug #4: cuDNN Missing in Docker

**Symptom:**
```
dlopen failed for libcudnn.so: libcudnn.so: cannot open shared object file
```

**Solution:**
Install cuDNN via pip and set library path:
```bash
pip install nvidia-cudnn-cu12
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

---

### Bug #5: ONNX Runtime CUDA 12 Compatibility

**Symptom:**
```
Require cuDNN 9.* and CUDA 13.*
```

**Root Cause:**
Default PyPI `onnxruntime-gpu` builds for CUDA 11 or CUDA 13. Docker has CUDA 12.

**Solution:**
Use Microsoft's CUDA 12 specific wheel:
```bash
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
```

---

## Performance

### Benchmark Results (RTX 4090, 640x480 images)

| Component | PyTorch | ONNX CUDA | ONNX TensorRT |
|-----------|---------|-----------|---------------|
| SuperPoint | 8ms | 4ms | 4ms |
| MegaLoc | 45ms | 12ms | 6ms (FP32) |
| LightGlue | 25ms | 15ms | 8ms |
| Image I/O | 50ms | 50ms | 50ms |
| HDF5 I/O | 100ms | 100ms | 20ms* |

*HDF5 I/O reduced by pre-loading features into GPU memory

### Full Pipeline Timing (my_office map)

**With IO Binding + Parallel Feature Extraction (current):**
```
Average per query: 1.18s
├── Feature extraction (parallel): 0.31s
│   ├── SuperPoint:              (0.28s)
│   └── MegaLoc:                 (0.31s)
│   └── Saved by parallelism:    (0.28s)
├── Pairs retrieval:             0.10s
├── LightGlue matching:          0.53s
└── Localization (PnP+RANSAC):   0.23s
```

**Without IO Binding (parallel only):**
```
Average per query: 1.26s
├── Feature extraction (parallel): 0.37s
│   ├── SuperPoint:              (0.35s)
│   └── MegaLoc:                 (0.31s)
├── Pairs retrieval:             0.11s
├── LightGlue matching:          0.61s
└── Localization (PnP+RANSAC):   0.23s
```

**Sequential (baseline, no optimizations):**
```
Average per query: 1.57s
├── SuperPoint extraction:     0.32s
├── MegaLoc extraction:        0.25s
├── Pairs retrieval:           0.12s
├── LightGlue matching:        0.63s
└── Localization (PnP+RANSAC): 0.25s
```

**Cumulative speedup:**
- Parallel execution: 1.57s → 1.32s (16% faster)
- IO Binding: 1.26s → 1.18s (6% faster)
- **Total: 1.57s → 1.18s (25% faster)**

### Memory Usage

| Model | VRAM (FP32) | VRAM (FP16) |
|-------|-------------|-------------|
| SuperPoint | ~200MB | N/A |
| LightGlue | ~500MB | ~300MB |
| MegaLoc | ~800MB | N/A (FP16 broken) |

---

## Reference Feature Caching

### Problem

During localization, each query image is matched against ~30 retrieved reference images. With multiple queries, the same reference images are often re-used (high overlap in retrieval results). Loading features from HDF5 for each match is expensive (~100ms I/O per image).

### Solution: LRU Cache

A simple LRU (Least Recently Used) cache keeps frequently-accessed reference features in GPU memory:

```python
# hloc/utils/cache.py
class LRUCache:
    """GPU-aware LRU cache using OrderedDict."""
    
    def __init__(self, max_items=200, device="cuda"):
        self.max_items = max_items
        self.device = device
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key, load_fn):
        """Get item from cache, or load via load_fn on miss."""
        if key in self.cache:
            self.cache.move_to_end(key)  # Mark as recently used
            self.hits += 1
            return self.cache[key]
        
        # Cache miss - load and store
        self.misses += 1
        value = load_fn(key)
        
        # Move tensors to GPU
        if isinstance(value, dict):
            value = {k: v.to(self.device) if hasattr(v, 'to') else v 
                     for k, v in value.items()}
        
        # Evict LRU item if at capacity
        if len(self.cache) >= self.max_items:
            self.cache.popitem(last=False)
        
        self.cache[key] = value
        return value
    
    def stats(self):
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'max_size': self.max_items,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f'{100*self.hits/total:.1f}%' if total > 0 else '0%'
        }
```

### Integration

The cache is passed as an optional parameter through the call chain:

```python
# inference/inference.py
from hloc.utils.cache import LRUCache

ref_cache = LRUCache(max_items=200, device="cuda")

for query in queries:
    match_features.main(..., ref_cache=ref_cache)

print(f"Cache stats: {ref_cache.stats()}")
```

```python
# hloc/match_features.py - modified to accept cache
def match_from_paths_fast(..., ref_cache=None):
    if ref_cache is not None:
        def load_ref(name):
            grp = features1[name]
            return {k: torch.from_numpy(np.array(v)) for k, v in grp.items()}
        ref_features = {name: ref_cache.get(name, load_ref) for name in ref_names}
    else:
        # Original HDF5 loading
        ...
```

### Performance

With 3 query images against ~30 refs each:
```
Cache stats: {'size': 84, 'max_size': 200, 'hits': 6, 'misses': 84, 'hit_rate': '6.7%'}
```

- 84 unique refs loaded (misses)
- 6 refs shared across queries (hits)
- Hit rate improves significantly with more queries sharing the same map

### Design Choices

1. **Optional parameter** - Cache is opt-in, no API breakage
2. **Caller creates cache** - Lives in `inference.py`, not module-level state
3. **`get(key, load_fn)` pattern** - All cache logic in one place, caller just calls it
4. **GPU-aware** - Auto-moves tensors to device on cache miss
5. **LRU eviction** - Bounded memory usage via `max_items`

---

## Code Structure

### Directory Layout

```
onnx_models/
├── __init__.py              # Public API: SuperPointONNX, LightGlueONNX, MegaLocONNX
├── utils.py                 # get_onnx_providers(), create_session()
├── hloc_wrappers.py         # BaseModel wrappers for hloc integration
├── setup.py                 # Root setup (calls each model's setup)
├── README.md
│
├── superpoint_onnx/
│   ├── __init__.py
│   ├── model.py             # SuperPointONNX class
│   ├── setup.py             # Downloads from GitHub releases
│   ├── models/
│   │   └── superpoint_onnx.onnx
│   └── README.md
│
├── lightglue_onnx/
│   ├── __init__.py
│   ├── model.py             # LightGlueONNX class
│   ├── setup.py             # Downloads from GitHub releases
│   ├── models/
│   │   ├── lightglue_onnx.trt.onnx
│   │   └── .trt_cache/      # TensorRT engine cache
│   └── README.md
│
└── megaloc_onnx/
    ├── __init__.py
    ├── model.py             # MegaLocONNX class
    ├── setup.py             # Exports from PyTorch
    ├── export.py            # Advanced export options
    ├── models/
    │   ├── megaloc_onnx.onnx
    │   ├── megaloc_onnx.onnx.data  # External weights
    │   └── .trt_cache/
    └── README.md
```

### Key Design Decisions

1. **Per-model folders** with their own `setup.py` for modularity
2. **Models stored in each folder** (`superpoint_onnx/models/`) not centralized
3. **TRT cache alongside models** for easy cleanup
4. **Naming convention**: `*_onnx.onnx` to identify ONNX files easily

### Public API

```python
from onnx_models import SuperPointONNX, LightGlueONNX, MegaLocONNX

# Feature extraction
sp = SuperPointONNX()
keypoints, scores, descriptors = sp.extract(grayscale_image)

# Matching
lg = LightGlueONNX()
matches, scores = lg.match(kp0, kp1, desc0, desc1, size0, size1)

# Global descriptors
ml = MegaLocONNX()
descriptor = ml.extract(rgb_image)
```

---

## hloc Integration

### Architecture

hloc uses `BaseModel` as the interface for all models. ONNX models don't inherit from `BaseModel` (they're not PyTorch modules), so we use wrapper classes.

```
hloc/extractors/superpoint_onnx.py ──► onnx_models.hloc_wrappers.SuperPointONNXWrapper
hloc/extractors/megaloc_onnx.py   ──► onnx_models.hloc_wrappers.MegaLocONNXWrapper
hloc/matchers/lightglue_onnx.py   ──► onnx_models.hloc_wrappers.LightGlueONNXWrapper
```

### Wrapper Pattern

```python
class SuperPointONNXWrapper:
    """Wraps SuperPointONNX for hloc's BaseModel interface."""
    
    default_conf = {...}
    required_inputs = ["image"]
    
    def __init__(self, conf):
        self._model = SuperPointONNX(...)
    
    def eval(self):
        return self  # No-op, ONNX is always "eval"
    
    def to(self, device):
        return self  # No-op, device set at init
    
    def __call__(self, data):
        # Convert hloc format to ONNX format
        image = data["image"]
        kp, scores, desc = self._model.extract(image)
        return {"keypoints": kp, "scores": scores, "descriptors": desc}
```

### Configuration

```python
# In hloc/extract_features.py
confs["superpoint_onnx"] = {
    "output": "feats-superpoint-onnx-n2048-r1024",
    "model": {"name": "superpoint_onnx", "max_num_keypoints": 2048},
    "preprocessing": {"grayscale": True, "resize_max": 1024},
}

confs["megaloc_onnx"] = {
    "output": "global-feats-megaloc-onnx",
    "model": {"name": "megaloc_onnx"},
    "preprocessing": {"resize_max": 1024},
}

# In hloc/match_features.py
confs["superpoint_onnx+lightglue_onnx"] = {
    "output": "matches-superpoint-lightglue-onnx",
    "model": {"name": "lightglue_onnx", "features": "superpoint"},
}
```

### Shared `load_model` Function

```python
# hloc/utils/base_model.py
def load_model(module, conf, device=None):
    """Load a model from a module (extractors or matchers)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(module, conf["model"]["name"])
    return Model(conf["model"]).eval().to(device)

# hloc/extract_features.py
def get_model(conf, device=None):
    return load_model(extractors, conf, device)

# hloc/match_features.py
def get_model(conf, device=None):
    return load_model(matchers, conf, device)
```

---

## Docker Setup

### Base Image

```dockerfile
FROM colmap/colmap:latest
# Has CUDA 12.9.1, but no cuDNN
```

### Required Installations

```dockerfile
# cuDNN for ONNX Runtime CUDA EP
RUN pip install nvidia-cudnn-cu12

# ONNX Runtime with CUDA 12 support
RUN pip install onnxruntime-gpu \
    --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Set library path
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

### TensorRT Version Compatibility

**Critical:** ONNX Runtime requires a specific TensorRT version. ORT 1.28 requires TensorRT **10.x** (`libnvinfer.so.10`).

**Symptom:**
```
Failed to load library libonnxruntime_providers_tensorrt.so with error: 
libnvinfer.so.10: cannot open shared object file: No such file or directory
```

**Cause:** TensorRT 11.x installed (provides `libnvinfer.so.11`), but ORT 1.28 links against `libnvinfer.so.10`.

**Solution:** Pin TensorRT to 10.7.0:

```dockerfile
# First, set global pip config to allow break-system-packages (needed for tensorrt build)
RUN mkdir -p /root/.config/pip && \
    echo '[global]\nbreak-system-packages = true' > /root/.config/pip/pip.conf

# Install TensorRT 10.7.0 (compatible with ORT 1.28)
RUN pip3 install tensorrt-cu12==10.7.0 tensorrt-cu12-bindings==10.7.0 tensorrt-cu12-libs==10.7.0

# Add TensorRT libs to LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/tensorrt_libs:$LD_LIBRARY_PATH
```

**Note:** The `tensorrt-cu12` package internally calls pip during build, so the global pip config must be set first.

### Docker Run Command

```bash
docker run -it --gpus all \
    -v $(pwd):/app \
    -w /app \
    hloc-inference:latest \
    bash
```

### Verify GPU Access

```python
import onnxruntime as ort
print(ort.get_available_providers())
# Should include: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

---

## ONNX Runtime Configuration

### Session Options

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.log_severity_level = 3  # ERROR level
sess_options.intra_op_num_threads = 4
sess_options.inter_op_num_threads = 4
```

### Provider Priority

ONNX Runtime tries providers in order:
1. TensorrtExecutionProvider - Best performance (if available)
2. CUDAExecutionProvider - Good fallback
3. CPUExecutionProvider - Always available

To check which provider is active:
```python
session = ort.InferenceSession(model_path, ...)
print(session.get_providers())  # Active providers
```

### Environment Variables

```bash
# Disable TensorRT warnings
export ORT_TENSORRT_LOG_LEVEL=ERROR

# CUDA device selection
export CUDA_VISIBLE_DEVICES=0

# cuDNN library path
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

### IO Binding for GPU Tensors

**Problem:** When input data is already on GPU (PyTorch tensors), the standard ONNX Runtime API requires a CPU roundtrip:

```python
# Without IO Binding (slow):
tensor_gpu = torch.tensor(..., device='cuda')
numpy_cpu = tensor_gpu.cpu().numpy()           # GPU → CPU copy
outputs = session.run(None, {'input': numpy_cpu})  # CPU → GPU copy internally
```

**Solution:** IO Binding allows ONNX Runtime to read directly from GPU memory:

```python
# With IO Binding (fast):
io_binding = session.io_binding()

io_binding.bind_input(
    name='input',
    device_type='cuda',
    device_id=0,
    element_type=np.float32,
    shape=tuple(tensor_gpu.shape),
    buffer_ptr=tensor_gpu.data_ptr(),  # Direct GPU pointer!
)

# Let ONNX allocate outputs on CPU (for variable-size outputs)
io_binding.bind_output('output', device_type='cpu')

session.run_with_iobinding(io_binding)
outputs = io_binding.copy_outputs_to_cpu()
```

**Key requirements:**
- Tensor must be **contiguous** (`tensor.contiguous()`)
- Tensor must be **float32** (`.float()`)
- Use `.data_ptr()` to get the raw GPU pointer
- Provider must be CUDA or TensorRT (CPU provider can't read GPU memory)

**Implementation in our models:**

```python
# superpoint_onnx/model.py
def extract_gpu(self, image: torch.Tensor):
    """Extract features using IO Binding (avoids CPU copies)."""
    io_binding = self.session.io_binding()
    io_binding.bind_input(
        name='image',
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        shape=tuple(image.shape),
        buffer_ptr=image.data_ptr(),
    )
    ...

# lightglue_onnx/model.py  
def match_gpu(self, kp0, kp1, desc0, desc1, ...):
    """Match using IO Binding - 4 inputs bound directly."""
    io_binding = self.session.io_binding()
    for name, tensor in [('kpts0', kp0), ('kpts1', kp1), 
                         ('desc0', desc0), ('desc1', desc1)]:
        io_binding.bind_input(name, 'cuda', 0, np.float32,
                              tuple(tensor.shape), tensor.data_ptr())
    ...
```

**Performance impact:**

| Model | Without IO Binding | With IO Binding | Savings |
|-------|-------------------|-----------------|---------|
| SuperPoint (first run) | 0.54s | 0.34s | **-0.20s** |
| SuperPoint (avg) | 0.35s | 0.28s | **-0.07s** |
| LightGlue (per match) | ~35ms | ~28ms | **-7ms** |
| LightGlue (30 matches) | 0.61s | 0.53s | **-0.08s** |

**Total pipeline improvement:** ~1.26s → ~1.18s per image

**Output quality:** Unchanged. IO Binding only affects data transfer, not computation.

---

## Lessons Learned

### 1. TensorRT FP16 is Not Always Safe

**Lesson:** Always test FP16 outputs against FP32. Some models (especially ViTs) have numerical issues.

**How to verify:**
```python
# FP32 inference
desc_fp32 = model_fp32.extract(image)

# FP16 inference  
desc_fp16 = model_fp16.extract(image)

# Check
print(f"FP32 norm: {np.linalg.norm(desc_fp32)}")  # Should be ~1.0
print(f"FP16 norm: {np.linalg.norm(desc_fp16)}")  # NaN = problem!
print(f"Any NaN: {np.isnan(desc_fp16).any()}")
```

### 2. TensorRT Caching is Essential

**Lesson:** Without caching, TensorRT rebuilds engines every run (~30s overhead).

**Solution:**
```python
"trt_engine_cache_enable": True,
"trt_engine_cache_path": str(cache_dir),
"trt_timing_cache_enable": True,
```

### 3. Dynamic Shapes Need Profiles

**Lesson:** TensorRT requires min/opt/max shapes for dynamic dimensions.

**Solution:**
```python
"trt_profile_min_shapes": "input:1x1x256",
"trt_profile_opt_shapes": "input:1x512x256",  # Optimize for common case
"trt_profile_max_shapes": "input:1x8192x256",
```

### 4. ONNX Runtime Provider Options Changed

**Lesson:** API changed between versions. Check documentation for your version.

**Old (broken):**
```python
providers = [("TensorrtExecutionProvider", {...})]
```

**New (correct):**
```python
providers = ["TensorrtExecutionProvider"]
provider_options = [{...}]
session = ort.InferenceSession(..., providers=providers, provider_options=provider_options)
```

### 5. Per-Model Organization is Better

**Lesson:** Centralizing all models in one folder makes it hard to manage dependencies and versions.

**Better:**
```
onnx_models/
├── superpoint_onnx/
│   ├── models/
│   └── setup.py
├── lightglue_onnx/
│   ├── models/
│   └── setup.py
```

### 6. Wrapper Classes for Framework Integration

**Lesson:** ONNX models don't fit existing frameworks (hloc expects BaseModel). Use wrappers.

```python
class MyONNXWrapper:
    def __init__(self, conf):
        self._model = ActualONNXModel(...)
    
    def eval(self): return self
    def to(self, device): return self
    def __call__(self, data): ...
```

### 7. Pre-load Features for Inference

**Lesson:** HDF5 I/O is slow. For inference, load all features into GPU memory first.

```python
# Slow: read from HDF5 each iteration
for pair in pairs:
    features = h5py.File(...)[name]  # Disk I/O
    
# Fast: pre-load once
features = {}
with h5py.File(...) as f:
    for name in names:
        features[name] = {k: torch.tensor(v).cuda() for k, v in f[name].items()}
```

### 8. ONNX Export Requires Care

**Lesson:** PyTorch → ONNX export has many pitfalls.

**Tips:**
- Use `opset_version=18` or higher for PyTorch 2.x
- Set `TORCH_ONNX_USE_NEW_EXPORTER=0` if new exporter fails
- Use `do_constant_folding=True` for smaller models
- Verify with `onnx.checker.check_model()`

### 9. Git Ignore Generated Files

**Lesson:** TensorRT cache is GPU-specific, should never be committed.

```gitignore
# ONNX models
onnx_models/*/models/*.onnx
onnx_models/*/models/*.onnx.data
onnx_models/*/models/.trt_cache/
.trt_cache/
```

### 10. Test in Docker Early

**Lesson:** Local Python environment differs from Docker. Test early.

```bash
docker exec CONTAINER python3 -c "from onnx_models import ..."
```

### 11. Parallelize Independent Model Inference

**Lesson:** SuperPoint and MegaLoc are independent — run them concurrently with `ThreadPoolExecutor`.

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    sp_future = executor.submit(run_superpoint)
    ml_future = executor.submit(run_megaloc)
    
    features = sp_future.result()
    retrieval = ml_future.result()
```

**Why it works:**
- ONNX Runtime releases the GIL during GPU inference
- CUDA can interleave kernels from different streams
- Threading overhead is minimal (~5ms)

**Results:**
- Sequential: 0.36s + 0.31s = **0.67s**
- Parallel: max(0.36s, 0.31s) + overhead = **0.38s**
- **Savings: 0.29s (43% faster)** for feature extraction
- **Overall: 16% faster** (1.57s → 1.32s per image)

### 12. Use LRU Cache for Repeated Data Access

**Lesson:** Reference features are often reused across queries. Cache them in GPU memory.

**Solution:**
```python
# Simple LRU cache with get(key, load_fn) pattern
cache = LRUCache(max_items=200, device="cuda")
features = cache.get(name, lambda n: load_from_hdf5(n))
```

**Key principles:**
- Pass cache as optional parameter (no module-level state)
- Caller creates and owns the cache
- `get(key, load_fn)` handles all hit/miss logic
- Bounded memory via LRU eviction

### 13. Threaded Matching Has Limited Gains

**Lesson:** Running LightGlue matches in parallel threads gives only ~10% speedup.

**Implementation:**
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(match_one_pair, n0, n1): (n0, n1) for n0, n1 in pairs}
    for future in as_completed(futures):
        results.append(future.result())
```

**Why limited gains:**
- GPU is already saturated — parallel calls queue up
- TensorRT execution serializes on the GPU
- Python GIL overhead for data prep still exists
- ONNX Runtime releases GIL, but GPU is the bottleneck

**Results:** 0.61s → 0.55s matching (~10% improvement)

**Better alternatives:**
- Reduce `num_matched` from 30 → 20 (guaranteed 33% faster)
- Batched model export (50-80% faster, but requires padding)

### 14. Pipeline Dependencies Limit Parallelism

**Lesson:** Understanding dependencies prevents wasted optimization effort.

**Current pipeline:**
```
SuperPoint ─────────────▶ WAIT ──▶ Matching
MegaLoc ──▶ Retrieval ──▶ WAIT ──▘
```

**Why we can't parallelize matching with feature extraction:**
1. Matching needs query SuperPoint features
2. Matching needs to know WHICH 30 refs (from MegaLoc retrieval)
3. Retrieval needs MegaLoc output

**Timing analysis:**
- SuperPoint: 0.35s
- MegaLoc + Retrieval: 0.30s + 0.11s = 0.41s

SuperPoint finishes BEFORE we know which refs to match. No overlap possible.

**Theoretical optimization (if SuperPoint were slower):**
```
SuperPoint ──────────────────────────────▶
MegaLoc ──▶ Retrieval ──▶ Preload refs ──▶ Matching
```
But currently SuperPoint is the faster path, so it waits anyway.

### 15. Batched LightGlue Export is Complex

**Lesson:** Batching LightGlue requires handling variable keypoint counts.

**The problem:**
```python
# Each image has different keypoint counts
kpts0: (1, N_query, 2)      # Query: N keypoints
kpts1: (30, ???, 2)         # 30 refs with DIFFERENT counts
```

**Options:**
1. Pad all to max (e.g., 4096) → wasteful but works
2. Fix SuperPoint to always output exactly K → change extractor
3. Use ragged tensors → ONNX doesn't support well

**Current dynamic shapes work for single pairs:**
```python
"trt_profile_min_shapes": "kpts0:1x1x2,kpts1:1x1x2,..."
"trt_profile_max_shapes": "kpts0:1x8192x2,kpts1:1x8192x2,..."
```

But batching multiple refs requires same shape per batch element.

### 16. IO Binding Eliminates CPU Roundtrips

**Lesson:** When data is already on GPU, use IO Binding to avoid the GPU→CPU→GPU copy overhead.

**The problem:**
```python
# PyTorch tensor on GPU
image_gpu = transform(img).cuda()

# Standard ONNX Runtime API forces CPU copy:
image_cpu = image_gpu.cpu().numpy()  # ~5-10ms for large tensors
outputs = session.run(None, {'image': image_cpu})  # Copies back to GPU internally
```

**Solution:**
```python
io_binding = session.io_binding()
io_binding.bind_input('image', 'cuda', 0, np.float32, 
                      tuple(image_gpu.shape), image_gpu.data_ptr())
session.run_with_iobinding(io_binding)
```

**Gotchas:**
- Tensor must be `.contiguous()` or binding fails
- Check `self.use_io_binding` flag to fall back gracefully on CPU
- Variable-size outputs (like matches) should bind to CPU

**Results:**
- SuperPoint: 0.35s → 0.28s per image
- LightGlue: 0.61s → 0.53s for 30 matches
- Output numerically identical (just faster transfer)

### 17. CUDA Graphs Don't Work with Dynamic Shapes

**Lesson:** `trt_cuda_graph_enable=True` causes OOM errors on models with dynamic input dimensions.

**The problem:**
```python
# This TRT option
"trt_cuda_graph_enable": True

# Causes CUDA to pre-allocate memory for MAX shape:
# LightGlue: kpts0:1x8192x2, desc0:1x8192x256 = ~6.4GB
# Error: "OutOfMemory (Requested size was 6443499520 bytes.)"
```

**Why it happens:**
- CUDA Graphs capture a fixed execution pattern
- TensorRT pre-allocates buffers for maximum dimensions
- With large max shapes (8192 keypoints × 256 dims), this exceeds GPU memory

**Affected models:**
- LightGlue ONNX (variable keypoint counts) ❌
- MegaLoc ONNX (internal Squeeze op has dynamic dimension) ❌
- SuperPoint ONNX (dynamic image sizes) ❌

**Solution:** Remove `trt_cuda_graph_enable` from TRT options:
```python
# onnx_models/lightglue_onnx/model.py
provider_options = [{
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    # "trt_cuda_graph_enable": True,  # REMOVED - causes OOM with dynamic shapes
    "trt_context_memory_sharing_enable": True,  # Safe alternative
    "trt_timing_cache_enable": True,
    "trt_builder_optimization_level": 5,
}]
```

**Safe TRT optimizations (no OOM risk):**
- `trt_context_memory_sharing_enable` — shares memory between subgraphs
- `trt_timing_cache_enable` — caches kernel timings for faster rebuilds
- `trt_builder_optimization_level: 5` — max optimization at build time

**When CUDA Graphs ARE safe:**
- Fixed-size inputs only (e.g., always 640×480)
- Models without internal dynamic ops
- After careful memory profiling

---

## References

- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)
- [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [LightGlue-ONNX](https://github.com/fabio-sim/LightGlue-ONNX)
- [MegaLoc](https://github.com/gmberton/MegaLoc)
- [hloc](https://github.com/cvg/Hierarchical-Localization)

---

## Appendix: Quick Reference

### Setup Commands

```bash
# Setup all models
python -m onnx_models.setup

# Setup individual models
python -m onnx_models.superpoint_onnx.setup
python -m onnx_models.lightglue_onnx.setup
python -m onnx_models.megaloc_onnx.setup
```

### Import Pattern

```python
from onnx_models import SuperPointONNX, LightGlueONNX, MegaLocONNX
```

### Verify Installation

```python
import onnxruntime as ort
print(f"Version: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")
```

### Debug TensorRT

```python
# Verbose TensorRT logging
import os
os.environ["ORT_TENSORRT_LOG_LEVEL"] = "VERBOSE"
```

---

## LightGlue Optimization Research

**Date:** 2026-08-03

Research into further optimization opportunities for the ONNX/TensorRT LightGlue matcher.

### PyTorch-Only Optimizations (Not Applicable to ONNX)

These exist in the original PyTorch LightGlue but require dynamic control flow:

| Optimization | Description | Why Not in ONNX |
|--------------|-------------|------------------|
| `depth_confidence` | Early exit from transformer layers | Requires dynamic iteration count |
| `width_confidence` | Prune low-confidence keypoints | Requires dynamic tensor shapes |
| FlashAttention | Fused attention kernel | PyTorch-specific dispatch |
| `torch.compile()` | JIT compilation | PyTorch-only |

### ONNX/TensorRT Optimizations (Potentially Applicable)

#### 1. FP8 Quantization

**Impact:** Up to 6× faster, 68% smaller engines

**How it works:**
- Uses NVIDIA Model Optimizer (`nvidia-modelopt`)
- Adds quantize/dequantize (Q/DQ) nodes to ONNX graph
- TensorRT uses FP8 kernels on supported hardware (Ada/Hopper)

**Process:**
```python
import modelopt.torch.quantization as mtq

# Calibrate with representative data
mtq.quantize(model, config=mtq.FP8_DEFAULT_CFG, forward_loop=calib_loop)

# Export to ONNX
torch.onnx.export(model, ...)
```

**Requirements:**
- NVIDIA Model Optimizer package
- Calibration dataset
- Re-export and rebuild TRT engines

#### 2. Hierarchical Chunked TopK

**Impact:** 10× speedup on detector TopK operations

**How it works:**
```
Global TopK → Chunk TopK → Concatenate → Final TopK
```

- Split score map into 65,536-element chunks
- Apply TopK within each chunk
- Final TopK on concatenated local winners

**When useful:** Most beneficial for detectors at high resolution (1280²).
SuperPoint already has efficient TopK, so limited impact for our pipeline.

#### 3. BatchNorm Folding

**Impact:** Eliminates redundant BN layers

**Formula:**
```
W' = γW / √(σ² + ε)
b' = β + γ(b - μ) / √(σ² + ε)
```

TensorRT usually does this automatically, but sometimes misses it.
Can be done manually in ONNX graph with `onnx-graphsurgeon`.

#### 4. Native SELU + Logit-Space Processing

**Impact:** Better TRT kernel fusion

- Replace fragmented SELU with native `Selu` op
- Skip softmax when only ordering matters (monotonic)

#### 5. True Batching

**Impact:** 50-80% faster for multiple pairs

**Current:** Sequential matching of 30 pairs (~18ms each)
**Batched:** Process multiple pairs in single forward pass

**Challenge:** Variable keypoint counts require padding to max.

### Current Bottleneck Analysis

```
Total: 1.22s/image
├── Feature extraction: 0.29s (22%) ← Already optimized
├── Retrieval:          0.10s (8%)  ← Minimal
├── Matching:           0.53s (46%) ← MAIN BOTTLENECK
└── Localize:           0.39s (24%) ← COLMAP/PnP
```

**Matching breakdown:**
- 30 pairs × ~18ms/pair = 0.53s
- Each pair: LightGlue inference + overhead

**Optimization priority:**
1. Reduce pairs (30 → 20) → ~33% faster matching
2. FP8 quantization → up to 50% faster matching
3. Reduce keypoints (4096 → 2048) → ~40% faster matching

### References

- [FP8 Quantization Blog](https://fabio-sim.github.io/blog/fp8-quantized-lightglue-tensorrt-nvidia-model-optimizer/)
- [GPT-5.6 Sol TRT Optimizations](https://fabio-sim.github.io/blog/gpt-5-6-sol-discovers-tensorrt-optimizations-raco-aliked-lightglue/)
- [LightGlue-ONNX Repo](https://github.com/fabio-sim/LightGlue-ONNX)

---

## In-Memory Inference Pipeline (NEW)

**Last Updated:** 2026-08-04

### Overview

We created a fully in-memory localization pipeline that eliminates ALL disk I/O during inference. The entire pipeline runs:
- Image → GPU features → GPU matching → CPU PnP → Pose

**No intermediate files written:** No HDF5, no text files, no temp directories.

### Key Module: `hloc/match_and_localize.py`

```python
from hloc.match_and_localize import localize_image

pose = localize_image(
    image_path=query_image,
    query_camera=camera,
    # Pre-loaded map data (one-time setup)
    reconstruction=colmap_model,
    db_names=db_names,
    db_descriptors=db_descriptors,  # [N, 8448] on GPU
    features_ref=map_features_h5,
    # Pre-loaded models
    feature_model=superpoint,
    retrieval_model=megaloc,
    matcher_model=lightglue,
    # Configs
    feature_conf=feature_conf,
    retrieval_conf=retrieval_conf,
    # Options
    num_matched=30,
    ref_cache=lru_cache,
)
```

### What Was Eliminated

| File | Original Use | Replacement |
|------|--------------|-------------|
| `query/feats-superpoint.h5` | Store query features | In-memory dict |
| `query/megaloc.h5` | Store global descriptor | In-memory tensor |
| `pairs.txt` | Store retrieval results | In-memory list |
| `matches.h5` | Store matches | In-memory dict |
| `results.txt` | Store poses | Direct return value |

### Performance Results

**Before (with I/O):**
```
Average: 1.35s/image
I/O operations: ~55 per query
```

**After (in-memory):**
```
Average: 0.96s/image  ← Sub-1.0s achieved!
I/O operations: ~1 per query (map features via cache)
```

**Improvement:** 29% faster (1.35s → 0.96s)

### How It Works

1. **Image Preprocessing** (`_preprocess_image()`):
   - Loads image with cv2
   - Resizes to model input size
   - Converts to normalized tensor

2. **Feature Extraction** (`extract_features_inmem()`):
   - Calls model directly: `pred = model({"image": tensor})`
   - Returns dict of tensors (stays on GPU)

3. **Retrieval** (`retrieve_topk_inmem()`):
   - Computes similarity: `sim = torch.einsum("d,nd->n", query_desc, db_desc)`
   - Returns top-k names (no file write)

4. **Matching** (`match_query_to_db_inmem()`):
   - Loads ref features via LRU cache
   - Runs matcher for each pair
   - Returns dict of matches (stays on CPU as numpy)

5. **Localization** (`_localize_from_matches()`):
   - Builds 2D-3D correspondences
   - Calls pycolmap PnP
   - Returns pose dict

### hloc Dependencies

The in-memory pipeline only needs:
- `hloc.utils.cache.LRUCache` - for caching map features
- Model loaders (`extract_features.get_model()`, `match_features.get_model()`)

It does NOT use:
- `extract_features.main()` (writes HDF5)
- `match_features.main()` (writes HDF5)
- `pairs_from_retrieval.main()` (writes text)
- `localize_sfm.main()` (reads HDF5, writes text)

### Cache Behavior

First run (cold cache):
```
Cache stats: {'size': 85, 'max_size': 200, 'hits': 5, 'misses': 85, 'hit_rate': '5.6%'}
```

Subsequent runs (warm cache):
```
Cache stats: {'size': 85, 'max_size': 200, 'hits': 90, 'misses': 0, 'hit_rate': '100%'}
```

With warm cache, **no HDF5 reads during inference**.

### Code Structure

```
hloc/match_and_localize.py
├── Image Preprocessing
│   └── _preprocess_image()
├── Feature Extraction
│   ├── extract_features_inmem()
│   └── extract_global_descriptor_inmem()
├── Retrieval
│   └── retrieve_topk_inmem()
├── Feature Loading
│   ├── _load_features_to_device()
│   └── _load_features_via_cache()
├── Matching
│   ├── _postprocess_matches()
│   └── match_query_to_db_inmem()
├── Covisibility Clustering
│   └── cluster_by_covisibility()
├── Localization
│   ├── _build_2d3d_correspondences()
│   └── _localize_from_matches()
└── Main Entry Points
    ├── localize_image()      ← NEW: fully in-memory
    ├── match_pairs()         ← Legacy API
    └── match_and_localize()  ← Legacy API
```

### Remaining I/O

**During inference:**
- Map features HDF5 read (via cache, amortized to ~0)
- `visualization.html` write (optional, end of pipeline)

**At setup (one-time):**
- Load COLMAP reconstruction
- Load global descriptors to GPU
- Load ONNX models

### Future Improvements

1. **Pre-load all map features to GPU** at startup (eliminates HDF5 entirely)
2. **Batch multiple queries** in single forward pass
3. **Reduce `num_matched`** from 30 → 20 for faster matching

---

## Session: 2026-08-09 - Performance Optimizations

### Benchmark Reference Configuration

This is the current working configuration for reference:

```
==================================================
Running inference on map: my_office
==================================================

Found 3 query images:
  img1: IMG_1766.jpg
  img2: my_desk.jpg
  img3: my_office1.jpg

[SETUP] Loading map and models (one-time)...
  Map: 196 images, 59652 points
  Building covisibility graph...
  Loading SuperPoint ONNX with TensorRT...
Loading SuperPoint ONNX from /app/onnx_models/superpoint_onnx/models/superpoint_onnx.onnx...
  Provider: CUDAExecutionProvider
  Loading MegaLoc ONNX with TensorRT...
Loading MegaLoc ONNX from /app/onnx_models/megaloc_onnx/models/megaloc_onnx.onnx
  Providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']
  Active provider: CUDAExecutionProvider
Warming up MegaLoc ONNX (3 iterations)...
  Warmup complete!
  Loading LightGlue TensorRT matcher...
Loading LightGlue ONNX from /app/onnx_models/lightglue_onnx/models/lightglue_onnx.trt.onnx...
  Provider: TensorrtExecutionProvider
  Loading database global descriptors...
  Loaded 196 database descriptors
  Setup time: 5.35s

[INFERENCE] Processing 3 queries (fully in-memory)...

  [1/3] IMG_1766.jpg
    TOTAL: 1.12s
    Position: (2.290, 0.292, -1.509) | Inliers: 934

  [2/3] my_desk.jpg
    TOTAL: 0.62s
    Position: (2.118, 0.982, -3.228) | Inliers: 27

  [3/3] my_office1.jpg
    TOTAL: 0.93s
    Position: (2.405, -0.024, -1.305) | Inliers: 87

==================================================
TIMING SUMMARY (Fully In-Memory Pipeline)
==================================================
  Setup (one-time): 5.35s
  ---------------------------------
    IMG_1766.jpg: 1.12s
    my_desk.jpg: 0.62s
    my_office1.jpg: 0.93s
  ---------------------------------
  AVERAGE: 0.89s/image
  Cache stats: {'size': 85, 'max_size': 200, 'hits': 5, 'misses': 85, 'hit_rate': '5.6%'}
  Total pipeline: 8.02s
```

### Model Providers Summary

| Model | Provider | Notes |
|-------|----------|-------|
| SuperPoint | CUDAExecutionProvider | TRT unsupported (ScatterND) |
| MegaLoc | CUDAExecutionProvider | TRT unsupported (ScatterND) |
| LightGlue | TensorrtExecutionProvider | Uses `.trt.onnx` model |

### Optimizations Implemented

#### 1. Pre-built Covisibility Graph

**Files changed:** `hloc/match_and_localize.py`, `inference/inference.py`

**What:** Pre-compute covisibility graph once at map load instead of computing during each query.

```python
# At setup (once):
covis_graph = build_covisibility_graph(colmap_model)

# At query time (fast):
clusters = cluster_by_covisibility(db_ids, reconstruction, covisibility_graph=covis_graph)
```

**Savings:** ~5-15ms per query (varies with map size)
### Key Insight: Matching Bottleneck

The 30 sequential LightGlue matches dominate the runtime (~600-700ms). Each `match_gpu()` call does an internal CUDA sync via `io_binding.copy_outputs_to_cpu()`.

**To go faster would require:**
1. Re-export LightGlue ONNX with GPU-resident outputs
2. Batch multiple match pairs in single forward pass
3. Reduce `num_matched` from 30 (accuracy tradeoff)
