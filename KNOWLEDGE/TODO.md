# TODO

## High Priority

### Code Cleanup
- [ ] Clean up current inference code
- [ ] Review code structure and organization
- [ ] Clean up the `hloc/` folder - remove unused files
- [ ] Remove legacy API functions from `match_and_localize.py`
- [ ] Extract standalone `localize/` package (zero hloc dependency for inference)

### ONNX Models Standardization
- [ ] Standardize `onnx_models/` folder structure (consistent layout across all models)
- [ ] Create `BaseONNXModel` class that all models inherit from
- [ ] Standardize code format: consistent method signatures (`extract()`, `match()`, etc.)
- [ ] Each model subclass implements: `__init__`, `preprocess`, `forward`, `postprocess`
- [ ] Unified provider configuration (TensorRT, CUDA, CPU fallback)

### Repository Restructuring

**Analysis:** Do we even need hloc anymore?

| Use Case | Need hloc? | Why |
|----------|------------|-----|
| **Inference** | **No** | `match_and_localize.py` + LRUCache (50 lines) is all we need |
| **Map Building** | **Yes** | Would need to copy ~600 lines of pycolmap integration |

**Current hloc dependencies:**
- `match_and_localize.py` only imports: `from .utils.cache import LRUCache` (~50 lines)
- `build_map.py` imports: `extract_features`, `match_features`, `reconstruction`, `pairs_from_retrieval`

**Recommendation:** Extract `localize/` as standalone package (~700 lines total)

```
project/
├── localize/              # Standalone inference package (NO hloc dependency)
│   ├── __init__.py
│   ├── match_and_localize.py  # Copy from hloc/
│   ├── cache.py               # Copy LRUCache (~50 lines)
│   └── models/                # ONNX model wrappers
│       ├── superpoint.py
│       ├── lightglue.py
│       └── megaloc.py
│
├── hloc/                  # Keep as git submodule for map building
│   └── (original hloc repo - only used by build_map.py)
│
├── scripts/
│   ├── build_map.py       # Uses hloc (one-time operation)
│   └── inference.py       # Uses localize/ (fast path)
│
└── maps/
    └── {map_name}/sfm/...
```

**Key insight:** 
- Inference needs ~10% of hloc (can be extracted)
- Map building needs ~60% of hloc (keep using it, it's a one-time operation)
- Clean separation: `localize/` for fast inference, `hloc/` for map creation

### Performance Tracking
- [x] Achieve <1.0s/image (DONE: 0.89s average)
- [x] Pre-build covisibility graph (DONE: 2026-08-09)
- [x] Analyze GPU cache transfers (DONE: already optimized - cache uses `device="cuda"`)
- [x] Analyze deferred CUDA sync (DONE: doesn't help - `match_gpu()` syncs internally)
- [ ] Document final performance breakdown
- [ ] Test warmed-up cache performance
- [ ] Re-export LightGlue with GPU-resident outputs (for async matching)
- [ ] Batch multiple match pairs in single forward pass

### Parameter Standardization
- [ ] Create single config source for parameters such as `max_num_keypoints` used by SuperPoint, LightGlue profiles, and inference scripts
  - Example: LightGlue `max_shapes=8192` but SuperPoint `max_num_keypoints=4096` → mismatch wastes TensorRT memory

### TensorRT Configuration Cleanup
- [ ] Make TensorRT limitations explicit in code (not just comments)
  - SuperPoint: Remove `use_tensorrt` option entirely, or raise error if True
  - MegaLoc: Same - remove TRT option or error on True
  - Document in model docstrings why TRT is unsupported (ScatterND with reduction)
- [ ] Update `build_trt_engines.py` to skip SuperPoint/MegaLoc (they can't use TRT)
- [ ] Add validation in `get_onnx_providers()` to warn if TRT requested for unsupported models
- [ ] Consider removing TRT-related code from SuperPoint/MegaLoc model files entirely

## Medium Priority

### Code Quality
- [ ] Add type hints throughout
- [ ] Add docstrings to all public functions
- [ ] Write unit tests for `match_and_localize.py`

### Documentation
- [ ] Update README with new architecture
- [ ] Document the in-memory pipeline
- [ ] Create setup guide for inference-only deployment

## Low Priority

### Future Improvements
- [ ] Consider ALIKED/RaCo-ALIKED models
- [ ] Explore batch inference for multiple queries
- [ ] limap integration for line features
- [ ] Mobile deployment (ONNX → CoreML/TFLite)

## Completed
- [x] ONNX model integration (SuperPoint, LightGlue, MegaLoc)
- [x] Pre-build covisibility graph at map load (2026-08-09) - ~5-15ms/query savings
- [x] Achieve 0.89s/image average (2026-08-09)
- [x] TensorRT optimization
- [x] Parallel feature extraction (SuperPoint + MegaLoc)
- [x] LRU cache for map features
- [x] In-memory pipeline (no intermediate files)
- [x] Eliminate HDF5 writes during inference
- [x] Achieve sub-1.0s latency
