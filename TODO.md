# Optimization TODO

**Current:** 1.20s/image | **Target:** <1.0s/image

## Pending Optimizations

### 1. CUDA Streams for Parallel Matching
- **Effort:** Low
- **Expected Gain:** ~20% on matching (~0.10s)
- Run multiple LightGlue pairs on different CUDA streams
- No model changes needed

### 2. Pad Keypoints + CUDA Graphs
- **Effort:** Medium
- **Expected Gain:** ~15% on matching (~0.08s)
- Pad all keypoints to fixed count (e.g., 2048)
- Enables `trt_cuda_graph_enable=True`
- Requires mask to track valid keypoints

### 3. Batched LightGlue Export
- **Effort:** High
- **Expected Gain:** ~50% on matching (~0.25s)
- Re-export ONNX with batch dimension for refs
- Process 30 pairs in single forward pass
- Requires LightGlue architecture changes

---

**Combined potential:** 1.20s → ~0.90s
