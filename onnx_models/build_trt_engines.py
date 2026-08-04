#!/usr/bin/env python3
"""
Pre-build TensorRT engines for all ONNX models.

Run this once with an empty GPU to cache TRT engines.
After caching, inference uses much less memory.

Usage:
    python3 onnx_models/build_trt_engines.py
"""

import sys
import time

# Ensure onnx_models package is importable when run as script
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_gpu_memory():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        free, total = map(int, result.stdout.strip().split(", "))
        print(f"  GPU Memory: {free} MB free / {total} MB total")
        
        # LightGlue TRT compilation needs ~6GB contiguous memory
        if free < 6000:
            print(f"  WARNING: LightGlue TRT build needs ~6GB. You have {free}MB free.")
            print(f"           Kill other GPU processes first for best results.")
        return free
    except Exception as e:
        print(f"  Could not check GPU memory: {e}")
        return None


def build_lightglue():
    # LightGlue: attention-based matcher, ~6GB during build, FP16
    print("\n[1/3] Building LightGlue TRT engine...")
    start = time.time()
    
    try:
        from onnx_models.lightglue_onnx.model import LightGlueONNX
        import numpy as np
        
        model = LightGlueONNX()
        # Run inference to trigger full TRT build (constructor only loads ONNX)
        n_kpts = 100
        dummy_kp = np.random.randn(1, n_kpts, 2).astype(np.float32)
        dummy_desc = np.random.randn(1, n_kpts, 256).astype(np.float32)
        model.match(dummy_kp[0], dummy_kp[0], dummy_desc[0], dummy_desc[0])
        print(f"  Done! ({time.time() - start:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def build_megaloc():
    # MegaLoc: DINOv2-based global descriptor, FP32 (numerical stability)
    print("\n[2/3] Building MegaLoc TRT engine...")
    start = time.time()
    
    try:
        from onnx_models.megaloc_onnx.model import MegaLocONNX
        model = MegaLocONNX(use_tensorrt=True)
        # Warmup runs inference which triggers lazy TRT compilation
        model.warmup(n_iters=1)
        print(f"  Done! ({time.time() - start:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def build_superpoint():
    # SuperPoint: CNN keypoint detector, FP16
    print("\n[3/3] Building SuperPoint TRT engine...")
    start = time.time()
    
    try:
        from onnx_models.superpoint_onnx.model import SuperPointONNX
        import numpy as np
        
        model = SuperPointONNX(use_tensorrt=True)
        # Run inference to trigger TRT compilation
        dummy = np.random.randn(1, 1, 768, 1024).astype(np.float32)
        model.extract(dummy)
        print(f"  Done! ({time.time() - start:.1f}s)")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    print("=" * 60)
    print("TensorRT Engine Builder")
    print("=" * 60)
    print("\nPre-builds TRT engines so they're cached for inference.")
    print("Run with an EMPTY GPU for best results (~6GB needed).\n")
    
    check_gpu_memory()
    
    # Build largest model first while GPU memory is empty
    results = {
        'lightglue': build_lightglue(),
        'megaloc': build_megaloc(),
        'superpoint': build_superpoint(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {name}: {status}")
    
    if all(results.values()):
        print("\nAll engines cached. Ready for inference.")
        return 0
    else:
        print("\nSome builds failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
