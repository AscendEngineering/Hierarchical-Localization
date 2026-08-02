#!/usr/bin/env python3
"""
Download LightGlue ONNX model.
"""

import urllib.request
import urllib.error
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

MODEL_URL = "https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v1.0.0/superpoint_lightglue.trt.onnx"
MODEL_NAME = "lightglue_onnx.trt.onnx"


def download(force: bool = False) -> bool:
    """Download LightGlue ONNX model."""
    output_path = MODELS_DIR / MODEL_NAME
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists() and not force:
        print(f"  {MODEL_NAME}: exists (skip)")
        return True
    
    print(f"  {MODEL_NAME}: downloading...")
    
    try:
        def progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, block_num * block_size * 100 // total_size)
                mb = block_num * block_size / (1024 * 1024)
                print(f"\r    {pct}% ({mb:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(MODEL_URL, str(output_path), progress)
        print(" done")
        return True
        
    except urllib.error.HTTPError as e:
        print(f" FAILED (HTTP {e.code})")
        return False
    except Exception as e:
        print(f" FAILED ({e})")
        return False


def verify() -> bool:
    """Verify model loads correctly."""
    path = MODELS_DIR / MODEL_NAME
    
    if not path.exists():
        print(f"  {MODEL_NAME}: not found")
        return False
    
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(
            str(path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        print(f"  {MODEL_NAME}: OK ({session.get_providers()[0]})")
        return True
    except Exception as e:
        print(f"  {MODEL_NAME}: FAILED ({e})")
        return False


def setup(force: bool = False) -> bool:
    """Download and verify LightGlue model."""
    print("\n[LightGlue ONNX]")
    success = download(force)
    if success:
        verify()
    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup LightGlue ONNX model")
    parser.add_argument("--force", action="store_true", help="Re-download existing model")
    args = parser.parse_args()
    
    exit(0 if setup(args.force) else 1)
