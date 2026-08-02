#!/usr/bin/env python3
"""
Export MegaLoc ONNX model from PyTorch.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
MODEL_NAME = "megaloc_onnx.onnx"


def export(force: bool = False) -> bool:
    """Export MegaLoc from PyTorch to ONNX."""
    output_path = MODELS_DIR / MODEL_NAME
    
    if output_path.exists() and not force:
        print(f"  {MODEL_NAME}: exists (skip)")
        return True
    
    print("  Checking PyTorch...")
    try:
        import torch
        print(f"    PyTorch: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("  ERROR: CUDA not available")
            return False
            
    except ImportError:
        print("  ERROR: PyTorch not installed")
        print("  Install with: pip install torch")
        return False
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("  Loading MegaLoc from torch.hub...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model.eval()
    model.cuda()
    
    dummy_input = torch.randn(1, 3, 322, 322, device="cuda")
    
    print(f"  Exporting to {output_path}...")
    os.environ["TORCH_ONNX_USE_NEW_EXPORTER"] = "0"
    
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["images"],
        output_names=["descriptors"],
        dynamic_axes={
            "images": {0: "batch", 2: "height", 3: "width"},
            "descriptors": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    
    # Verify export
    import onnx
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    
    print("  Export complete!")
    return True


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
    """Export and verify MegaLoc model."""
    print("\n[MegaLoc ONNX]")
    success = export(force)
    if success:
        verify()
    return success


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup MegaLoc ONNX model")
    parser.add_argument("--force", action="store_true", help="Re-export existing model")
    args = parser.parse_args()
    
    exit(0 if setup(args.force) else 1)
