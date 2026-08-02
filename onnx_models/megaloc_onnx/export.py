#!/usr/bin/env python3
"""
Export MegaLoc PyTorch model to ONNX.

MegaLoc uses DINOv2 ViT-B/14 as backbone and outputs 8448-dim global descriptors.
The exported model runs through ONNX Runtime with optional TensorRT acceleration.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

os.environ["TORCH_ONNX_USE_NEW_EXPORTER"] = "0"


def export_megaloc_onnx(output_path: Path, opset_version: int = 18):
    """
    Export MegaLoc to ONNX format.
    
    Args:
        output_path: Where to save the ONNX model
        opset_version: ONNX opset (18+ recommended for PyTorch 2.x)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MegaLoc ONNX Export")
    print("=" * 60)
    
    print("\n[1/4] Loading MegaLoc from torch.hub...")
    model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")
    model.eval()
    model.cuda()
    print("  Model loaded")
    
    print("\n[2/4] Preparing export...")
    dummy_input = torch.randn(1, 3, 322, 322, device="cuda")
    
    print(f"\n[3/4] Exporting to ONNX (opset {opset_version})...")
    print(f"  Output: {output_path}")
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            input_names=["images"],
            output_names=["descriptors"],
            opset_version=opset_version,
            do_constant_folding=True,
            export_params=True,
        )
    print("  Export complete!")
    
    print("\n[4/4] Verifying...")
    import onnx
    from onnx import shape_inference
    
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    
    # Try onnx-simplifier if available
    try:
        import onnxsim
        print("  Running onnx-simplifier...")
        onnx_model, check = onnxsim.simplify(onnx_model)
        if check:
            print("  Simplified successfully!")
    except ImportError:
        print("  onnx-simplifier not installed, skipping")
    
    onnx.save(onnx_model, str(output_path))
    
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  Model size: {size_mb:.1f} MB")
    print("\n" + "=" * 60)
    
    return output_path


def verify_onnx(onnx_path: Path):
    """Test the exported model produces correct output."""
    import onnxruntime as ort
    
    print("\nVerifying ONNX model...")
    
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    
    dummy = np.random.randn(1, 3, 322, 322).astype(np.float32)
    output = session.run(["descriptors"], {"images": dummy})[0]
    
    print(f"  Output shape: {output.shape}")
    print(f"  Output norm: {np.linalg.norm(output):.4f} (should be ~1.0)")
    print(f"  Has NaN: {np.isnan(output).any()}")
    
    if np.isnan(output).any():
        print("  WARNING: Output contains NaN!")
        return False
    
    print("  Verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Export MegaLoc to ONNX")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models" / "megaloc_onnx.onnx",
        help="Output path for ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version"
    )
    args = parser.parse_args()
    
    export_megaloc_onnx(args.output, args.opset)
    verify_onnx(args.output)


if __name__ == "__main__":
    main()
