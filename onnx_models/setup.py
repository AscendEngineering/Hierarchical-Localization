#!/usr/bin/env python3
"""
Setup all ONNX models.

Downloads SuperPoint and LightGlue from GitHub releases,
exports MegaLoc from PyTorch.
"""

from pathlib import Path


def check_dependencies():
    """Check required packages are installed."""
    print("Checking dependencies...")
    
    try:
        import onnxruntime as ort
        print(f"  onnxruntime: {ort.__version__}")
        
        providers = ort.get_available_providers()
        print(f"  Providers: {providers}")
        
        if "TensorrtExecutionProvider" in providers:
            print("  TensorRT: available")
        if "CUDAExecutionProvider" in providers:
            print("  CUDA: available")
            
    except ImportError:
        print("  onnxruntime: NOT INSTALLED")
        print("  Install with: pip install onnxruntime-gpu")
        return False
    
    try:
        import numpy
        print(f"  numpy: {numpy.__version__}")
    except ImportError:
        print("  numpy: NOT INSTALLED")
        return False
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup ONNX models")
    parser.add_argument("--force", action="store_true", help="Re-download/export existing models")
    parser.add_argument("--skip-megaloc", action="store_true", help="Skip MegaLoc export (if PyTorch/CUDA unavailable)")
    args = parser.parse_args()
    
    print("=" * 50)
    print("ONNX Models Setup")
    print("=" * 50)
    
    if not check_dependencies():
        return 1
    
    # Import and run each model's setup
    from .superpoint_onnx.setup import setup as setup_superpoint
    from .lightglue_onnx.setup import setup as setup_lightglue
    
    setup_superpoint(args.force)
    setup_lightglue(args.force)
    
    if not args.skip_megaloc:
        from .megaloc_onnx.setup import setup as setup_megaloc
        setup_megaloc(args.force)
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    exit(main())
