"""
MegaLoc ONNX global descriptor extractor.

Runs MegaLoc using ONNX Runtime with TensorRT acceleration.
Produces 8448-dimensional global descriptors for image retrieval.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union

# Import ort from utils  
from ..utils import ort, get_onnx_providers, create_session

# Model directory for this module
MODELS_DIR = Path(__file__).parent / "models"


class MegaLocONNX:
    """
    MegaLoc global descriptor extractor using ONNX Runtime.
    
    Based on DINOv2 ViT-B/14, outputs 8448-dim L2-normalized descriptors.
    Supports TensorRT for ~8x speedup over PyTorch.
    """
    
    MODEL_NAME = "megaloc_onnx.onnx"
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        use_tensorrt: bool = True,
    ):
        """
        Args:
            model_path: Path to ONNX model. Run setup.py to export.
            device: "cuda" or "cpu"
            use_tensorrt: Enable TensorRT (FP32 mode for numerical stability)
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        
        self.device = device
        
        # Find model
        if model_path is None:
            model_path = MODELS_DIR / self.MODEL_NAME
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MegaLoc ONNX model not found at {self.model_path}\n"
                "Run: python -m onnx_models.setup"
            )
        
        # Build providers
        # Note: FP16 disabled for TensorRT due to numerical issues in DINOv2
        trt_cache = MODELS_DIR / ".trt_cache" if use_tensorrt else None
        providers, provider_options = get_onnx_providers(
            device=device,
            use_tensorrt=use_tensorrt,
            trt_cache_dir=trt_cache,
            trt_fp16=False,
        )
        
        print(f"Loading MegaLoc ONNX from {model_path}")
        print(f"  Providers: {[p if isinstance(p, str) else p[0] for p in providers]}")
        
        self.session = create_session(self.model_path, providers, provider_options, log_level=4)
        self.active_provider = self.session.get_providers()[0]
        print(f"  Active provider: {self.active_provider}")
        
        # ImageNet normalization constants
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    
    def extract(self, image: np.ndarray, resize: int = 322) -> np.ndarray:
        """
        Extract global descriptor from RGB image.
        
        Args:
            image: RGB image [H, W, 3] uint8 or [1, 3, H, W] float32
            resize: Target size (must be multiple of 14 for ViT)
        
        Returns:
            L2-normalized descriptor [1, 8448]
        """
        import cv2
        
        # Preprocess: [H, W, 3] uint8 -> [1, 3, H, W] float32
        if image.ndim == 3 and image.shape[2] == 3:
            if resize is not None:
                image = cv2.resize(image, (resize, resize), interpolation=cv2.INTER_LINEAR)
            
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, 0)
        
        # ImageNet normalize
        image = (image - self.mean) / self.std
        image = np.ascontiguousarray(image)
        
        # Run inference
        outputs = self.session.run(None, {"images": image})
        if not outputs:
            raise RuntimeError(
                f"MegaLoc ONNX session returned empty outputs. "
                f"Active provider: {self.active_provider}. "
                f"This may indicate a TensorRT fallback failure."
            )
        return outputs[0]
    
    def warmup(self, n_iters: int = 3):
        """Run warmup iterations (important for TensorRT engine building)."""
        print(f"Warming up MegaLoc ONNX ({n_iters} iterations)...")
        dummy = np.random.randn(1, 3, 322, 322).astype(np.float32)
        for _ in range(n_iters):
            self.session.run(None, {"images": dummy})
        print("  Warmup complete!")
    
    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Extract descriptor (callable interface)."""
        return self.extract(image, **kwargs)
