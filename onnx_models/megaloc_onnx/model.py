"""
MegaLoc ONNX global descriptor extractor.

Runs MegaLoc using ONNX Runtime with CUDA acceleration.
Produces 8448-dimensional global descriptors for image retrieval.

Note: TensorRT is not used due to numerical instability in DINOv2 backbone.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import cv2

from ..base_onnx import BaseONNXModel, HAS_TORCH
if HAS_TORCH:
    import torch

# Model directory for this module
MODELS_DIR = Path(__file__).parent / "models"


class MegaLocONNX(BaseONNXModel):
    """
    MegaLoc global descriptor extractor using ONNX Runtime.
    
    Based on DINOv2 ViT-B/14, outputs 8448-dim L2-normalized descriptors.
    
    Note: TensorRT disabled due to numerical issues in DINOv2 backbone.
    """
    
    MODEL_NAME = "megaloc_onnx.onnx"
    MODELS_DIR = MODELS_DIR
    
    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        resize: int = 322,
        verbose: bool = True,
    ):
        """
        Args:
            model_path: Path to ONNX model. Run setup.py to export.
            device: "cuda" or "cpu"
            resize: Target image size (must be multiple of 14 for ViT)
            verbose: Print loading messages
        """
        self.resize = resize
        
        # Initialize base class (loads model, creates session)
        super().__init__(
            model_path=model_path,
            device=device,
            use_tensorrt=False,  # TensorRT disabled for numerical stability
            verbose=verbose,
        )
    
    def _supports_tensorrt(self) -> bool:
        """MegaLoc/DINOv2 has numerical issues with TensorRT FP16."""
        return False
    
    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess RGB image for inference.
        
        Args:
            image: RGB image [H, W, 3] uint8 or [1, 3, H, W] float32
        
        Returns:
            Dict with 'images' key containing [1, 3, H, W] normalized float32
        """  
        # Handle [H, W, 3] uint8 input
        if image.ndim == 3 and image.shape[2] == 3:
            if self.resize is not None:
                image = cv2.resize(image, (self.resize, self.resize), interpolation=cv2.INTER_LINEAR)
            
            image = image.astype(np.float32) / 255.0
            image = image.transpose(2, 0, 1)  # [H, W, 3] -> [3, H, W]
            image = np.expand_dims(image, 0)  # [1, 3, H, W]
        
        # ImageNet normalize
        image = (image - self.MEAN) / self.STD

        # Ensure contiguous memory layout
        image = np.ascontiguousarray(image)
        
        return {"images": image}
    
    def postprocess(self, outputs: List[np.ndarray]) -> np.ndarray:
        """
        Return the L2-normalized global descriptor.
        
        Args:
            outputs: [descriptor] from ONNX session
        
        Returns:
            L2-normalized descriptor [1, 8448]
        """
        if not outputs:
            raise RuntimeError(
                f"MegaLoc ONNX session returned empty outputs. "
                f"Active provider: {self.provider}. "
                f"This may indicate a provider fallback failure."
            )
        # Return the first output (descriptor)
        return outputs[0]
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract global descriptor from RGB image.
        
        Args:
            image: RGB image [H, W, 3] uint8 or [1, 3, H, W] float32
        
        Returns:
            L2-normalized descriptor [8448]
        """
        # Use the base __call__ which does preprocess -> forward -> postprocess
        result = self(image)
        return result.squeeze(0)  # Remove batch dim: [1, 8448] -> [8448]
    
    def extract_gpu(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Extract global descriptor from GPU tensor using IO Binding (zero-copy).
        
        Args:
            image: RGB float32 tensor on CUDA, shape [1, 3, H, W] or [3, H, W]
                   Expected to be normalized to [0, 1] range
        
        Returns:
            L2-normalized descriptor tensor [8448] on GPU
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for extract_gpu")
        
        # Fallback if IO binding not available
        if not self.use_io_binding:
            img_np = image.cpu().numpy()
            if img_np.ndim == 3:
                img_np = img_np[np.newaxis, ...]
            desc = self.extract((img_np[0].transpose(1, 2, 0) * 255).astype(np.uint8))
            return torch.from_numpy(desc).to(image.device)
        
        # Ensure 4D and contiguous
        if image.ndim == 3:
            image = image.unsqueeze(0)
        
        # Apply ImageNet normalization on GPU
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        image = (image - mean) / std
        image = image.contiguous()
        
        # Pre-allocate output tensor on GPU (zero-copy)
        output = torch.empty(1, 8448, dtype=torch.float32, device=image.device)
        
        # IO binding with GPU output
        self.clear_io_binding()
        self.bind_input_tensor('images', image)
        self.bind_output_tensor('descriptors', output)
        
        # Run inference (output written directly to pre-allocated tensor)
        self.run_with_io_binding_gpu()
        
        return output.squeeze(0)  # [8448]
    
    def warmup(self, n_iters: int = 3):
        """Run warmup iterations with correctly sized dummy input."""
        if self.verbose:
            print(f"Warming up {self.__class__.__name__} ({n_iters} iterations)...")
        
        # Use correct input size for MegaLoc
        dummy = np.random.randn(1, 3, self.resize, self.resize).astype(np.float32)
        for _ in range(n_iters):
            self.session.run(None, {"images": dummy})
        
        if self.verbose:
            print("  Warmup complete!")
