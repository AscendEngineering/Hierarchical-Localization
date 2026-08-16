"""
SuperPoint ONNX extractor.

Runs SuperPoint keypoint detection and description using ONNX Runtime.
Uses CUDA by default, with automatic fallback to CPU if unavailable.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..base_onnx import BaseONNXModel, HAS_TORCH
if HAS_TORCH:
    import torch

# Model directory for this module
MODELS_DIR = Path(__file__).parent / "models"


class SuperPointONNX(BaseONNXModel):
    """
    SuperPoint feature extractor using ONNX Runtime.
    
    Extracts keypoints, scores, and 256-dim descriptors from grayscale images.
    
    Note: TensorRT is not supported due to ScatterND op limitations.
    """
    
    MODEL_NAME = "superpoint_onnx.onnx"
    MODELS_DIR = MODELS_DIR
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.0005,
        device: str = "cuda",
        force_num_keypoints: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            model_path: Path to ONNX model. Auto-detects if None.
            max_num_keypoints: Maximum keypoints to return.
            detection_threshold: Minimum keypoint score.
            device: "cuda" or "cpu"
            force_num_keypoints: If True, always return exactly max_num_keypoints
                                 (top-K by score, no threshold filtering).
                                 Required for batched LightGlue inference.
            verbose: Print loading messages.
        """
        # Store config before super().__init__ (needed for potential overrides)
        self.max_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.force_num_keypoints = force_num_keypoints
        
        # Initialize base class (loads model, creates session)
        super().__init__(
            model_path=model_path,
            device=device,
            use_tensorrt=False,  # SuperPoint doesn't support TRT
            verbose=verbose,
        )
    
    def _supports_tensorrt(self) -> bool:
        """SuperPoint uses ScatterND which TensorRT doesn't support."""
        return False
    
    def preprocess(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Preprocess grayscale image for inference.
        
        Args:
            image: Grayscale float32 image, normalized to [0, 1].
                   Shape: [H, W], [1, H, W], or [1, 1, H, W]
        
        Returns:
            Dict with 'image' key containing [1, 1, H, W] float32 array
        """
        # Ensure 4D: [1, 1, H, W]
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]
        
        return {"image": image.astype(np.float32)}
    
    def postprocess(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Postprocess ONNX outputs to filtered keypoints.
        
        Args:
            outputs: [keypoints, scores, descriptors] from ONNX session
        
        Returns:
            keypoints: [N, 2] array of (x, y) coordinates
            scores: [N] array of detection scores
            descriptors: [N, 256] array of unit-normalized descriptors
        """
        # Unpack outputs
        keypoints, scores, descriptors = outputs
        
        # Remove batch dimension
        kp = keypoints[0].astype(np.float32)
        sc = scores[0].astype(np.float32)
        desc = descriptors[0].astype(np.float32)
        
        # Filter and return
        return self._filter_keypoints(kp, sc, desc)
    
    def _filter_keypoints(
        self, kp: np.ndarray, sc: np.ndarray, desc: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filter and select keypoints based on configuration.
        
        If force_num_keypoints: return exactly max_keypoints (top-K by score).
        Otherwise: threshold filter, then truncate to max_keypoints.
        """
        # If forced number of keypoints, select top-K by score
        if self.force_num_keypoints:
            indices = np.argsort(sc)[::-1][:self.max_keypoints]
            return kp[indices], sc[indices], desc[indices]
        # Else filter by detection threshold
        else:
            mask = sc > self.detection_threshold
            kp, sc, desc = kp[mask], sc[mask], desc[mask]
            
            # If necessary, truncate to max_keypoints
            if len(kp) > self.max_keypoints:
                indices = np.argsort(sc)[::-1][:self.max_keypoints]
                kp, sc, desc = kp[indices], sc[indices], desc[indices]
            
            return kp, sc, desc
    
    def extract(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features from a grayscale image.
        
        Args:
            image: Grayscale float32 image, normalized to [0, 1].
                   Shape: [H, W], [1, H, W], or [1, 1, H, W]
        
        Returns:
            keypoints: [N, 2] array of (x, y) coordinates
            scores: [N] array of detection scores
            descriptors: [N, 256] array of unit-normalized descriptors
        """
        # Use the base __call__ which does preprocess -> forward -> postprocess
        return self(image)
    
    def extract_gpu(self, image: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        Extract features from a GPU tensor using IO Binding (faster).
        
        Args:
            image: Grayscale float32 tensor on CUDA, shape [1, 1, H, W]
        
        Returns:
            keypoints: [N, 2] tensor of (x, y) coordinates on GPU
            scores: [N] tensor of detection scores on GPU
            descriptors: [N, 256] tensor of descriptors on GPU
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for extract_gpu")
        
        # If IO binding not available, fallback to CPU path
        if not self.use_io_binding:
            kp, sc, desc = self.extract(image.cpu().numpy())
            return (
                torch.from_numpy(kp).to(image.device),
                torch.from_numpy(sc).to(image.device),
                torch.from_numpy(desc).to(image.device),
            )
        
        # Ensure contiguous and correct shape
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.contiguous()
        
        # Clear and bind inputs
        self.clear_io_binding()
        self.bind_input_tensor('image', image)
        
        # Bind outputs to CPU (variable size outputs)
        self.bind_output('keypoints', device_type='cpu')
        self.bind_output('scores', device_type='cpu')
        self.bind_output('descriptors', device_type='cpu')
        
        # Run inference and postprocess
        outputs = self.run_with_io_binding()
        kp, sc, desc = self.postprocess(outputs)
        
        # Convert to torch and move to GPU
        device = image.device
        return (
            torch.from_numpy(kp).to(device),
            torch.from_numpy(sc).to(device),
            torch.from_numpy(desc).to(device),
        )
    
    def warmup(self, n_iters: int = 3):
        """Run warmup iterations with correctly sized dummy input."""
        if self.verbose:
            print(f"Warming up {self.__class__.__name__} ({n_iters} iterations)...")
        
        # Use reasonable image size for SuperPoint (640x480 grayscale)
        dummy = np.random.randn(1, 1, 480, 640).astype(np.float32)
        for _ in range(n_iters):
            self.session.run(None, {"image": dummy})
        
        if self.verbose:
            print("  Warmup complete!")
