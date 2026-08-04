"""
SuperPoint ONNX extractor.

Runs SuperPoint keypoint detection and description using ONNX Runtime.
Uses CUDA by default, with automatic fallback to CPU if unavailable.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np

# Import ort from utils  
from ..utils import ort, get_onnx_providers, create_session

# Model directory for this module
MODELS_DIR = Path(__file__).parent / "models"

# Try to import torch for IO Binding support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class SuperPointONNX:
    """
    SuperPoint feature extractor using ONNX Runtime.
    
    Extracts keypoints, scores, and 256-dim descriptors from grayscale images.
    Provides ~2x speedup over PyTorch on GPU, ~3x with TensorRT.
    """
    
    MODEL_NAME = "superpoint_onnx.onnx"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.0005,
        device: str = "cuda",
        use_tensorrt: bool = False,
    ):
        """
        Args:
            model_path: Path to ONNX model. Auto-detects if None.
            max_num_keypoints: Maximum keypoints to return.
            detection_threshold: Minimum keypoint score.
            device: "cuda" or "cpu"
            use_tensorrt: Enable TensorRT acceleration (FP16)
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        
        self.max_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        self.device = device
        
        # Find model
        if model_path is None:
            model_path = MODELS_DIR / self.MODEL_NAME
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create session with TRT or CUDA provider
        trt_cache = MODELS_DIR / ".trt_cache" if use_tensorrt else None
        providers, provider_options = get_onnx_providers(
            device=device,
            use_tensorrt=use_tensorrt,
            trt_cache_dir=trt_cache,
            trt_fp16=True,  # SuperPoint works fine with FP16
        )
        
        # Add TRT profile shapes for dynamic input (image size varies)
        if use_tensorrt and provider_options and "TensorrtExecutionProvider" in providers:
            # Image input: [1, 1, H, W] - grayscale images up to 1024x1024
            provider_options[0].update({
                "trt_profile_min_shapes": "image:1x1x256x256",
                "trt_profile_opt_shapes": "image:1x1x768x1024",
                "trt_profile_max_shapes": "image:1x1x1024x1024",
            })
        
        print(f"Loading SuperPoint ONNX from {model_path}...")
        self.session = create_session(model_path, providers, provider_options)
        self.provider = self.session.get_providers()[0]
        print(f"  Provider: {self.provider}")
        
        # If IO Binding is available (CUDA only)
        self.use_io_binding = (
            HAS_TORCH and 
            device == "cuda" and 
            "CUDA" in self.provider
        )
    
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
        # Ensure 4D: [1, 1, H, W]
        if image.ndim == 2:
            image = image[np.newaxis, np.newaxis, :, :]
        elif image.ndim == 3:
            image = image[np.newaxis, :, :, :]
        
        image = image.astype(np.float32)
        
        # Run inference
        outputs = self.session.run(None, {"image": image})
        keypoints, scores, descriptors = outputs
        
        # Remove batch dimension
        kp = keypoints[0].astype(np.float32)
        sc = scores[0].astype(np.float32)
        desc = descriptors[0].astype(np.float32)
        
        # Filter by threshold
        mask = sc > self.detection_threshold
        kp = kp[mask]
        sc = sc[mask]
        desc = desc[mask]
        
        # Keep top-K by score
        if len(kp) > self.max_keypoints:
            indices = np.argsort(sc)[::-1][:self.max_keypoints]
            kp = kp[indices]
            sc = sc[indices]
            desc = desc[indices]
        
        return kp, sc, desc
    
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
        # If IO Binding is not available
        if not self.use_io_binding:
            # Fallback to CPU path
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
        
        # Create IO binding
        io_binding = self.session.io_binding()
        
        # Bind input directly from GPU tensor (no copy)
        io_binding.bind_input(
            name='image',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(image.shape),
            buffer_ptr=image.data_ptr()
        )
        
        # Bind outputs, let ONNX RT allocate them
        io_binding.bind_output('keypoints', device_type='cpu')
        io_binding.bind_output('scores', device_type='cpu')
        io_binding.bind_output('descriptors', device_type='cpu')
        
        # Run inference
        self.session.run_with_iobinding(io_binding)
        
        # Get outputs (on CPU as numpy)
        outputs = io_binding.copy_outputs_to_cpu()
        keypoints, scores, descriptors = outputs
        
        # Remove batch dimension
        kp = keypoints[0].astype(np.float32)
        sc = scores[0].astype(np.float32)
        desc = descriptors[0].astype(np.float32)
        
        # Filter by threshold
        mask = sc > self.detection_threshold
        kp = kp[mask]
        sc = sc[mask]
        desc = desc[mask]
        
        # Keep top-K by score
        if len(kp) > self.max_keypoints:
            indices = np.argsort(sc)[::-1][:self.max_keypoints]
            kp = kp[indices]
            sc = sc[indices]
            desc = desc[indices]
        
        # Convert to torch and move to GPU
        device = image.device
        return (
            torch.from_numpy(kp).to(device),
            torch.from_numpy(sc).to(device),
            torch.from_numpy(desc).to(device),
        )
    
    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features, returning dict format."""
        kp, sc, desc = self.extract(image)
        return {"keypoints": kp, "scores": sc, "descriptors": desc}
