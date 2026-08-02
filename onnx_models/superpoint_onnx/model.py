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


class SuperPointONNX:
    """
    SuperPoint feature extractor using ONNX Runtime.
    
    Extracts keypoints, scores, and 256-dim descriptors from grayscale images.
    Provides ~2x speedup over PyTorch on GPU.
    """
    
    MODEL_NAME = "superpoint_onnx.onnx"
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.0005,
        device: str = "cuda",
    ):
        """
        Args:
            model_path: Path to ONNX model. Auto-detects if None.
            max_num_keypoints: Maximum keypoints to return.
            detection_threshold: Minimum keypoint score.
            device: "cuda" or "cpu"
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        
        self.max_keypoints = max_num_keypoints
        self.detection_threshold = detection_threshold
        
        # Find model
        if model_path is None:
            model_path = MODELS_DIR / self.MODEL_NAME
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create session with CUDA provider
        providers, provider_options = get_onnx_providers(device=device, use_tensorrt=False)
        
        print(f"Loading SuperPoint ONNX from {model_path}...")
        self.session = create_session(model_path, providers, provider_options)
        self.provider = self.session.get_providers()[0]
        print(f"  Provider: {self.provider}")
    
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
    
    def __call__(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract features, returning dict format."""
        kp, sc, desc = self.extract(image)
        return {"keypoints": kp, "scores": sc, "descriptors": desc}
