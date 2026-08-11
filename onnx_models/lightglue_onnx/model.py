"""
LightGlue ONNX matcher.

Runs LightGlue feature matching using ONNX Runtime with TensorRT acceleration.
Matches keypoints between image pairs based on their descriptors.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

from ..base_onnx import BaseONNXModel, HAS_TORCH
if HAS_TORCH:
    import torch

# Model directory for this module
MODELS_DIR = Path(__file__).parent / "models"


class LightGlueONNX(BaseONNXModel):
    """
    LightGlue feature matcher using ONNX Runtime.
    
    Matches keypoints between two images using learned attention.
    """
    
    MODEL_NAME = "lightglue_onnx.trt.onnx"  # Prefer TRT-optimized model
    MODELS_DIR = MODELS_DIR
    
    DESCRIPTOR_DIMS = {
        "superpoint": 256,
        "disk": 128,
        "aliked": 128,
    }
    
    def __init__(
        self,
        model_path: Optional[Path] = None,
        features: str = "superpoint",
        device: str = "cuda",
        trt_cache_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Args:
            model_path: Path to ONNX model. Auto-detects if None.
            features: Feature type ("superpoint", "disk", "aliked")
            device: "cuda" or "cpu"
            trt_cache_dir: Directory for TensorRT engine cache
            verbose: Print loading messages
        """
        self.features = features
        self.desc_dim = self.DESCRIPTOR_DIMS.get(features, 256)
        self._trt_cache_dir = trt_cache_dir
        
        # Find model if not provided
        if model_path is None:
            model_path = self._find_model()
        
        # Check if TRT-compatible model is being used
        self._is_trt_model = ".trt.onnx" in str(model_path)
        
        # Initialize base class
        super().__init__(
            model_path=model_path,
            device=device,
            use_tensorrt=(self._is_trt_model and device == "cuda"),
            trt_fp16=True,
            trt_cache_dir=trt_cache_dir or (MODELS_DIR / ".trt_cache"),
            verbose=verbose,
        )
    
    def _find_model(self) -> Path:
        """Locate the best available model file."""
        # Check for TensorRT-optimized model first, then fallback to fused model
        for name in ["lightglue_onnx.trt.onnx", "lightglue_onnx_fused.onnx"]:
            candidate = MODELS_DIR / name
            if candidate.exists():
                return candidate
        
        raise FileNotFoundError(
            f"LightGlue ONNX model not found in {MODELS_DIR}.\n"
            "Download from: https://github.com/fabio-sim/LightGlue-ONNX/releases"
        )
    
    def _supports_tensorrt(self) -> bool:
        """LightGlue supports TensorRT with the .trt.onnx model."""
        return self._is_trt_model
    
    def _get_trt_profile_shapes(self) -> Optional[Dict[str, str]]:
        """Return TensorRT dynamic shape profiles for variable keypoint counts."""
        d = self.desc_dim
        return {
            "trt_profile_min_shapes": f"kpts0:1x1x2,kpts1:1x1x2,desc0:1x1x{d},desc1:1x1x{d}",
            "trt_profile_opt_shapes": f"kpts0:1x4096x2,kpts1:1x4096x2,desc0:1x4096x{d},desc1:1x4096x{d}",
            "trt_profile_max_shapes": f"kpts0:1x4096x2,kpts1:1x4096x2,desc0:1x4096x{d},desc1:1x4096x{d}",
            # Additional TRT options for LightGlue
            "trt_context_memory_sharing_enable": True,
            "trt_timing_cache_enable": True,
            "trt_builder_optimization_level": 3,  # Level 3 avoids Myelin conflicts
        }
    
    def preprocess(
        self,
        keypoints0: np.ndarray,
        keypoints1: np.ndarray,
        descriptors0: np.ndarray,
        descriptors1: np.ndarray,
        image_size0: Optional[Tuple[int, int]] = None,
        image_size1: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Preprocess keypoints and descriptors for matching.
        
        Args:
            keypoints0: [N, 2] keypoints from image 0 (x, y pixels)
            keypoints1: [M, 2] keypoints from image 1
            descriptors0: [N, D] descriptors from image 0
            descriptors1: [M, D] descriptors from image 1
            image_size0: (H, W) of image 0 for normalization
            image_size1: (H, W) of image 1 for normalization
        
        Returns:
            Dict with kpts0, kpts1, desc0, desc1 ready for ONNX
        """
        # Store original counts for postprocess
        self._n0 = len(keypoints0)
        self._n1 = len(keypoints1)
        
        # Add batch dimension and cast
        kp0 = keypoints0[np.newaxis, :, :].astype(np.float32)
        kp1 = keypoints1[np.newaxis, :, :].astype(np.float32)
        desc0 = descriptors0[np.newaxis, :, :].astype(np.float32)
        desc1 = descriptors1[np.newaxis, :, :].astype(np.float32)
        
        # Normalize keypoints to [-1, 1]
        if image_size0 is not None:
            h0, w0 = image_size0
            kp0 = 2.0 * kp0 / np.array([[[w0, h0]]], dtype=np.float32) - 1.0
        
        if image_size1 is not None:
            h1, w1 = image_size1
            kp1 = 2.0 * kp1 / np.array([[[w1, h1]]], dtype=np.float32) - 1.0
        
        return {"kpts0": kp0, "kpts1": kp1, "desc0": desc0, "desc1": desc1}
    
    def postprocess(self, outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Postprocess ONNX outputs to filtered matches.
        
        Args:
            outputs: [matches, scores] or [matches] from ONNX session
        
        Returns:
            matches: [K, 2] matched index pairs (idx0, idx1)
            scores: [K] match confidence scores
        """
        # If two outputs, unpack matches and scores
        if len(outputs) == 2:
            matches_raw, scores_raw = outputs
        # Else assume scores are all ones
        else:
            matches_raw = outputs[0]
            scores_raw = np.ones(len(matches_raw), dtype=np.float32)
        
        # Convert to numpy arrays
        matches_raw = np.asarray(matches_raw, dtype=np.int64)
        scores_raw = np.asarray(scores_raw, dtype=np.float32)
        
        # Filter valid matches
        if len(matches_raw) > 0:
            valid = (matches_raw[:, 0] < self._n0) & (matches_raw[:, 1] < self._n1)
            matches_raw = matches_raw[valid]
            scores_raw = scores_raw[valid]
        
        return matches_raw, scores_raw
    
    def match(
        self,
        keypoints0: np.ndarray,
        keypoints1: np.ndarray,
        descriptors0: np.ndarray,
        descriptors1: np.ndarray,
        image_size0: Optional[Tuple[int, int]] = None,
        image_size1: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match features between two images.
        
        Args:
            keypoints0: [N, 2] keypoints from image 0 (x, y pixels)
            keypoints1: [M, 2] keypoints from image 1
            descriptors0: [N, D] descriptors from image 0
            descriptors1: [M, D] descriptors from image 1
            image_size0: (H, W) of image 0 for normalization
            image_size1: (H, W) of image 1 for normalization
        
        Returns:
            matches: [K, 2] matched index pairs (idx0, idx1)
            scores: [K] match confidence scores
        """
        return self(keypoints0, keypoints1, descriptors0, descriptors1, image_size0, image_size1)
    
    def match_gpu(
        self,
        keypoints0: "torch.Tensor",
        keypoints1: "torch.Tensor",
        descriptors0: "torch.Tensor",
        descriptors1: "torch.Tensor",
        image_size0: Optional[Tuple[int, int]] = None,
        image_size1: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match features using IO Binding (avoids GPU->CPU->GPU copies).
        
        Args:
            keypoints0: [N, 2] GPU tensor of keypoints from image 0
            keypoints1: [M, 2] GPU tensor of keypoints from image 1
            descriptors0: [N, D] GPU tensor of descriptors from image 0
            descriptors1: [M, D] GPU tensor of descriptors from image 1
            image_size0: (H, W) of image 0 for normalization
            image_size1: (H, W) of image 1 for normalization
        
        Returns:
            matches: [K, 2] matched index pairs
            scores: [K] match confidence scores
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for match_gpu")
        
        # Store counts for postprocess
        self._n0 = keypoints0.shape[0]
        self._n1 = keypoints1.shape[0]
        
        # Add batch dimension and normalize on GPU
        kp0 = keypoints0.unsqueeze(0).float().contiguous()
        kp1 = keypoints1.unsqueeze(0).float().contiguous()
        desc0 = descriptors0.unsqueeze(0).float().contiguous()
        desc1 = descriptors1.unsqueeze(0).float().contiguous()
        
        # Normalize keypoints to [-1, 1] on GPU
        if image_size0 is not None:
            h0, w0 = image_size0
            kp0 = 2.0 * kp0 / torch.tensor([[[w0, h0]]], device=kp0.device, dtype=torch.float32) - 1.0
        
        if image_size1 is not None:
            h1, w1 = image_size1
            kp1 = 2.0 * kp1 / torch.tensor([[[w1, h1]]], device=kp1.device, dtype=torch.float32) - 1.0
        
        # Ensure contiguous after normalization
        kp0 = kp0.contiguous()
        kp1 = kp1.contiguous()
        
        # Clear and bind inputs
        self.clear_io_binding()
        self.bind_input_tensor('kpts0', kp0)
        self.bind_input_tensor('kpts1', kp1)
        self.bind_input_tensor('desc0', desc0)
        self.bind_input_tensor('desc1', desc1)
        
        # Bind outputs to CPU (variable size)
        for name in self.output_names:
            self.bind_output(name, device_type='cpu')
        
        # Run inference and postprocess
        outputs = self.run_with_io_binding()
        return self.postprocess(outputs)
