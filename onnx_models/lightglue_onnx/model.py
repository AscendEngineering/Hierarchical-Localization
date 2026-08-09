"""
LightGlue ONNX matcher.

Runs LightGlue feature matching using ONNX Runtime with TensorRT acceleration.
Matches keypoints between image pairs based on their descriptors.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Import ort from utils  
from ..utils import ort, get_onnx_providers, create_session

# Try to import torch for IO Binding support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Model directory for this module
MODELS_DIR = Path(__file__).parent / "models"


class LightGlueONNX:
    """
    LightGlue feature matcher using ONNX Runtime.
    
    Matches keypoints between two images using learned attention.
    Supports TensorRT for ~3x speedup over PyTorch.
    """
    
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
    ):
        """
        Args:
            model_path: Path to ONNX model. Auto-detects if None.
            features: Feature type ("superpoint", "disk", "aliked")
            device: "cuda" or "cpu"
            trt_cache_dir: Directory for TensorRT engine cache
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        
        self.features = features
        self.desc_dim = self.DESCRIPTOR_DIMS.get(features, 256)
        
        # Find model
        if model_path is None:
            model_path = self._find_model(features)
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Check if TRT-compatible model
        is_trt_model = ".trt.onnx" in str(model_path)
        
        # Build providers
        if is_trt_model and device == "cuda":
            providers, provider_options = self._build_trt_providers(trt_cache_dir)
        else:
            providers, provider_options = get_onnx_providers(device=device, use_tensorrt=False)
        
        print(f"Loading LightGlue ONNX from {model_path}...")
        self.session = create_session(model_path, providers, provider_options)
        self.provider = self.session.get_providers()[0]
        print(f"  Provider: {self.provider}")
        
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # Enable IO Binding for GPU inputs if CUDA provider is active
        self.use_io_binding = (
            HAS_TORCH and 
            self.provider in ('CUDAExecutionProvider', 'TensorrtExecutionProvider')
        )
        
        # Pre-allocate IO Binding for reuse (avoids allocation overhead per call)
        self._io_binding = self.session.io_binding() if self.use_io_binding else None
    
    def _find_model(self, features: str) -> Path:
        """Locate the best available model file."""
        # Check for TRT-optimized model first
        for name in ["lightglue_onnx.trt.onnx", "lightglue_onnx_fused.onnx"]:
            candidate = MODELS_DIR / name
            if candidate.exists():
                return candidate
        
        raise FileNotFoundError(
            f"LightGlue ONNX model not found.\n"
            "Download from: https://github.com/fabio-sim/LightGlue-ONNX/releases"
        )
    
    def _build_trt_providers(self, cache_dir: Optional[Path]) -> Tuple[List[str], List[Dict]]:
        """Build TensorRT provider config with dynamic shape profiles."""
        if cache_dir is None:
            cache_dir = MODELS_DIR / ".trt_cache"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic shapes for variable keypoint counts
        d = self.desc_dim
        min_shapes = f"kpts0:1x1x2,kpts1:1x1x2,desc0:1x1x{d},desc1:1x1x{d}"
        opt_shapes = f"kpts0:1x4096x2,kpts1:1x4096x2,desc0:1x4096x{d},desc1:1x4096x{d}"
        max_shapes = f"kpts0:1x4096x2,kpts1:1x4096x2,desc0:1x4096x{d},desc1:1x4096x{d}"
        
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        provider_options = [
            {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache_dir),
                "trt_profile_min_shapes": min_shapes,
                "trt_profile_max_shapes": max_shapes,
                "trt_profile_opt_shapes": opt_shapes,
                # NOTE: trt_cuda_graph_enable=True causes OOM with dynamic shapes
                "trt_context_memory_sharing_enable": True,  # Share memory between subgraphs
                "trt_timing_cache_enable": True,            # Cache kernel timings for faster builds
                "trt_timing_cache_path": str(cache_dir),
                # Level 3 avoids Myelin stream capture conflicts
                "trt_builder_optimization_level": 3,
            },
            {"device_id": 0},
            {},
        ]
        
        return providers, provider_options
    
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
        N0, N1 = len(keypoints0), len(keypoints1)
        
        # Add batch dimension
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
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {"kpts0": kp0, "kpts1": kp1, "desc0": desc0, "desc1": desc1}
        )
        
        # Parse outputs
        if len(outputs) == 2:
            matches_raw, scores_raw = outputs
        else:
            matches_raw = outputs[0]
            scores_raw = np.ones(len(matches_raw), dtype=np.float32)
        
        matches_raw = np.asarray(matches_raw, dtype=np.int64)
        scores_raw = np.asarray(scores_raw, dtype=np.float32)
        
        # Filter valid matches
        if len(matches_raw) > 0:
            valid = (matches_raw[:, 0] < N0) & (matches_raw[:, 1] < N1)
            matches_raw = matches_raw[valid]
            scores_raw = scores_raw[valid]
        
        return matches_raw, scores_raw

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
        N0, N1 = keypoints0.shape[0], keypoints1.shape[0]
        
        # Clone and add batch dimension [1, N, D] - normalize on GPU
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
        
        # Reuse pre-allocated IO Binding (clear previous bindings)
        self._io_binding.clear_binding_inputs()
        self._io_binding.clear_binding_outputs()
        
        # Bind inputs from GPU tensors
        self._io_binding.bind_input(
            name='kpts0',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(kp0.shape),
            buffer_ptr=kp0.data_ptr(),
        )
        self._io_binding.bind_input(
            name='kpts1',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(kp1.shape),
            buffer_ptr=kp1.data_ptr(),
        )
        self._io_binding.bind_input(
            name='desc0',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(desc0.shape),
            buffer_ptr=desc0.data_ptr(),
        )
        self._io_binding.bind_input(
            name='desc1',
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(desc1.shape),
            buffer_ptr=desc1.data_ptr(),
        )
        
        # Let ONNX allocate outputs (variable size)
        for name in self.output_names:
            self._io_binding.bind_output(name, device_type='cpu')
        
        # Run inference
        self.session.run_with_iobinding(self._io_binding)
        
        # Get outputs
        outputs = self._io_binding.copy_outputs_to_cpu()
        
        # Parse outputs
        if len(outputs) == 2:
            matches_raw, scores_raw = outputs
        else:
            matches_raw = outputs[0]
            scores_raw = np.ones(len(matches_raw), dtype=np.float32)
        
        matches_raw = np.asarray(matches_raw, dtype=np.int64)
        scores_raw = np.asarray(scores_raw, dtype=np.float32)
        
        # Filter valid matches
        if len(matches_raw) > 0:
            valid = (matches_raw[:, 0] < N0) & (matches_raw[:, 1] < N1)
            matches_raw = matches_raw[valid]
            scores_raw = scores_raw[valid]
        
        return matches_raw, scores_raw
    
    def __call__(
        self,
        keypoints0: np.ndarray,
        keypoints1: np.ndarray,
        descriptors0: np.ndarray,
        descriptors1: np.ndarray,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Match features, returning dict format."""
        matches, scores = self.match(keypoints0, keypoints1, descriptors0, descriptors1, **kwargs)
        return {"matches": matches, "scores": scores}
