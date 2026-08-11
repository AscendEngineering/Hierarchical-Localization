"""
Base class for ONNX models.

Provides unified provider configuration, session management, and inference patterns.
All ONNX model wrappers inherit from BaseONNXModel.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np

from .utils import ort, get_onnx_providers, create_session

# Try to import torch for IO Binding support
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseONNXModel(ABC):
    """
    Abstract base class for ONNX model wrappers.
    
    Subclasses must implement:
        - MODEL_NAME: str - default model filename
        - MODELS_DIR: Path - directory containing model files
        - preprocess(*args, **kwargs) -> Dict[str, np.ndarray]
        - postprocess(outputs: List[np.ndarray]) -> Any
    
    Optional overrides:
        - _get_trt_profile_shapes() -> Dict[str, str] - TensorRT dynamic shape profiles
        - _supports_tensorrt() -> bool - whether model supports TensorRT (default: False)
    
    Example usage:
        class MyModel(BaseONNXModel):
            MODEL_NAME = "my_model.onnx"
            MODELS_DIR = Path(__file__).parent / "models"
            
            def preprocess(self, image):
                return {"input": image.astype(np.float32)}
            
            def postprocess(self, outputs):
                return outputs[0]
    """
    
    # Subclasses must define these
    MODEL_NAME: str = None
    MODELS_DIR: Path = None
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "cuda",
        use_tensorrt: bool = False,
        trt_fp16: bool = True,
        trt_cache_dir: Optional[Path] = None,
        verbose: bool = True,
    ):
        """
        Initialize ONNX model.
        
        Args:
            model_path: Path to ONNX model. Uses MODEL_NAME in MODELS_DIR if None.
            device: "cuda" or "cpu"
            use_tensorrt: Enable TensorRT acceleration
            trt_fp16: Enable FP16 in TensorRT (ignored if use_tensorrt=False)
            trt_cache_dir: Directory for TensorRT engine cache
            verbose: Print loading messages
        """
        if ort is None:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
        
        self.device = device
        self.verbose = verbose
        
        # Resolve model path
        self.model_path = self._resolve_model_path(model_path)
        
        # Validate TensorRT support
        if use_tensorrt and not self._supports_tensorrt():
            if verbose:
                print(f"Warning: {self.__class__.__name__} does not support TensorRT. Using CUDA.")
            use_tensorrt = False
        
        # Build providers
        if trt_cache_dir is None and use_tensorrt:
            trt_cache_dir = self.MODELS_DIR / ".trt_cache" if self.MODELS_DIR else Path(".trt_cache")
        
        # Get ONNX Runtime providers and options
        providers, provider_options = get_onnx_providers(
            device=device,
            use_tensorrt=use_tensorrt,
            trt_cache_dir=trt_cache_dir,
            trt_fp16=trt_fp16,
        )
        
        # Add TensorRT profile shapes if applicable
        if use_tensorrt and provider_options:
            trt_profiles = self._get_trt_profile_shapes()
            if trt_profiles and "TensorrtExecutionProvider" in providers:
                provider_options[0].update(trt_profiles)
        
        # Create session
        self.session = create_session(self.model_path, providers, provider_options)
        if verbose: print(f"Loading {self.__class__.__name__} from {self.model_path}...")
        
        # Cache provider name
        self.provider = self.session.get_providers()[0]
        if verbose: print(f"  Provider: {self.provider}")
        
        # Cache input/output names
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        # IO Binding support
        self.use_io_binding = (
            HAS_TORCH and
            device == "cuda" and
            self.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider")
        )
        self._io_binding = self.session.io_binding() if self.use_io_binding else None
    
    def _resolve_model_path(self, model_path: Optional[Union[str, Path]]) -> Path:
        """Resolve model path from argument or defaults."""
        # If model_path is provided
        if model_path is not None:
            path = Path(model_path)
            if not path.exists():
                raise FileNotFoundError(f"Model not found: {path}")
            return path
        
        # If model name or models directory is not defined, raise error
        if self.MODEL_NAME is None or self.MODELS_DIR is None:
            raise ValueError(
                f"{self.__class__.__name__} must define MODEL_NAME and MODELS_DIR, "
                "or provide model_path argument"
            )
        
        # Create full path from MODELS_DIR and MODEL_NAME
        path = self.MODELS_DIR / self.MODEL_NAME
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found: {path}\n"
                f"Run setup.py or download the model manually."
            )
        return path
    
    def _supports_tensorrt(self) -> bool:
        """
        Whether this model supports TensorRT.
        
        Override in subclass to enable TensorRT support.
        Default is False because most ONNX models have ops unsupported by TRT.
        """
        return False
    
    def _get_trt_profile_shapes(self) -> Optional[Dict[str, str]]:
        """
        Return TensorRT dynamic shape profile configuration.
        
        Override in subclass to specify min/opt/max shapes for dynamic inputs.
        
        Returns:
            Dict with keys: trt_profile_min_shapes, trt_profile_opt_shapes, trt_profile_max_shapes
            Each value is a comma-separated string like "input0:1x3x224x224,input1:1x10"
        """
        return None
    
    @abstractmethod
    def preprocess(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Preprocess inputs for ONNX inference.
        
        Returns:
            Dict mapping input names to numpy arrays
        """
        pass
    
    def forward(self, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Run ONNX inference.
        
        Args:
            inputs: Dict mapping input names to numpy arrays
        
        Returns:
            List of output numpy arrays
        """
        return self.session.run(self.output_names, inputs)
    
    @abstractmethod
    def postprocess(self, outputs: List[np.ndarray]) -> Any:
        """
        Postprocess ONNX outputs.
        
        Args:
            outputs: List of numpy arrays from session.run()
        
        Returns:
            Processed output (model-specific)
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """
        Full inference pipeline: preprocess -> forward -> postprocess.
        
        Subclasses can override for custom behavior.
        """
        # Preprocess
        inputs = self.preprocess(*args, **kwargs)
        
        # Forward pass
        outputs = self.forward(inputs)
        
        # Return postprocessed outputs
        return self.postprocess(outputs)
    
    def warmup(self, n_iters: int = 3):
        """
        Run warmup iterations.
        
        Important for TensorRT engine building and CUDA kernel warmup.
        Subclasses should override to provide appropriate dummy inputs.
        """
        if self.verbose:
            print(f"Warming up {self.__class__.__name__} ({n_iters} iterations)...")
        
        # Get input shapes from session
        dummy_inputs = {}
        for inp in self.session.get_inputs():
            shape = inp.shape
            # Replace dynamic dims with reasonable values
            shape = [s if isinstance(s, int) else 1 for s in shape]
            dtype = np.float32  # Most models use float32
            dummy_inputs[inp.name] = np.random.randn(*shape).astype(dtype)
        
        # Run warmup iterations
        for _ in range(n_iters):
            self.session.run(None, dummy_inputs)
        
        if self.verbose:
            print("  Warmup complete!")
    
    # ==================== IO Binding Helpers ====================
    
    def bind_input_tensor(self, name: str, tensor: "torch.Tensor"):
        """Bind a PyTorch tensor as input."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for IO Binding")
        if self._io_binding is None:
            raise RuntimeError("IO Binding not available (device must be 'cuda')")
        
        # Ensure tensor is contiguous for correct memory layout
        tensor = tensor.contiguous()
        
        # Bind input tensor to IO binding
        self._io_binding.bind_input(
            name=name,
            device_type=self.device,
            device_id=0,
            element_type=np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
    
    def bind_output(self, name: str, device_type: str = 'cpu'):
        """Bind an output (let ONNX Runtime allocate)."""
        if self._io_binding is None:
            raise RuntimeError("IO Binding not available (device must be 'cuda')")
        self._io_binding.bind_output(name, device_type=device_type)
    
    def bind_output_tensor(self, name: str, tensor: "torch.Tensor"):
        """Bind a pre-allocated PyTorch tensor as output (zero-copy GPU output)."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch required for IO Binding")
        if self._io_binding is None:
            raise RuntimeError("IO Binding not available (device must be 'cuda')")
        
        # Ensure tensor is contiguous
        tensor = tensor.contiguous()
        
        self._io_binding.bind_output(
            name=name,
            device_type='cuda',
            device_id=0,
            element_type=np.float32,
            shape=tuple(tensor.shape),
            buffer_ptr=tensor.data_ptr(),
        )
    
    def run_with_io_binding(self) -> List[np.ndarray]:
        """Run inference with IO Binding and return outputs as numpy arrays (copies to CPU)."""
        if self._io_binding is None:
            raise RuntimeError("IO Binding not available (device must be 'cuda')")
        self.session.run_with_iobinding(self._io_binding)
        return self._io_binding.copy_outputs_to_cpu()
    
    def run_with_io_binding_gpu(self):
        """Run inference with IO Binding. Outputs stay where they were bound (no copy)."""
        if self._io_binding is None:
            raise RuntimeError("IO Binding not available (device must be 'cuda')")
        self.session.run_with_iobinding(self._io_binding)
    
    def clear_io_binding(self):
        """Clear IO binding for reuse."""
        # Clear binding inputs and outputs
        if self._io_binding:
            self._io_binding.clear_binding_inputs()
            self._io_binding.clear_binding_outputs()
