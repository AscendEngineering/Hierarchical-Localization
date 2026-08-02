"""
Shared utilities for ONNX model loading and configuration.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Suppress ONNX Runtime warnings 
os.environ.setdefault("ORT_LOG_LEVEL", "ERROR")

try:
    import onnxruntime as ort
    # Suppress warnings at runtime level (3=ERROR, 4=FATAL)
    ort.set_default_logger_severity(3)
except ImportError:
    ort = None


def get_onnx_providers(
    device: str = "cuda",
    use_tensorrt: bool = False,
    trt_cache_dir: Optional[Path] = None,
    trt_fp16: bool = True,
) -> Tuple[List[str], List[Dict]]:
    """
    Build ONNX Runtime execution providers list.
    
    Args:
        device: "cuda" or "cpu"
        use_tensorrt: Enable TensorRT execution provider
        trt_cache_dir: Directory to cache TensorRT engines
        trt_fp16: Enable FP16 precision in TensorRT
    
    Returns:
        Tuple of (providers list, provider_options list)
    """
    if ort is None:
        raise ImportError("onnxruntime not installed. Run: pip install onnxruntime-gpu")
    
    available = ort.get_available_providers()
    providers = []
    provider_options = []
    
    if device == "cuda":
        if use_tensorrt and "TensorrtExecutionProvider" in available:
            if trt_cache_dir is None:
                raise ValueError("trt_cache_dir must be provided when use_tensorrt=True")
            trt_cache_dir = Path(trt_cache_dir)
            trt_cache_dir.mkdir(parents=True, exist_ok=True)
            
            providers.append("TensorrtExecutionProvider")
            provider_options.append({
                "device_id": 0,
                "trt_fp16_enable": trt_fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(trt_cache_dir),
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": str(trt_cache_dir),
                "trt_builder_optimization_level": 5,
            })
        
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
            provider_options.append({"device_id": 0})
    
    providers.append("CPUExecutionProvider")
    provider_options.append({})
    
    return providers, provider_options


def create_session(
    model_path: Path,
    providers: List,
    provider_options: List[Dict],
    log_level: int = 3,
) -> "ort.InferenceSession":
    """
    Create ONNX Runtime inference session.
    
    Args:
        model_path: Path to ONNX model
        providers: Execution providers list
        provider_options: Provider configuration options
        log_level: 0=VERBOSE, 1=INFO, 2=WARNING, 3=ERROR, 4=FATAL
    
    Returns:
        Configured InferenceSession
    """
    if ort is None:
        raise ImportError("onnxruntime not installed")
    
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = log_level
    
    return ort.InferenceSession(
        str(model_path),
        sess_options=sess_options,
        providers=providers,
        provider_options=provider_options,
    )
