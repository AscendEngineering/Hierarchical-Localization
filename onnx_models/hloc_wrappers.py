"""
hloc wrappers for ONNX models.

Adapts SuperPointONNX, LightGlueONNX, and MegaLocONNX to work with
hloc's BaseModel interface for extract_features and match_features.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import torch

from .superpoint_onnx import SuperPointONNX
from .lightglue_onnx import LightGlueONNX
from .megaloc_onnx import MegaLocONNX


class SuperPointONNXWrapper:
    """Wraps SuperPointONNX for hloc's extract_features pipeline."""
    
    default_conf = {
        "model_path": None,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "nms_radius": 4,
        "device": "cuda",
        "use_tensorrt": False,
        "force_num_keypoints": False,
    }
    required_inputs = ["image"]
    
    def __init__(self, conf: Dict):
        self.conf = {**self.default_conf, **conf}
        self._model = SuperPointONNX(
            model_path=self.conf.get("model_path"),
            max_num_keypoints=self.conf.get("max_num_keypoints", 2048),
            detection_threshold=self.conf.get("detection_threshold", 0.0005),
            device=self.conf.get("device", "cuda"),
            use_tensorrt=self.conf.get("use_tensorrt", False),
            force_num_keypoints=self.conf.get("force_num_keypoints", False),
        )
        # Warmup to trigger CUDA kernel compilation
        self._warmup()
    
    def _warmup(self):
        """Run dummy inference to compile CUDA kernels."""
        # If model has extract_gpu method and IO Binding, use it for warmup
        if hasattr(self._model, 'extract_gpu') and self._model.use_io_binding:
            dummy = torch.rand(1, 1, 768, 1024, device='cuda')
            _ = self._model.extract_gpu(dummy)
        # Else, fallback to CPU path for warmup
        else:
            dummy = np.random.rand(1, 768, 1024).astype(np.float32)
            _ = self._model.extract(dummy)
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def __call__(self, data: Dict) -> Dict:
        image = data["image"]
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # Convert to grayscale
        if C == 3:
            image = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        device = data["image"].device
        
        # Use GPU path with IO Binding if available (faster)
        if hasattr(self._model, 'extract_gpu') and device.type == 'cuda':
            all_keypoints = []
            all_scores = []
            all_descriptors = []
            
            # For each image in the batch, extract features on GPU
            for b in range(B):
                kp, sc, desc = self._model.extract_gpu(image[b])
                all_keypoints.append(kp)
                all_scores.append(sc)
                all_descriptors.append(desc)
            
            # Pad to max length
            max_n = max(len(kp) for kp in all_keypoints) if all_keypoints else 0
            
            # Preallocate tensors for keypoints, scores, and descriptors
            keypoints = torch.zeros((B, max_n, 2), dtype=torch.float32, device=device)
            scores = torch.zeros((B, max_n), dtype=torch.float32, device=device)
            descriptors = torch.zeros((B, max_n, 256), dtype=torch.float32, device=device)
            
            # Fill in the results for each image in the batch
            for b in range(B):
                n = len(all_keypoints[b])
                keypoints[b, :n] = all_keypoints[b]
                scores[b, :n] = all_scores[b]
                descriptors[b, :n] = all_descriptors[b]
            
            return {
                "keypoints": keypoints,
                "scores": scores,
                "descriptors": descriptors,
            }
        
        # Fallback to CPU path
        all_keypoints = []
        all_scores = []
        all_descriptors = []
        
        for b in range(B):
            img_np = image[b].cpu().numpy().astype(np.float32)
            kp, sc, desc = self._model.extract(img_np)
            all_keypoints.append(kp)
            all_scores.append(sc)
            all_descriptors.append(desc)
        
        max_n = max(len(kp) for kp in all_keypoints) if all_keypoints else 0
        
        keypoints = np.zeros((B, max_n, 2), dtype=np.float32)
        scores = np.zeros((B, max_n), dtype=np.float32)
        descriptors = np.zeros((B, max_n, 256), dtype=np.float32)
        
        for b in range(B):
            n = len(all_keypoints[b])
            keypoints[b, :n] = all_keypoints[b]
            scores[b, :n] = all_scores[b]
            descriptors[b, :n] = all_descriptors[b]
        
        return {
            "keypoints": torch.from_numpy(keypoints).to(device),
            "scores": torch.from_numpy(scores).to(device),
            "descriptors": torch.from_numpy(descriptors).to(device),
        }


class LightGlueONNXWrapper:
    """Wraps LightGlueONNX for hloc's match_features pipeline."""
    
    default_conf = {
        "model_path": None,
        "features": "superpoint",
        "device": "cuda",
    }
    required_inputs = [
        "image0", "keypoints0", "descriptors0",
        "image1", "keypoints1", "descriptors1",
    ]
    
    DESCRIPTOR_DIMS = {
        "superpoint": 256,
        "disk": 128,
        "aliked": 128,
    }
    
    def __init__(self, conf: Dict):
        self.conf = {**self.default_conf, **conf}
        features = self.conf.get("features", "superpoint")
        self.desc_dim = self.DESCRIPTOR_DIMS.get(features, 256)
        
        self._model = LightGlueONNX(
            model_path=self.conf.get("model_path"),
            features=features,
            device=self.conf.get("device", "cuda"),
        )
        # Warmup to trigger CUDA/TensorRT kernel compilation
        self._warmup()
    
    def _warmup(self):
        """Run dummy inference to compile CUDA kernels."""
        n_kpts = 500
        # Use GPU path with IO Binding if available (warms up the right code path)
        if hasattr(self._model, 'match_gpu') and getattr(self._model, 'use_io_binding', False):
            dummy_kp0 = torch.rand(n_kpts, 2, device='cuda') * 500
            dummy_kp1 = torch.rand(n_kpts, 2, device='cuda') * 500
            dummy_desc0 = torch.rand(n_kpts, self.desc_dim, device='cuda')
            dummy_desc1 = torch.rand(n_kpts, self.desc_dim, device='cuda')
            _ = self._model.match_gpu(dummy_kp0, dummy_kp1, dummy_desc0, dummy_desc1,
                                      image_size0=(768, 1024), image_size1=(768, 1024))
        else:
            dummy_kp0 = np.random.rand(n_kpts, 2).astype(np.float32) * 500
            dummy_kp1 = np.random.rand(n_kpts, 2).astype(np.float32) * 500
            dummy_desc0 = np.random.rand(n_kpts, self.desc_dim).astype(np.float32)
            dummy_desc1 = np.random.rand(n_kpts, self.desc_dim).astype(np.float32)
            _ = self._model.match(dummy_kp0, dummy_kp1, dummy_desc0, dummy_desc1,
                                  image_size0=(768, 1024), image_size1=(768, 1024))
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def __call__(self, data: Dict) -> Dict:
        kp0 = data["keypoints0"]
        kp1 = data["keypoints1"]
        desc0 = data["descriptors0"]
        desc1 = data["descriptors1"]
        
        shape0 = self._get_shape(data.get("image0"))
        shape1 = self._get_shape(data.get("image1"))
        
        if kp0.dim() == 2:
            kp0 = kp0.unsqueeze(0)
            kp1 = kp1.unsqueeze(0)
            desc0 = desc0.unsqueeze(0)
            desc1 = desc1.unsqueeze(0)
        
        B = kp0.shape[0]
        N0, N1 = kp0.shape[1], kp1.shape[1]
        
        if desc0.shape[-1] != self.desc_dim:
            desc0 = desc0.transpose(-1, -2)
            desc1 = desc1.transpose(-1, -2)
        
        device = data["keypoints0"].device
        all_matches0 = []
        all_matches1 = []
        all_scores0 = []
        all_scores1 = []
        
        # Use GPU path with IO Binding if available (faster)
        use_gpu = (
            hasattr(self._model, 'match_gpu') and 
            getattr(self._model, 'use_io_binding', False) and 
            device.type == 'cuda'
        )
        
        for b in range(B):
            img_size0 = (shape0[-2], shape0[-1]) if shape0 is not None else None
            img_size1 = (shape1[-2], shape1[-1]) if shape1 is not None else None
            
            # If GPU path is available, use it for matching
            if use_gpu:
                # GPU path avoids CPU copies
                matches, scores = self._model.match_gpu(
                    kp0[b], kp1[b], desc0[b], desc1[b],
                    image_size0=img_size0, image_size1=img_size1,
                )
            # Else, fallback to CPU path
            else:
                # Convert tensors to numpy for CPU matching
                kp0_np = kp0[b].cpu().numpy().astype(np.float32)
                kp1_np = kp1[b].cpu().numpy().astype(np.float32)
                desc0_np = desc0[b].cpu().numpy().astype(np.float32)
                desc1_np = desc1[b].cpu().numpy().astype(np.float32)
                
                matches, scores = self._model.match(
                    kp0_np, kp1_np, desc0_np, desc1_np,
                    image_size0=img_size0, image_size1=img_size1,
                )
            
            matches0 = np.full(N0, -1, dtype=np.int64)
            matches1 = np.full(N1, -1, dtype=np.int64)
            scores0 = np.zeros(N0, dtype=np.float32)
            scores1 = np.zeros(N1, dtype=np.float32)
            
            if len(matches) > 0:
                matches0[matches[:, 0]] = matches[:, 1]
                matches1[matches[:, 1]] = matches[:, 0]
                scores0[matches[:, 0]] = scores
                scores1[matches[:, 1]] = scores
            
            all_matches0.append(matches0)
            all_matches1.append(matches1)
            all_scores0.append(scores0)
            all_scores1.append(scores1)
        
        matches0 = torch.from_numpy(np.stack(all_matches0)).to(device)
        matches1 = torch.from_numpy(np.stack(all_matches1)).to(device)
        scores0 = torch.from_numpy(np.stack(all_scores0)).to(device)
        scores1 = torch.from_numpy(np.stack(all_scores1)).to(device)
        
        if data["keypoints0"].dim() == 2:
            matches0 = matches0.squeeze(0)
            matches1 = matches1.squeeze(0)
            scores0 = scores0.squeeze(0)
            scores1 = scores1.squeeze(0)
        
        return {
            "matches0": matches0,
            "matches1": matches1,
            "matching_scores0": scores0,
            "matching_scores1": scores1,
        }
    
    def _get_shape(self, image):
        if image is None:
            return None
        if isinstance(image, dict):
            return image.get("shape", image.get("image_size"))
        if isinstance(image, torch.Tensor):
            return image.shape
        return None


class MegaLocONNXWrapper:
    """Wraps MegaLocONNX for hloc's extract_features pipeline."""
    
    default_conf = {
        "model_path": None,
        "resize": 322,
        "use_tensorrt": True,
        "device": "cuda",
    }
    required_inputs = ["image"]
    
    def __init__(self, conf: Dict):
        self.conf = {**self.default_conf, **conf}
        self._model = MegaLocONNX(
            model_path=self.conf.get("model_path"),
            device=self.conf.get("device", "cuda"),
            use_tensorrt=self.conf.get("use_tensorrt", True),
        )
        self.resize = self.conf.get("resize", 322)
        self._model.warmup(3)
    
    def eval(self):
        return self
    
    def to(self, device):
        return self
    
    def __call__(self, data: Dict) -> Dict:
        image = data["image"]
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        device = data["image"].device
        
        img_np = image[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        desc = self._model.extract(img_np, resize=self.resize)
        desc_tensor = torch.from_numpy(desc).to(device)
        
        return {"global_descriptor": desc_tensor}
