"""
hloc wrappers for ONNX models.

Extends SuperPointONNX, LightGlueONNX, and MegaLocONNX with hloc-compatible
__call__ interface for extract_features and match_features pipelines.
"""

from typing import Dict
import numpy as np
import torch

from .superpoint_onnx import SuperPointONNX
from .lightglue_onnx import LightGlueONNX
from .megaloc_onnx import MegaLocONNX


class SuperPointONNXWrapper(SuperPointONNX):
    """SuperPointONNX with hloc-compatible __call__ interface."""
    
    default_conf = {
        "model_path": None,
        "max_num_keypoints": 2048,
        "detection_threshold": 0.0005,
        "device": "cuda",
        "force_num_keypoints": False,
    }
    required_inputs = ["image"]
    
    def __init__(self, conf: Dict):
        self.conf = {**self.default_conf, **conf}
        super().__init__(
            model_path=self.conf.get("model_path"),
            max_num_keypoints=self.conf.get("max_num_keypoints", 2048),
            detection_threshold=self.conf.get("detection_threshold", 0.0005),
            device=self.conf.get("device", "cuda"),
            force_num_keypoints=self.conf.get("force_num_keypoints", False),
        )
        # Skip warmup for now (TODO: fix warmup input shapes)
        # self.warmup()
    
    def eval(self):
        """No-op for hloc compatibility (ONNX models are always in eval mode)."""
        return self
    
    def to(self, device):
        """No-op for hloc compatibility (ONNX device is set at construction)."""
        return self
    
    def __call__(self, data: Dict) -> Dict:
        """
        hloc-compatible interface: Dict with 'image' tensor -> Dict with features.
        
        Args:
            data: {"image": torch.Tensor [B, C, H, W] or [C, H, W]}
        
        Returns:
            {"keypoints": [B, N, 2], "scores": [B, N], "descriptors": [B, N, 256]}
        """
        image = data["image"]
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        B, C, H, W = image.shape
        
        # Convert to grayscale if RGB
        if C == 3:
            image = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        
        device = data["image"].device
        
        # Use GPU path if available
        if self.use_io_binding and device.type == 'cuda':
            all_keypoints = []
            all_scores = []
            all_descriptors = []
            
            for b in range(B):
                kp, sc, desc = self.extract_gpu(image[b])
                all_keypoints.append(kp)
                all_scores.append(sc)
                all_descriptors.append(desc)
            
            max_n = max(len(kp) for kp in all_keypoints) if all_keypoints else 0
            
            keypoints = torch.zeros((B, max_n, 2), dtype=torch.float32, device=device)
            scores = torch.zeros((B, max_n), dtype=torch.float32, device=device)
            descriptors = torch.zeros((B, max_n, 256), dtype=torch.float32, device=device)
            
            for b in range(B):
                n = len(all_keypoints[b])
                keypoints[b, :n] = all_keypoints[b]
                scores[b, :n] = all_scores[b]
                descriptors[b, :n] = all_descriptors[b]
            
            return {"keypoints": keypoints, "scores": scores, "descriptors": descriptors}
        
        # Fallback to CPU path
        all_keypoints = []
        all_scores = []
        all_descriptors = []
        
        for b in range(B):
            img_np = image[b].cpu().numpy().astype(np.float32)
            kp, sc, desc = self.extract(img_np)
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


class LightGlueONNXWrapper(LightGlueONNX):
    """LightGlueONNX with hloc-compatible __call__ interface."""
    
    default_conf = {
        "model_path": None,
        "features": "superpoint",
        "device": "cuda",
    }
    required_inputs = [
        "image0", "keypoints0", "descriptors0",
        "image1", "keypoints1", "descriptors1",
    ]
    
    def __init__(self, conf: Dict):
        self.conf = {**self.default_conf, **conf}
        super().__init__(
            model_path=self.conf.get("model_path"),
            features=self.conf.get("features", "superpoint"),
            device=self.conf.get("device", "cuda"),
        )
        # Warmup CUDA/TensorRT kernels
        self.warmup()
    
    def eval(self):
        """No-op for hloc compatibility (ONNX models are always in eval mode)."""
        return self
    
    def to(self, device):
        """No-op for hloc compatibility (ONNX device is set at construction)."""
        return self
    
    def __call__(self, data: Dict) -> Dict:
        """
        hloc-compatible interface: Dict with keypoints/descriptors -> matches.
        
        Args:
            data: {"keypoints0", "descriptors0", "keypoints1", "descriptors1", ...}
        
        Returns:
            {"matches0": [B, N0], "matches1": [B, N1], "matching_scores0/1": ...}
        """
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
        
        use_gpu = self.use_io_binding and device.type == 'cuda'
        
        for b in range(B):
            img_size0 = (shape0[-2], shape0[-1]) if shape0 is not None else None
            img_size1 = (shape1[-2], shape1[-1]) if shape1 is not None else None
            
            if use_gpu:
                matches, scores = self.match_gpu(
                    kp0[b], kp1[b], desc0[b], desc1[b],
                    image_size0=img_size0, image_size1=img_size1,
                )
            else:
                kp0_np = kp0[b].cpu().numpy().astype(np.float32)
                kp1_np = kp1[b].cpu().numpy().astype(np.float32)
                desc0_np = desc0[b].cpu().numpy().astype(np.float32)
                desc1_np = desc1[b].cpu().numpy().astype(np.float32)
                
                matches, scores = self.match(
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


class MegaLocONNXWrapper(MegaLocONNX):
    """MegaLocONNX with hloc-compatible __call__ interface."""
    
    default_conf = {
        "model_path": None,
        "resize": 322,
        "device": "cuda",
    }
    required_inputs = ["image"]
    
    def __init__(self, conf: Dict):
        self.conf = {**self.default_conf, **conf}
        super().__init__(
            model_path=self.conf.get("model_path"),
            device=self.conf.get("device", "cuda"),
            resize=self.conf.get("resize", 322),
        )
        self.warmup(3)
    
    def eval(self):
        """No-op for hloc compatibility (ONNX models are always in eval mode)."""
        return self
    
    def to(self, device):
        """No-op for hloc compatibility (ONNX device is set at construction)."""
        return self
    
    def __call__(self, data: Dict) -> Dict:
        """
        hloc-compatible interface: Dict with 'image' tensor -> global descriptor.
        
        Args:
            data: {"image": torch.Tensor [B, 3, H, W] or [3, H, W]}
        
        Returns:
            {"global_descriptor": torch.Tensor [B, 8448]}
        """
        image = data["image"]
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Get batch size and device
        B = image.shape[0]
        device = data["image"].device
        
        # GPU path: keep tensor on GPU, use IO binding
        if self.use_io_binding and device.type == 'cuda':
            # Resize on GPU if needed
            if image.shape[-1] != self.resize or image.shape[-2] != self.resize:
                image = torch.nn.functional.interpolate(
                    image, size=(self.resize, self.resize), mode='bilinear', align_corners=False
                )
            
            # Extract descriptors for each image in the batch
            descriptors = []
            for b in range(B):
                desc = self.extract_gpu(image[b])
                descriptors.append(desc)
            
            return {"global_descriptor": torch.stack(descriptors)}
        
        # CPU fallback: convert [1, 3, H, W] float [0,1] -> [H, W, 3] uint8
        descriptors = []
        for b in range(B):
            img_np = image[b].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            desc = self.extract(img_np)
            descriptors.append(torch.from_numpy(desc))
        
        return {"global_descriptor": torch.stack(descriptors).to(device)}
