"""
Fully in-memory visual localization pipeline.

This module provides a complete localization pipeline that operates entirely
in memory, eliminating all disk I/O during inference. Pre-built map data
(features, global descriptors, SfM model) is loaded once at startup and
reused for all queries.

Main entry point:
    localize_image() - Image path -> camera pose (no intermediate files)
"""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import pycolmap
import torch

from .utils.cache import LRUCache


# =============================================================================
# Image Preprocessing
# =============================================================================


def _preprocess_image(
    image_path: Path,
    grayscale: bool = True,
    resize_max: int = 1024,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load and preprocess an image for feature extraction.

    Returns:
        (image_tensor, original_size) where image_tensor is [1, C, H, W]
    """
    if grayscale:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError(f"Cannot read image {image_path}")

    original_size = np.array(image.shape[:2][::-1])  # (W, H)

    # Resize if needed
    if resize_max and max(original_size) > resize_max:
        scale = resize_max / max(original_size)
        new_size = tuple(int(round(x * scale)) for x in original_size)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # Convert to tensor
    if grayscale:
        image = image[None]  # Add channel dim
    else:
        image = image.transpose((2, 0, 1))  # HWC to CHW

    image = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

    return image, original_size


# =============================================================================
# Feature Extraction (In-Memory)
# =============================================================================


@torch.inference_mode()
def extract_features_inmem(
    image_path: Path,
    model,
    conf: Dict,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Extract features from an image without writing to disk.

    Args:
        image_path: Path to query image.
        model: Pre-loaded feature extractor (SuperPoint, etc.).
        conf: Preprocessing config with 'grayscale', 'resize_max'.
        device: Device to run on.

    Returns:
        Dict with 'keypoints', 'descriptors', 'scores', 'image_size' tensors.
    """
    preproc = conf.get("preprocessing", {})
    image, original_size = _preprocess_image(
        image_path,
        grayscale=preproc.get("grayscale", True),
        resize_max=preproc.get("resize_max", 1024),
    )

    # Run model
    pred = model({"image": image.to(device)})

    # Post-process keypoints to original resolution
    result = {}
    for k, v in pred.items():
        result[k] = v[0]  # Remove batch dim

    result["image_size"] = tuple(original_size.tolist())

    if "keypoints" in result:
        size = np.array(image.shape[-2:][::-1])
        # Scale keypoints on GPU (avoid CPU round-trip)
        scales = torch.tensor(
            (original_size / size).astype(np.float32), device=device
        )
        kp = result["keypoints"]
        result["keypoints"] = (kp + 0.5) * scales - 0.5

    return result


@torch.inference_mode()
def extract_global_descriptor_inmem(
    image_path: Path,
    model,
    conf: Dict,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Extract global descriptor from an image without writing to disk.

    Args:
        image_path: Path to query image.
        model: Pre-loaded global descriptor model (MegaLoc, NetVLAD, etc.).
        conf: Preprocessing config.
        device: Device to run on.

    Returns:
        Global descriptor tensor [D].
    """
    preproc = conf.get("preprocessing", {})
    image, _ = _preprocess_image(
        image_path,
        grayscale=preproc.get("grayscale", False),
        resize_max=preproc.get("resize_max", 1024),
    )

    pred = model({"image": image.to(device)})
    return pred["global_descriptor"][0]  # [D]


# =============================================================================
# Retrieval (In-Memory)
# =============================================================================


def retrieve_topk_inmem(
    query_desc: torch.Tensor,
    db_descriptors: torch.Tensor,
    db_names: List[str],
    num_matched: int = 30,
) -> List[str]:
    """
    Find top-k similar database images by descriptor similarity.

    Args:
        query_desc: Query global descriptor [D].
        db_descriptors: Database descriptors [N, D].
        db_names: Database image names.
        num_matched: Number of matches to return.

    Returns:
        List of top-k database image names.
    """
    # Compute similarities
    sim = torch.einsum("d,nd->n", query_desc, db_descriptors)

    # Get top-k indices
    k = min(num_matched, len(db_names))
    _, indices = torch.topk(sim, k)

    return [db_names[i] for i in indices.cpu().numpy()]


# =============================================================================
# Feature Loading from HDF5
# =============================================================================


def _load_features_to_device(
    h5_path: Path,
    names: set,
    device: str,
) -> Dict[str, Dict]:
    """Load features from HDF5 directly to GPU."""
    features = {}
    with h5py.File(h5_path, "r") as fd:
        for name in names:
            grp = fd[name]
            features[name] = {
                k: torch.from_numpy(v.__array__()).float().to(device)
                for k, v in grp.items()
                if k != "image_size"
            }
            features[name]["image_size"] = tuple(grp["image_size"])
    return features


def _load_features_via_cache(
    h5_path: Path,
    names: set,
    cache: LRUCache,
) -> Dict[str, Dict]:
    """Load features through LRU cache (CPU tensors)."""
    hits, misses = cache.partition_keys(names)
    cache.record_stats(hits=len(hits), misses=len(misses))

    if misses:
        with h5py.File(h5_path, "r") as fd:
            for name in misses:
                grp = fd[name]
                feats = {
                    k: torch.from_numpy(v.__array__()).float()
                    for k, v in grp.items()
                    if k != "image_size"
                }
                feats["image_size"] = tuple(grp["image_size"])
                cache.put(name, feats)

    return cache.get_many(names)


# =============================================================================
# Matching (In-Memory)
# =============================================================================


def _postprocess_matches(pred: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw matcher output to (matches, scores) arrays."""
    raw_matches = pred["matches0"][0].cpu().numpy()
    raw_scores = (
        pred["matching_scores0"][0].cpu().numpy()
        if "matching_scores0" in pred
        else None
    )

    valid = np.where(raw_matches != -1)[0]
    matches = np.stack([valid, raw_matches[valid]], axis=-1)
    scores = raw_scores[valid] if raw_scores is not None else np.ones(len(valid))

    return matches, scores


@torch.inference_mode()
def match_query_to_db_inmem(
    query_feats: Dict[str, torch.Tensor],
    db_names: List[str],
    features_ref: Path,
    matcher_model,
    ref_cache: Optional[LRUCache] = None,
    device: str = "cuda",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Match query features against database images.

    Args:
        query_feats: Query features dict with 'keypoints', 'descriptors', etc.
        db_names: Database image names to match against.
        features_ref: Path to reference features HDF5.
        matcher_model: Pre-loaded matcher.
        ref_cache: Optional LRU cache for reference features.
        device: Device to run on.

    Returns:
        Dict mapping db_name -> (matches, scores).
    """
    if not db_names:
        return {}

    # Load reference features
    ref_names = set(db_names)
    if ref_cache is not None:
        ref_feats = _load_features_via_cache(features_ref, ref_names, ref_cache)
    else:
        ref_feats = _load_features_to_device(features_ref, ref_names, device)

    # Pre-build query data dict (reused for all matches)
    qf_data = {
        f"{k}0": (v.to(device) if isinstance(v, torch.Tensor) and v.device.type != device else v).unsqueeze(0)
        for k, v in query_feats.items()
        if k != "image_size" and isinstance(v, torch.Tensor)
    }
    qf_shape = {"image0": {"shape": (1, 1) + query_feats["image_size"][::-1]}}

    # Match against each database image
    results = {}
    for db_name in db_names:
        rf = ref_feats[db_name]

        # Build input dict (reuse query data)
        data = dict(qf_data)
        data.update(
            {
                f"{k}1": (v.to(device) if v.device.type != device else v).unsqueeze(0)
                for k, v in rf.items()
                if k != "image_size" and isinstance(v, torch.Tensor)
            }
        )
        data.update(qf_shape)
        data["image1"] = {"shape": (1, 1) + rf["image_size"][::-1]}

        pred = matcher_model(data)
        matches, scores = _postprocess_matches(pred)
        results[db_name] = (matches, scores)

    return results


# =============================================================================
# Covisibility Clustering
# =============================================================================


def build_covisibility_graph(
    reconstruction: pycolmap.Reconstruction,
) -> Dict[int, set]:
    """
    Pre-build covisibility graph from reconstruction (call once at map load).
    
    Args:
        reconstruction: COLMAP reconstruction.
    
    Returns:
        Dict mapping image_id -> set of covisible image_ids.
    """
    graph = {}
    for img_id, img in reconstruction.images.items():
        covisible = set()
        for p2D in img.points2D:
            if p2D.has_point3D():
                for obs in reconstruction.points3D[p2D.point3D_id].track.elements:
                    if obs.image_id != img_id:
                        covisible.add(obs.image_id)
        graph[img_id] = covisible
    return graph


def cluster_by_covisibility(
    frame_ids: List[int],
    reconstruction: pycolmap.Reconstruction,
    covisibility_graph: Optional[Dict[int, set]] = None,
) -> List[List[int]]:
    """
    Group frames into clusters based on shared 3D point observations.

    Args:
        frame_ids: List of database image IDs to cluster.
        reconstruction: COLMAP reconstruction (used if covisibility_graph is None).
        covisibility_graph: Pre-built graph from build_covisibility_graph() (faster).

    Returns clusters sorted by size (largest first).
    """
    clusters = []
    visited = set()
    frame_set = set(frame_ids)

    for seed_id in frame_ids:
        if seed_id in visited:
            continue

        cluster = []
        queue = {seed_id}

        while queue:
            fid = queue.pop()
            if fid in visited:
                continue
            visited.add(fid)
            cluster.append(fid)

            # If covisibility graph is provided (fast path)
            if covisibility_graph is not None:
                covisible = covisibility_graph.get(fid, set())
            # Compute on-the-fly (slow path)
            else:
                observed = reconstruction.images[fid].points2D
                covisible = {
                    obs.image_id
                    for p2D in observed
                    if p2D.has_point3D()
                    for obs in reconstruction.points3D[p2D.point3D_id].track.elements
                }
            
            queue |= (covisible & frame_set) - visited

        clusters.append(cluster)

    return sorted(clusters, key=len, reverse=True)


# =============================================================================
# Localization
# =============================================================================


def _build_2d3d_correspondences(
    db_ids: List[int],
    matches_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reconstruction: pycolmap.Reconstruction,
) -> Tuple[List[int], List[int], int]:
    """Aggregate 2D-3D correspondences from matches."""
    kp_to_3d = defaultdict(list)
    num_matches = 0

    for db_id in db_ids:
        image = reconstruction.images[db_id]
        if image.num_points3D == 0 or image.name not in matches_dict:
            continue

        p3d_ids = np.array(
            [p.point3D_id if p.has_point3D() else -1 for p in image.points2D]
        )

        matches, _ = matches_dict[image.name]
        valid = matches[p3d_ids[matches[:, 1]] != -1]
        num_matches += len(valid)

        for kp_idx, db_idx in valid:
            p3d_id = p3d_ids[db_idx]
            if p3d_id not in kp_to_3d[kp_idx]:
                kp_to_3d[kp_idx].append(p3d_id)

    kp_idxs = [i for i in kp_to_3d for _ in kp_to_3d[i]]
    p3d_ids = [j for i in kp_to_3d for j in kp_to_3d[i]]

    return kp_idxs, p3d_ids, num_matches


def _localize_from_matches(
    keypoints: np.ndarray,
    camera: pycolmap.Camera,
    db_ids: List[int],
    matches_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    reconstruction: pycolmap.Reconstruction,
    ransac_thresh: float = 12.0,
) -> Optional[Dict]:
    """Estimate camera pose from 2D-3D correspondences via PnP+RANSAC."""
    kp_idxs, p3d_ids, num_matches = _build_2d3d_correspondences(
        db_ids, matches_dict, reconstruction
    )

    if not kp_idxs:
        return None

    points2D = keypoints[kp_idxs]
    points3D = [reconstruction.points3D[i].xyz for i in p3d_ids]

    ret = pycolmap.estimate_and_refine_absolute_pose(
        points2D,
        points3D,
        camera,
        estimation_options={"ransac": {"max_error": ransac_thresh}},
        refinement_options={},
    )

    if ret is not None:
        ret["camera"] = camera
        ret["num_matches"] = num_matches

    return ret


# =============================================================================
# Main Entry Points
# =============================================================================


def localize_image(
    image_path: Path,
    query_camera: pycolmap.Camera,
    # Pre-loaded map data
    reconstruction: pycolmap.Reconstruction,
    db_names: List[str],
    db_descriptors: torch.Tensor,
    features_ref: Path,
    # Pre-loaded models
    feature_model,
    retrieval_model,
    matcher_model,
    # Configs
    feature_conf: Dict,
    retrieval_conf: Dict,
    # Options
    num_matched: int = 30,
    ref_cache: Optional[LRUCache] = None,
    covisibility_clustering: bool = True,
    covisibility_graph: Optional[Dict[int, set]] = None,
    ransac_thresh: float = 12.0,
    device: str = "cuda",
) -> Optional[Dict]:
    """
    Localize an image against a pre-built map. Fully in-memory, no disk writes.

    Args:
        image_path: Path to query image.
        query_camera: Query camera intrinsics.
        reconstruction: Pre-loaded SfM model.
        db_names: Database image names (for retrieval).
        db_descriptors: Pre-loaded database global descriptors [N, D].
        features_ref: Path to reference features HDF5.
        feature_model: Pre-loaded local feature extractor.
        retrieval_model: Pre-loaded global descriptor model.
        matcher_model: Pre-loaded feature matcher.
        feature_conf: Feature extraction config.
        retrieval_conf: Retrieval config.
        num_matched: Number of database images to match.
        ref_cache: Optional LRU cache for reference features.
        covisibility_clustering: Whether to cluster by covisibility.
        covisibility_graph: Pre-built graph from build_covisibility_graph() (faster).
        ransac_thresh: PnP RANSAC threshold in pixels.
        device: Device to run on.

    Returns:
        PnP result dict with 'cam_from_world', 'num_inliers', etc.
        None if localization fails.
    """
    db_name_to_id = {img.name: i for i, img in reconstruction.images.items()}

    # 1. Extract local features
    query_feats = extract_features_inmem(image_path, feature_model, feature_conf, device)

    # 2. Extract global descriptor and retrieve similar images
    query_desc = extract_global_descriptor_inmem(
        image_path, retrieval_model, retrieval_conf, device
    )
    retrieved_names = retrieve_topk_inmem(
        query_desc, db_descriptors, db_names, num_matched
    )

    # 3. Match against retrieved images
    matches_dict = match_query_to_db_inmem(
        query_feats, retrieved_names, features_ref, matcher_model, ref_cache, device
    )

    # 4. Get database IDs
    db_ids = [db_name_to_id[n] for n in retrieved_names if n in db_name_to_id]
    if not db_ids:
        return None

    # 5. Get keypoints for localization (convert to numpy, add 0.5 for COLMAP)
    keypoints = query_feats["keypoints"].cpu().numpy() + 0.5

    # 6. Localize
    if covisibility_clustering:
        clusters = cluster_by_covisibility(db_ids, reconstruction, covisibility_graph)
        best_ret, best_inliers = None, 0

        for cluster_ids in clusters:
            ret = _localize_from_matches(
                keypoints, query_camera, cluster_ids,
                matches_dict, reconstruction, ransac_thresh
            )
            if ret is not None and ret["num_inliers"] > best_inliers:
                best_ret, best_inliers = ret, ret["num_inliers"]

        return best_ret
    else:
        return _localize_from_matches(
            keypoints, query_camera, db_ids,
            matches_dict, reconstruction, ransac_thresh
        )


# =============================================================================
# Legacy API (for compatibility with existing code)
# =============================================================================


def match_pairs(
    pairs: List[Tuple[str, str]],
    features_q: Path,
    features_ref: Path,
    model,
    ref_cache: Optional[LRUCache] = None,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Match feature pairs from HDF5 files (legacy API)."""
    if not pairs:
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_names = set(p[0] for p in pairs)
    ref_names = set(p[1] for p in pairs)

    query_feats = _load_features_to_device(features_q, query_names, device)

    if ref_cache is not None:
        ref_feats = _load_features_via_cache(features_ref, ref_names, ref_cache)
    else:
        ref_feats = _load_features_to_device(features_ref, ref_names, device)

    results = {}
    for name0, name1 in pairs:
        qf, rf = query_feats[name0], ref_feats[name1]

        data = {
            f"{k}0": (v.to(device) if v.device.type != device else v).unsqueeze(0)
            for k, v in qf.items()
            if k != "image_size"
        }
        data.update(
            {
                f"{k}1": (v.to(device) if v.device.type != device else v).unsqueeze(0)
                for k, v in rf.items()
                if k != "image_size"
            }
        )
        data["image0"] = {"shape": (1, 1) + qf["image_size"][::-1]}
        data["image1"] = {"shape": (1, 1) + rf["image_size"][::-1]}

        pred = model(data)
        matches, scores = _postprocess_matches(pred)
        results.setdefault(name0, {})[name1] = (matches, scores)

    return results


def match_and_localize(
    pairs: List[Tuple[str, str]],
    features_q: Path,
    features_ref: Path,
    reference_sfm: Union[Path, pycolmap.Reconstruction],
    query_name: str,
    query_camera: pycolmap.Camera,
    matcher_model,
    ref_cache: Optional[LRUCache] = None,
    covisibility_clustering: bool = True,
    ransac_thresh: float = 12.0,
) -> Optional[Dict]:
    """Match and localize from HDF5 files (legacy API)."""
    if not isinstance(reference_sfm, pycolmap.Reconstruction):
        reference_sfm = pycolmap.Reconstruction(reference_sfm)

    db_name_to_id = {img.name: i for i, img in reference_sfm.images.items()}
    db_names = [p[1] for p in pairs if p[0] == query_name]
    db_ids = [db_name_to_id[n] for n in db_names if n in db_name_to_id]

    if not db_ids:
        return None

    all_matches = match_pairs(pairs, features_q, features_ref, matcher_model, ref_cache)
    query_matches = all_matches.get(query_name, {})

    with h5py.File(features_q, "r") as fd:
        keypoints = fd[query_name]["keypoints"].__array__() + 0.5

    if covisibility_clustering:
        clusters = cluster_by_covisibility(db_ids, reference_sfm)
        best_ret, best_inliers = None, 0

        for cluster_ids in clusters:
            ret = _localize_from_matches(
                keypoints, query_camera, cluster_ids,
                query_matches, reference_sfm, ransac_thresh
            )
            if ret is not None and ret["num_inliers"] > best_inliers:
                best_ret, best_inliers = ret, ret["num_inliers"]

        return best_ret
    else:
        return _localize_from_matches(
            keypoints, query_camera, db_ids,
            query_matches, reference_sfm, ransac_thresh
        )
