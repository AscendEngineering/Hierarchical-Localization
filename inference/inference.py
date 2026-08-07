#!/usr/bin/env python3
"""
Localize query images against a pre-built map and visualize results.
Usage: ./inference.py my_desk

This script:
1. Loads a pre-built map from inference/maps/{map_name}/
2. Localizes each query image using fully in-memory pipeline
3. Visualizes the results with origin, map cameras, and query poses
"""

# Standard library imports
import argparse
import sys
import time
from pathlib import Path

# NumPy for numerical operations
import numpy as np

# pycolmap for COLMAP reconstruction handling
import pycolmap

# OpenCV for image reading
import cv2

# hloc modules for feature extraction, matching, and localization
from hloc import extract_features, match_features
from hloc.utils.cache import LRUCache
from hloc.utils.io import list_h5_names
from hloc.match_and_localize import localize_image

# Local visualization utilities
from vis import visualize_localization, show_figure


def get_query_images(queries_dir: Path) -> list:
    """
    Get sorted list of query images from the queries directory.
    Supports common image formats: jpg, jpeg, png, bmp, webp
    """
    
    # Define supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    
    # Collect all image files
    images = []
    
    # Iterate through all files in queries directory
    for f in queries_dir.iterdir():
        # Check if file has a supported image extension
        if f.suffix.lower() in image_extensions:
            # Add to list
            images.append(f)
    
    # Sort images by filename for consistent ordering
    images = sorted(images, key=lambda x: x.name)
    
    # Return sorted list
    return images


def run_inference(map_name: str):
    """
    Run localization inference on all query images.
    
    Steps:
    1. Load the pre-built map
    2. Extract features from query images
    3. Find similar map images using retrieval
    4. Match query features to map features
    5. Localize queries using PnP + RANSAC
    6. Visualize results
    """
    
    # Get the inference directory (where this script is located)
    inference_dir = Path(__file__).parent
    
    # Define paths
    # Map directory containing the reconstruction
    map_dir = inference_dir / "maps" / map_name
    
    # SfM reconstruction directory
    sfm_dir = map_dir / "sfm"
    
    # Outputs directory (features, matches from build_map.py)
    map_outputs_dir = map_dir / "outputs"
    
    # Queries directory
    queries_dir = inference_dir / "queries"
    
    # Output directory for query processing
    query_outputs_dir = map_dir / "query_outputs"
    query_outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # ============ VALIDATE INPUTS ============
    print(f"\n{'='*50}")
    print(f"Running inference on map: {map_name}")
    print(f"{'='*50}\n")
    
    # If map doesn't exist, exit
    if not sfm_dir.exists():
        print(f"Error: Map not found at {sfm_dir}")
        print(f"Run 'python build_map.py {map_name}.mp4' first.")
        sys.exit(1)
    
    # If queries directory doesn't exist, exit
    if not queries_dir.exists():
        print(f"Error: Queries directory not found at {queries_dir}")
        print("Create the directory and add query images.")
        sys.exit(1)
    
    # Get list of query images
    query_images = get_query_images(queries_dir)
    
    # If there are no query images, exit
    if len(query_images) == 0:
        print(f"Error: No query images found in {queries_dir}")
        print("Add .jpg, .png, or other image files to the queries folder.")
        sys.exit(1)
    
    # Print query image info
    print(f"Found {len(query_images)} query images:")
    for idx, img in enumerate(query_images):
        print(f"  img{idx + 1}: {img.name}")
    
    # Track timing for each step
    timings = {}
    total_start = time.time()
    
    # ============ ONE-TIME SETUP ============
    print(f"\n[SETUP] Loading map and models (one-time)...")
    setup_start = time.time()
    
    # Load COLMAP reconstruction
    colmap_model = pycolmap.Reconstruction(sfm_dir)
    print(f"  Map: {colmap_model.num_reg_images()} images, {colmap_model.num_points3D()} points")
    
    # Configuration for feature extraction (TensorRT accelerated)
    feature_conf = {
        "output": "feats-superpoint-n4096-r1024",
        "model": {
            "name": "superpoint_onnx",
            "max_num_keypoints": 4096,
            "use_tensorrt": True,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    
    # Configuration for retrieval (ONNX/TensorRT accelerated)
    retrieval_conf = extract_features.confs["megaloc_onnx"]
    
    # Configuration for matching
    matcher_conf = match_features.confs["superpoint_onnx+lightglue_onnx"]
    
    # Pre-load all models (one-time)
    print("  Loading SuperPoint ONNX with TensorRT...")
    superpoint_model = extract_features.get_model(feature_conf)
    
    print("  Loading MegaLoc ONNX with TensorRT...")
    megaloc_model = extract_features.get_model(retrieval_conf)
    
    print("  Loading LightGlue TensorRT matcher...")
    matcher_model = match_features.get_model(matcher_conf)
    
    # Create LRU cache for map features (persists across queries)
    ref_cache = LRUCache(max_items=200, device="cuda")
    
    # Paths to pre-computed map features (compatible with both megaloc and megaloc_onnx)
    map_features_path = map_outputs_dir / f"{feature_conf['output']}.h5"
    map_retrieval_path = map_outputs_dir / f"{retrieval_conf['output']}.h5"
    
    # Pre-load database global descriptors for retrieval (no disk I/O during inference)
    print("  Loading database global descriptors...")
    import h5py
    import torch
    db_names = list_h5_names(map_retrieval_path)
    with h5py.File(map_retrieval_path, "r") as fd:
        db_descriptors = torch.stack([
            torch.from_numpy(fd[name]["global_descriptor"].__array__()).float()
            for name in db_names
        ]).cuda()
    print(f"  Loaded {len(db_names)} database descriptors")
    
    timings['setup'] = time.time() - setup_start
    print(f"  Setup time: {timings['setup']:.2f}s")
    
    # ============ PROCESS EACH IMAGE (TRUE PER-IMAGE LATENCY) ============
    print(f"\n[INFERENCE] Processing {len(query_images)} queries (fully in-memory)...")
    
    per_image_timings = {}
    all_poses = {}

    for img_idx, img_path in enumerate(query_images):
        img_name = img_path.name
        print(f"\n  [{img_idx + 1}/{len(query_images)}] {img_name}")
        
        img_total_start = time.time()
        
        # Create camera from image dimensions
        img_cv = cv2.imread(str(img_path))
        h, w = img_cv.shape[:2]
        focal = 0.7 * max(w, h)
        cx, cy = w / 2, h / 2
        query_camera = pycolmap.Camera(
            model="SIMPLE_PINHOLE",
            width=w,
            height=h,
            params=[focal, cx, cy],
        )
        
        # Single call: image -> pose (fully in-memory, no disk I/O)
        pnp_ret = localize_image(
            image_path=img_path,
            query_camera=query_camera,
            # Pre-loaded map data
            reconstruction=colmap_model,
            db_names=db_names,
            db_descriptors=db_descriptors,
            features_ref=map_features_path,
            # Pre-loaded models
            feature_model=superpoint_model,
            retrieval_model=megaloc_model,
            matcher_model=matcher_model,
            # Configs
            feature_conf=feature_conf,
            retrieval_conf=retrieval_conf,
            # Options
            num_matched=30,
            ref_cache=ref_cache,
            covisibility_clustering=True,
            ransac_thresh=12.0,
        )
        
        img_total = time.time() - img_total_start
        per_image_timings[img_name] = img_total
        
        print(f"    TOTAL: {img_total:.2f}s")
        
        # Store pose for visualization
        if pnp_ret is not None:
            cam_from_world = pnp_ret["cam_from_world"]
            R = cam_from_world.rotation.matrix()
            t = cam_from_world.translation
            center = -R.T @ t
            all_poses[img_name] = {
                "center": center,
                "R": R.T,
                "num_inliers": pnp_ret.get("num_inliers", 0),
            }
            print(f"    Position: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
                  f" | Inliers: {pnp_ret.get('num_inliers', 0)}")
    
    timings['total'] = time.time() - total_start
    
    # ============ TIMING SUMMARY ============
    print("\n" + "="*50)
    print("TIMING SUMMARY (Fully In-Memory Pipeline)")
    print("="*50)
    print(f"  Setup (one-time): {timings['setup']:.2f}s")
    print(f"  ---------------------------------")
    
    for name, t in per_image_timings.items():
        print(f"    {name}: {t:.2f}s")
    
    n_images = len(per_image_timings)
    avg_total = sum(per_image_timings.values()) / n_images
    print(f"  ---------------------------------")
    print(f"  AVERAGE: {avg_total:.2f}s/image")
    print(f"  Cache stats: {ref_cache.stats()}")
    print(f"  Total pipeline: {timings['total']:.2f}s")
    
    # ============ RESULTS ============
    print("\n" + "="*50)
    print("LOCALIZATION RESULTS")
    print("="*50)
    
    for idx, (name, pose) in enumerate(sorted(all_poses.items())):
        c = pose["center"]
        print(f"  [{idx+1}] {name}: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    
    # ============ CREATE VISUALIZATION ============
    print("\n" + "="*50)
    print("Creating visualization...")
    print("="*50)
    
    viz_path = query_outputs_dir / "visualization.html"
    fig = visualize_localization(
        model=colmap_model,
        query_poses=all_poses,
        output_path=viz_path,
        title=f"Localization: {map_name}"
    )
    show_figure(fig)
    
    print(f"\nVisualization saved to: {viz_path}")
    
    return all_poses


def main():
    """Main entry point for the script."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Localize query images in a pre-built map"
    )
    
    # Add map name argument (required)
    parser.add_argument(
        "map_name",
        type=str,
        help="Name of the map to use (e.g., my_desk)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run inference
    run_inference(args.map_name)


# Entry point when script is run directly
if __name__ == "__main__":
    main()
