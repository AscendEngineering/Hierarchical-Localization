#!/usr/bin/env python3
"""
Localize query images against a pre-built map and visualize results.
Usage: ./inference.py my_desk

This script:
1. Loads a pre-built map from inference/maps/{map_name}/
2. Extracts features from all images in inference/queries/
3. Localizes each query image in the map
4. Visualizes the results with origin, map cameras, and query poses
"""

# Standard library imports
import argparse
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# NumPy for numerical operations
import numpy as np

# scipy for rotation/quaternion handling
from scipy.spatial.transform import Rotation

# pycolmap for COLMAP reconstruction handling
import pycolmap

# OpenCV for image reading
import cv2

# hloc modules for feature extraction, matching, and localization
from hloc import extract_features, match_features, localize_sfm
from hloc import pairs_from_retrieval
from hloc.utils.cache import LRUCache

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


def parse_localization_results(results_file: Path) -> dict:
    """
    Parse the localization results file.
    Format: image_name qw qx qy qz tx ty tz
    Returns dict mapping image name to camera center position.
    """
    
    # Dictionary to store parsed poses
    poses = {}
    
    # Open and read the results file
    with open(results_file) as f:
        # Process each line
        for line in f:
            # Split line into parts
            parts = line.strip().split()
            
            # Check if line has correct format (8 values)
            if len(parts) == 8:
                # Extract image name
                name = parts[0]
                
                # Extract quaternion components (rotation)
                qw, qx, qy, qz = map(float, parts[1:5])
                
                # Extract translation components
                tx, ty, tz = map(float, parts[5:8])
                
                # Convert quaternion to numpy array (COLMAP format: qw,qx,qy,qz)
                qvec = np.array([qw, qx, qy, qz])
                
                # Convert translation to numpy array
                tvec = np.array([tx, ty, tz])
                
                # Convert COLMAP quaternion [qw,qx,qy,qz] to scipy [qx,qy,qz,qw]
                quat_scipy = [qx, qy, qz, qw]
                
                # Convert quaternion to rotation matrix using scipy
                # This gives R_cam_from_world (transforms world points to camera frame)
                R_cam_from_world = Rotation.from_quat(quat_scipy).as_matrix()
                
                # Compute camera center in world coordinates
                # Camera center = -R^T * t
                center = -R_cam_from_world.T @ tvec
                
                # For frustum visualization, we need world_t_camera rotation
                # world_t_camera = cam_from_world.inverse(), so R_world_t_camera = R^T
                R_world_t_camera = R_cam_from_world.T
                
                # Store pose information
                poses[name] = {
                    "center": center,           # 3D position in world frame
                    "R": R_world_t_camera,      # world_t_camera rotation (for frustum viz)
                    "qvec": qvec,               # Original quaternion
                    "tvec": tvec                # Original translation
                }
    
    # Return all parsed poses
    return poses


def _extract_superpoint(conf, image_dir, export_dir, model):
    """Extract SuperPoint features (for parallel execution)."""
    t = time.time()
    result = extract_features.main(
        conf=conf,
        image_dir=image_dir,
        export_dir=export_dir,
        model=model
    )
    return result, time.time() - t


def _extract_megaloc(conf, image_dir, export_dir, model):
    """Extract MegaLoc global descriptors (for parallel execution)."""
    t = time.time()
    result = extract_features.main(
        conf=conf,
        image_dir=image_dir,
        export_dir=export_dir,
        model=model
    )
    return result, time.time() - t


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
    
    # Frames directory (images used to build map)
    frames_dir = map_dir / "frames"
    
    # Outputs directory (features, matches from build_map.py)
    map_outputs_dir = map_dir / "outputs"
    
    # Queries directory
    queries_dir = inference_dir / "queries"
    
    # Output directory for query processing
    query_outputs_dir = map_dir / "query_outputs"
    
    # Clean up previous inference outputs
    if query_outputs_dir.exists():
        print(f"Cleaning previous inference outputs in {query_outputs_dir}...")
        shutil.rmtree(query_outputs_dir)
    
    # Create query outputs directory
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
    
    timings['setup'] = time.time() - setup_start
    print(f"  Setup time: {timings['setup']:.2f}s")
    
    # ============ PROCESS EACH IMAGE (TRUE PER-IMAGE LATENCY) ============
    print(f"\n[INFERENCE] Processing {len(query_images)} queries (full pipeline each)...")
    
    per_image_timings = {}
    per_image_breakdown = {}
    all_poses = {}

    for img_idx, img_path in enumerate(query_images):
        img_name = img_path.name
        print(f"\n  [{img_idx + 1}/{len(query_images)}] {img_name}")
        
        img_total_start = time.time()
        breakdown = {}
        
        # Create temp directory for this single image
        single_query_dir = query_outputs_dir / f"query_{img_idx}"
        single_query_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy image to temp directory
        import shutil as sh
        single_img_path = single_query_dir / img_name
        sh.copy(img_path, single_img_path)
        
        # --- STEP 1 & 2: Extract features in PARALLEL ---
        # SuperPoint and MegaLoc are independent, so we run them concurrently
        t0_parallel = time.time()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            sp_future = executor.submit(
                _extract_superpoint, feature_conf, single_query_dir, single_query_dir, superpoint_model
            )
            ml_future = executor.submit(
                _extract_megaloc, retrieval_conf, single_query_dir, single_query_dir, megaloc_model
            )
            
            single_features, breakdown['superpoint'] = sp_future.result()
            single_retrieval, breakdown['megaloc'] = ml_future.result()
        
        breakdown['feature_extraction'] = time.time() - t0_parallel
        
        # --- STEP 3: Find similar map images ---
        t0 = time.time()
        pairs_path = single_query_dir / "pairs.txt"
        pairs_from_retrieval.main(
            descriptors=single_retrieval,       # Query global descriptors
            output=pairs_path,                  # Where to save pairs
            num_matched=30,                     # Number of similar images
            db_descriptors=map_retrieval_path   # Map global descriptors
        )
        breakdown['retrieval'] = time.time() - t0
        
        # --- STEP 4: Match features ---
        t0 = time.time()
        matches_path = match_features.main(
            conf=matcher_conf,                  # SuperPoint+LightGlue ONNX TRT configuration
            pairs=pairs_path,                   # Path to pairs file (query, db_image)
            features=feature_conf["output"],    # Query features filename
            export_dir=single_query_dir,        # Where to save matches
            features_ref=map_features_path,     # Path to map features HDF5
            overwrite=True,                     # Overwrite existing matches
            model=matcher_model,                # Pre-loaded TRT model
            ref_cache=ref_cache,                # LRU cache for map features
        )
        breakdown['matching'] = time.time() - t0
        
        # --- STEP 5: Localize ---
        t0 = time.time()
        
        # Create intrinsics file
        img_cv = cv2.imread(str(img_path))
        h, w = img_cv.shape[:2]
        focal = 0.7 * max(w, h)
        cx, cy = w / 2, h / 2
        intrinsics_path = single_query_dir / "intrinsics.txt"
        with open(intrinsics_path, 'w') as f:
            f.write(f"{img_name} SIMPLE_PINHOLE {w} {h} {focal:.2f} {cx:.2f} {cy:.2f}\n")
        
        results_path = single_query_dir / "results.txt"
        localize_sfm.main(
            reference_sfm=sfm_dir,       # Path to the reference SfM model
            queries=intrinsics_path,     # Path to query intrinsics file
            retrieval=pairs_path,        # Path to pairs (query, db_image) to consider
            features=single_features,    # HDF5 file with query features
            matches=matches_path,        # HDF5 file with query-map matches
            results=results_path,        # Where to save localization results
            covisibility_clustering=True,     # Cluster map images to reduce PnP candidates
        )
        breakdown['localization'] = time.time() - t0
        
        # Total time for this image
        img_total = time.time() - img_total_start
        per_image_timings[img_name] = img_total
        per_image_breakdown[img_name] = breakdown
        
        parallel_savings = breakdown['superpoint'] + breakdown['megaloc'] - breakdown['feature_extraction']
        print(f"    Features (parallel): {breakdown['feature_extraction']:.2f}s "
              f"[SP:{breakdown['superpoint']:.2f}s + ML:{breakdown['megaloc']:.2f}s, saved {parallel_savings:.2f}s]")
        print(f"    Retrieval: {breakdown['retrieval']:.2f}s | Match: {breakdown['matching']:.2f}s | "
              f"Localize: {breakdown['localization']:.2f}s")
        print(f"    TOTAL: {img_total:.2f}s")
        
        # Parse results
        single_poses = parse_localization_results(results_path)
        all_poses.update(single_poses)
    
    timings['total'] = time.time() - total_start
    
    # ============ TIMING SUMMARY ============
    print("\n" + "="*50)
    print("TIMING SUMMARY (True Single-Image Latency)")
    print("="*50)
    print(f"  Setup (one-time): {timings['setup']:.2f}s")
    print(f"  ---------------------------------")
    print(f"  Per-image FULL pipeline:")
    
    # Calculate averages
    avg_breakdown = {k: 0 for k in per_image_breakdown[list(per_image_breakdown.keys())[0]]}
    for name, breakdown in per_image_breakdown.items():
        print(f"    {name}:")
        print(f"      Feature extraction (parallel): {breakdown['feature_extraction']:.2f}s")
        print(f"        - SuperPoint: {breakdown['superpoint']:.2f}s")
        print(f"        - MegaLoc:    {breakdown['megaloc']:.2f}s")
        parallel_savings = breakdown['superpoint'] + breakdown['megaloc'] - breakdown['feature_extraction']
        print(f"        - Saved:      {parallel_savings:.2f}s")
        print(f"      Retrieval:  {breakdown['retrieval']:.2f}s")
        print(f"      Matching:   {breakdown['matching']:.2f}s")
        print(f"      Localize:   {breakdown['localization']:.2f}s")
        print(f"      TOTAL:      {per_image_timings[name]:.2f}s")
        for k in avg_breakdown:
            avg_breakdown[k] += breakdown[k]
    
    n_images = len(per_image_breakdown)
    print(f"  ---------------------------------")
    print(f"  AVERAGE per image:")
    print(f"    Feature extraction (parallel): {avg_breakdown['feature_extraction']/n_images:.2f}s")
    print(f"      - SuperPoint: {avg_breakdown['superpoint']/n_images:.2f}s")
    print(f"      - MegaLoc:    {avg_breakdown['megaloc']/n_images:.2f}s")
    avg_savings = (avg_breakdown['superpoint'] + avg_breakdown['megaloc'] - avg_breakdown['feature_extraction']) / n_images
    print(f"      - Saved:      {avg_savings:.2f}s")
    print(f"    Retrieval:  {avg_breakdown['retrieval']/n_images:.2f}s")
    print(f"    Matching:   {avg_breakdown['matching']/n_images:.2f}s")
    print(f"    Localize:   {avg_breakdown['localization']/n_images:.2f}s")
    avg_total = sum(per_image_timings.values()) / n_images
    print(f"    TOTAL:      {avg_total:.2f}s/image")
    print(f"  ---------------------------------")
    print(f"  Cache stats: {ref_cache.stats()}")
    print(f"  Total pipeline: {timings['total']:.2f}s")
    
    # ============ RESULTS ============
    print("\n" + "="*50)
    print("LOCALIZATION RESULTS")
    print("="*50)
    
    query_poses = all_poses
    for idx, (name, pose) in enumerate(sorted(query_poses.items())):
        c = pose["center"]
        print(f"\n  [{idx+1}] {name}")
        print(f"      Position: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    
    # ============ CREATE VISUALIZATION ============
    print("\n" + "="*50)
    print("Creating visualization...")
    print("="*50)
    
    viz_path = query_outputs_dir / "visualization.html"
    fig = visualize_localization(
        model=colmap_model,
        query_poses=query_poses,
        output_path=viz_path,
        title=f"Localization: {map_name}"
    )
    show_figure(fig)
    
    print(f"\nVisualization saved to: {viz_path}")
    
    return query_poses


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
