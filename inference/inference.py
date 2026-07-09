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
    
    # Check if map exists
    if not sfm_dir.exists():
        print(f"Error: Map not found at {sfm_dir}")
        print(f"Run 'python build_map.py {map_name}.mp4' first.")
        sys.exit(1)
    
    # Check if queries directory exists
    if not queries_dir.exists():
        print(f"Error: Queries directory not found at {queries_dir}")
        print("Create the directory and add query images.")
        sys.exit(1)
    
    # Get list of query images
    query_images = get_query_images(queries_dir)
    
    # Check if there are any query images
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
    
    # ============ LOAD MAP ============
    print(f"\n[1/5] Loading map from {sfm_dir}...")
    step_start = time.time()
    
    # Load COLMAP reconstruction
    model = pycolmap.Reconstruction(sfm_dir)
    
    timings['load_map'] = time.time() - step_start
    
    # Print map statistics
    print(f"  Registered images: {model.num_reg_images()}")
    print(f"  3D points: {model.num_points3D()}")
    print(f"  Time: {timings['load_map']:.2f}s")
    
    # ============ EXTRACT QUERY FEATURES ============
    print("\n[2/5] Extracting features from query images...")
    step_start = time.time()
    
    # Use same feature configuration as build_map.py
    # Must match exactly for feature matching to work
    feature_conf = {
        "output": "feats-superpoint-n8192-r1600",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 8192,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1600,
        },
    }
    
    # Extract features from all query images
    query_features = extract_features.main(
        conf=feature_conf,               # SuperPoint configuration
        image_dir=queries_dir,           # Query images directory
        export_dir=query_outputs_dir     # Where to save features
    )
    
    timings['extract_features'] = time.time() - step_start
    print(f"  Time: {timings['extract_features']:.2f}s ({timings['extract_features']/len(query_images):.2f}s/image)")
    
    # ============ EXTRACT RETRIEVAL DESCRIPTORS ============
    print("\n[3/5] Computing image retrieval descriptors...")
    step_start = time.time()
    
    # Use MegaLoc for global image descriptors (must match build_map.py)
    retrieval_conf = extract_features.confs["megaloc"]
    
    # Check if map retrieval features exist
    map_retrieval_path = map_outputs_dir / f"{retrieval_conf['output']}.h5"
    
    # Extract retrieval features for map images (if not already done)
    if not map_retrieval_path.exists():
        print("  Extracting retrieval features for map images...")
        map_retrieval = extract_features.main(
            conf=retrieval_conf,
            image_dir=frames_dir,
            export_dir=map_outputs_dir
        )
    else:
        # Use existing features
        map_retrieval = map_retrieval_path
    
    # Extract retrieval features for query images
    print("  Extracting retrieval features for query images...")
    query_retrieval = extract_features.main(
        conf=retrieval_conf,             # MegaLoc configuration
        image_dir=queries_dir,           # Query images directory
        export_dir=query_outputs_dir     # Where to save features
    )
    
    timings['retrieval'] = time.time() - step_start
    print(f"  Time: {timings['retrieval']:.2f}s")
    
    # ============ FIND MATCHING IMAGE PAIRS ============
    print("\n[4/5] Finding similar map images for each query...")
    step_start = time.time()
    
    # Path for localization pairs file
    pairs_loc_path = query_outputs_dir / "pairs-loc.txt"
    
    # Find top-N most similar map images for each query
    pairs_from_retrieval.main(
        descriptors=query_retrieval,         # Query global descriptors
        output=pairs_loc_path,               # Where to save pairs
        num_matched=30,                      # Number of similar images
        db_descriptors=map_retrieval         # Map global descriptors
    )
    
    timings['pair_finding'] = time.time() - step_start
    print(f"  Time: {timings['pair_finding']:.2f}s")
    
    # ============ MATCH FEATURES AND LOCALIZE (PER-IMAGE) ============
    print("\n[5/5] Matching features and localizing queries...")
    
    # Use SuperGlue-fast matcher (5 iterations instead of 50)
    matcher_conf = match_features.confs["superglue-fast"]
    
    # Path to map features
    map_features_path = map_outputs_dir / f"{feature_conf['output']}.h5"
    
    # Path for localization results
    results_path = query_outputs_dir / "localization_results.txt"
    
    # Parse retrieval pairs to get per-query pairs
    query_to_db_images = {}
    with open(pairs_loc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                q_name, db_name = parts
                if q_name not in query_to_db_images:
                    query_to_db_images[q_name] = []
                query_to_db_images[q_name].append(db_name)
    
    # Store per-image timings
    per_image_timings = {}
    
    # Process each query image individually for accurate timing
    all_poses = {}
    all_logs = {}
    
    for img_idx, img_path in enumerate(query_images):
        img_name = img_path.name
        img_start = time.time()
        
        print(f"\n  Processing [{img_idx + 1}/{len(query_images)}] {img_name}...")
        
        # Create single-image pairs file
        single_pairs_path = query_outputs_dir / f"pairs-{img_name}.txt"
        with open(single_pairs_path, 'w') as f:
            for db_name in query_to_db_images.get(img_name, []):
                f.write(f"{img_name} {db_name}\n")
        
        # Match features for this query
        single_matches = match_features.main(
            conf=matcher_conf,
            pairs=single_pairs_path,
            features=feature_conf["output"],
            export_dir=query_outputs_dir,
            features_ref=map_features_path,
            overwrite=True
        )
        
        # Create single-query intrinsics file
        single_query_path = query_outputs_dir / f"query-{img_name}.txt"
        img_cv = cv2.imread(str(img_path))
        h, w = img_cv.shape[:2]
        focal = 0.7 * max(w, h)
        cx, cy = w / 2, h / 2
        with open(single_query_path, 'w') as f:
            f.write(f"{img_name} SIMPLE_PINHOLE {w} {h} {focal:.2f} {cx:.2f} {cy:.2f}\n")
        
        # Localize this single query
        single_results_path = query_outputs_dir / f"results-{img_name}.txt"
        localize_sfm.main(
            reference_sfm=sfm_dir,
            queries=single_query_path,
            retrieval=single_pairs_path,
            features=query_features,
            matches=single_matches,
            results=single_results_path
        )
        
        img_time = time.time() - img_start
        per_image_timings[img_name] = img_time
        print(f"    Time: {img_time:.2f}s")
        
        # Parse this image's result
        single_poses = parse_localization_results(single_results_path)
        all_poses.update(single_poses)
        
        # Load logs if available
        single_logs_path = str(single_results_path) + "_logs.pkl"
        if Path(single_logs_path).exists():
            with open(single_logs_path, 'rb') as f:
                logs_data = pickle.load(f)
                all_logs.update(logs_data.get('loc', {}))
    
    # Write combined results
    with open(results_path, 'w') as f:
        for name, pose in all_poses.items():
            qvec = pose['qvec']
            tvec = pose['tvec']
            f.write(f"{name} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]}\n")
    
    total_time = time.time() - total_start
    timings['total'] = total_time
    
    # ============ PARSE AND DISPLAY RESULTS ============
    print("\n" + "="*50)
    print("LOCALIZATION RESULTS")
    print("="*50)
    
    # Use already parsed poses and logs
    query_poses = all_poses
    detailed_logs = all_logs
    
    # Sort and display results
    sorted_names = sorted(query_poses.keys())
    
    # Track successfully localized queries
    localized_count = 0
    
    for idx, name in enumerate(sorted_names):
        # Get pose
        pose = query_poses[name]
        
        # Get camera center
        c = pose["center"]
        
        # Get inference time for this image
        img_time = per_image_timings.get(name, 0)
        
        # Print position with timing
        print(f"\n  [{idx + 1}] {name}  ({img_time:.2f}s)")
        print(f"      Position: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
        
        # Print detailed info if available
        if name in detailed_logs:
            log = detailed_logs[name]
            
            # Get matched database images
            db_ids = log.get('db', [])
            if db_ids:
                # Get image names from model
                matched_names = []
                for db_id in db_ids[:5]:  # Show top 5
                    if db_id in model.images:
                        matched_names.append(model.images[db_id].name)
                print(f"      Matched with: {', '.join(matched_names)}")
                if len(db_ids) > 5:
                    print(f"                    ... and {len(db_ids) - 5} more")
            
            # Get PnP result details
            pnp_ret = log.get('PnP_ret', {})
            if pnp_ret:
                num_inliers = pnp_ret.get('num_inliers', 'N/A')
                print(f"      Inliers: {num_inliers}")
            
            # Get number of matches
            num_matches = log.get('num_matches', 0)
            if num_matches:
                print(f"      Total matches: {num_matches}")
        
        # Increment counter
        localized_count += 1
    
    # Check for failed localizations
    total_queries = len(query_images)
    if localized_count < total_queries:
        print(f"\n  Warning: {total_queries - localized_count} queries failed to localize")
    
    # ============ TIMING SUMMARY ============
    print("\n" + "="*50)
    print("TIMING SUMMARY")
    print("="*50)
    print(f"  Setup (load map, extract features, retrieval, pairs):")
    print(f"    Load map:         {timings['load_map']:.2f}s")
    print(f"    Extract features: {timings['extract_features']:.2f}s")
    print(f"    Retrieval:        {timings['retrieval']:.2f}s")
    print(f"    Pair finding:     {timings['pair_finding']:.2f}s")
    print(f"  ---------------------------------")
    print(f"  Per-image inference (match + localize):")
    total_inference = sum(per_image_timings.values())
    for name in sorted(per_image_timings.keys()):
        print(f"    {name}: {per_image_timings[name]:.2f}s")
    print(f"  ---------------------------------")
    print(f"  Total inference:    {total_inference:.2f}s ({total_inference/len(query_images):.2f}s/image)")
    print(f"  Total pipeline:     {timings['total']:.2f}s")
    
    # ============ CREATE VISUALIZATION ============
    print("\n" + "="*50)
    print("Creating visualization...")
    print("="*50)
    
    # Path for HTML visualization
    viz_path = query_outputs_dir / "visualization.html"
    
    # Create and save visualization using shared vis.py
    fig = visualize_localization(
        model=model,
        query_poses=query_poses,
        output_path=viz_path,
        title=f"Localization: {map_name}"
    )
    show_figure(fig)
    
    # Print output location
    print(f"\nVisualization saved to: {viz_path}")
    
    # Return results for programmatic use
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
