#!/usr/bin/env python3
"""
Build a 3D map from a video file using COLMAP and learned features.
Usage: python build_map.py my_desk.mp4
"""

# Standard library imports for file/system operations
import argparse
import shutil
from pathlib import Path

# OpenCV for video frame extraction
import cv2

# hloc modules for feature extraction, matching, and reconstruction
from hloc import extract_features, match_features, reconstruction
from hloc import pairs_from_retrieval
from hloc.utils.io import list_h5_names
import numpy as np


# Local visualization utilities
from vis import visualize_map, show_figure


def create_sequential_pairs(features_path: Path, output_path: Path, window: int = 5):
    """
    Create sequential image pairs for video frames.
    Each frame is matched with +-window neighboring frames.
    
    For window=5, frame i matches with frames [i-5, i-4, ..., i-1, i+1, ..., i+4, i+5]
    This is much faster than exhaustive matching and better suited for video.
    """
    # Get sorted list of image names from features file
    image_names = sorted(list_h5_names(features_path))
    n_images = len(image_names)
    
    pairs = []
    for i in range(n_images):
        # Match with neighbors within window
        for j in range(i + 1, min(i + window + 1, n_images)):
            pairs.append((image_names[i], image_names[j]))
    
    # Write pairs to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for img1, img2 in pairs:
            f.write(f"{img1} {img2}\n")
    
    print(f"  Created {len(pairs)} sequential pairs (window=±{window})")
    return output_path


def extract_frames_from_video(video_path: Path, output_dir: Path) -> int:
    """
    Extract frames from video at a rate determined by video length.
    Shorter videos = more frames per second, longer videos = fewer frames per second.

    TODO:
        - Adapt frame selection based on how it contributes to reconstruction quality (e.g., more frames in dynamic scenes, fewer in static scenes)
    """
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    # Print video information
    print(f"Video: {video_path.name}")
    print(f"  Duration: {duration_sec:.1f} seconds")
    print(f"  FPS: {fps:.1f}")
    print(f"  Total frames: {total_frames}")
    
    # Determine target number of frames based on video length
    if duration_sec < 30:
        # Short video: aim for ~5 fps
        target_frames = int(duration_sec * 5)
    elif duration_sec < 120:
        # Medium video: aim for ~3 fps
        target_frames = int(duration_sec * 3)
    else:
        # Long video: aim for ~2 fps, cap at 400 frames
        target_frames = min(int(duration_sec * 2), 400)
    
    # Ensure we have at least 20 frames
    target_frames = max(target_frames, 20)
    
    # Calculate frame skip interval
    every_n_frames = max(1, total_frames // target_frames)
    
    print(f"  Extracting ~{target_frames} frames (every {every_n_frames} frames)")
    
    # Extract frames
    frame_idx = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % every_n_frames == 0:
            out_path = output_dir / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    
    print(f"  Saved {saved_count} frames to {output_dir}")
    return saved_count


def build_map(video_name: str):
    """
    Build a 3D map from a video 
    
    Steps:
    1. Extract frames from video
    2. Extract SuperPoint features from all frames (ONNX-accelerated)
    3. Create exhaustive image pairs (all-to-all matching)
    4. Match features using LightGlue (TensorRT-accelerated)
    5. Build COLMAP reconstruction
    """
    
    # Get the root directory (parent of inference/)
    root = Path(__file__).parent.parent
    
    # Construct path to the video file in datasets/
    video_path = root / "datasets" / video_name
    
    # If the video file does not exist, return
    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return None
    
    # Extract the video name without extension (e.g., "my_desk" from "my_desk.mp4")
    map_name = video_path.stem
    
    # Define output directories
    # maps_dir: where all maps are stored
    maps_dir = Path(__file__).parent / "maps"
    
    # map_dir: specific directory for this video's map
    map_dir = maps_dir / map_name
    
    # frames_dir: temporary directory for extracted frames
    frames_dir = map_dir / "frames"
    
    # sfm_dir: directory for COLMAP reconstruction output
    sfm_dir = map_dir / "sfm"
    
    # outputs_dir: directory for intermediate files (features, matches)
    outputs_dir = map_dir / "outputs"
    
    # Create the map directory
    map_dir.mkdir(parents=True, exist_ok=True)
    
    # If previous build exists, clean up (to avoid accumulating frames)
    if frames_dir.exists():
        print(f"Cleaning previous build in {map_dir}...")
        shutil.rmtree(frames_dir)
    if outputs_dir.exists():
        shutil.rmtree(outputs_dir)
    if sfm_dir.exists():
        shutil.rmtree(sfm_dir)
    
    # Print status
    print(f"\n{'='*50}")
    print(f"Building map: {map_name}")
    print(f"{'='*50}\n")
    
    # ============ STEP 1: EXTRACT FRAMES ============
    print("[1/6] Extracting frames from video...")
    
    # Extract frames from the video
    num_frames = extract_frames_from_video(video_path, frames_dir)
    
    # If we only got 1 frame, something went wrong (video too short or extraction failed)
    if num_frames == 1:
        print(f"Error: Only 1 frame extracted from video. Check the video file and extraction process.")
        return None
    
    # ============ STEP 2: CONFIGURE FEATURE EXTRACTION ============
    print("\n[2/6] Extracting SuperPoint features (CUDA-accelerated)...")
    
    # SuperPoint ONNX configuration:
    # - Grayscale images (SuperPoint requirement)
    # - 2048 keypoints with force_num_keypoints (topk, like fabio-sim)
    # - This ensures EXACTLY 2048 keypoints per image for batched LightGlue
    # - 1024px max resolution
    feature_conf = {
        "output": "feats-superpoint-n2048-r1024",
        "model": {
            "name": "superpoint_onnx",
            "max_num_keypoints": 2048,
            "force_num_keypoints": True,  # Always return exactly max_num_keypoints (topk)
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 1024,
        },
    }
    
    # Extract local features from all frames
    # Returns path to the HDF5 file containing features
    local_features_path = extract_features.main(
        conf=feature_conf,           # ALIKED configuration
        image_dir=frames_dir,        # Directory containing images
        export_dir=outputs_dir       # Where to save features
    )
    
    # ============ STEP 3: CREATE IMAGE PAIRS ============
    print("\n[3/6] Creating image pairs (sequential + retrieval for loop closure)...")
    
    # Define path for the pairs file
    sfm_pairs_path = outputs_dir / "pairs-sfm.txt"
    
    # Count images for adaptive window sizing
    num_images = len(list(frames_dir.glob("*.jpg")))
    
    # APPROACH: Sequential + Retrieval for loop closure
    print("  Using sequential pairing + retrieval for loop closure")
    
    # Sequential window size, larger for more images
    seq_window = 20 if num_images > 50 else 10
    
    # Step 3a: Extract global descriptors (CUDA-accelerated MegaLoc)
    print("  Extracting global descriptors (MegaLoc ONNX)...")
    global_conf = extract_features.confs["megaloc_onnx"]
    global_features_path = extract_features.main(
        conf=global_conf,
        image_dir=frames_dir,
        export_dir=outputs_dir
    )
    
    # Step 3b: Create global pairs 
    global_pairs_path = outputs_dir / "pairs-global.txt"
    pairs_from_retrieval.main(
        descriptors=global_features_path,
        output=global_pairs_path,
        num_matched=20  # Top N similar images per frame
    )
    
    # Step 3c: Create sequential pairs
    sequential_pairs_path = outputs_dir / "pairs-sequential.txt"
    create_sequential_pairs(
        features_path=local_features_path,
        output_path=sequential_pairs_path,
        window=seq_window
    )
    
    # Step 3d: Combine both pair files (remove duplicates)
    all_pairs = set()
    for pair_file in [sequential_pairs_path, global_pairs_path]:
        with open(pair_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    # Normalize pair order to avoid duplicates
                    pair = tuple(sorted(parts))
                    all_pairs.add(pair)
    
    # Write combined pairs to pairs-sfm.txt
    with open(sfm_pairs_path, 'w') as f:
        for img1, img2 in sorted(all_pairs):
            f.write(f"{img1} {img2}\n")
    
    print(f"  Combined {len(all_pairs)} unique pairs (sequential + retrieval)")
    
    # ============ STEP 4: MATCH FEATURES ============
    print("\n[4/6] Matching features with LightGlue (TensorRT-accelerated)...")
    
    # Use SuperPoint + LightGlue ONNX (TensorRT provider selected automatically)
    matcher_conf = match_features.confs["superpoint_onnx+lightglue_onnx"]
    
    # Match features between all image pairs
    # Returns path to the HDF5 file containing matches
    matches = match_features.main(
        conf=matcher_conf,                      # LightGlue configuration
        pairs=sfm_pairs_path,                   # Image pairs to match
        features=feature_conf["output"],        # Feature type name
        export_dir=outputs_dir                  # Where to save matches
    )
    
    # ============ STEP 5: BUILD RECONSTRUCTION ============
    print("\n[5/6] Building COLMAP reconstruction...")
    
    # Run incremental Structure-from-Motion using COLMAP
    # This triangulates 3D points and estimates camera poses
    model = reconstruction.main(
        sfm_dir=sfm_dir,             # Where to save reconstruction
        image_dir=frames_dir,        # Directory containing images
        pairs=sfm_pairs_path,            # Image pairs with matches
        features=local_features_path,  # Path to features file
        matches=matches              # Path to matches file
    )
    
    # If reconstruction failed, return
    if model is None:
        print("Error: Reconstruction failed!")
        return None
    
    # Print reconstruction statistics
    print(f"\n{'='*50}")
    print(f"Map built successfully!")
    print(f"  Images registered: {model.num_reg_images()}")
    print(f"  3D points: {model.num_points3D()}")
    print(f"  Map saved to: {sfm_dir}")
    print(f"{'='*50}\n")
    
    # ============ STEP 6: VISUALIZE MAP ============
    print("[6/6] Creating visualization...")
    
    # Create and save visualization
    viz_path = map_dir / "map_visualization.html"
    fig = visualize_map(model, viz_path, title=f"3D Map: {map_name}")
    show_figure(fig)
    
    print(f"\nVisualization saved to: {viz_path}")
    print(f"Open in browser: file://{viz_path.absolute()}")
    
    # Return the path to the map directory
    return map_dir


def main():
    """Main entry point for the script."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Build a 3D map from a video file"
    )
    
    # Add video argument (required)
    parser.add_argument(
        "video",
        type=str,
        help="Video filename in datasets/ (e.g., my_desk.mp4)"
    )
    
    # Parse command line arguments
    args = parser.parse_args()
    
    # Build the map from the specified video
    build_map(args.video)


if __name__ == "__main__":
    main()
