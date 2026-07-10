#!/bin/bash
# Dense reconstruction pipeline using COLMAP (undistort) + OpenMVS
# Usage: ./build_dense_map.sh <map_name> [--skip-undistort] [--skip-mesh]

set -e

MAP_NAME="${1:-my_office}"
SKIP_UNDISTORT=false
SKIP_MESH=false

# Parse optional flags
for arg in "$@"; do
    case $arg in
        --skip-undistort) SKIP_UNDISTORT=true ;;
        --skip-mesh) SKIP_MESH=true ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_DIR="$SCRIPT_DIR/maps/$MAP_NAME"
DENSE_DIR="$MAP_DIR/dense"

echo "=== Dense Reconstruction Pipeline ==="
echo "Map: $MAP_NAME"
echo "Map directory: $MAP_DIR"
echo ""

# Check if sparse reconstruction exists
if [ ! -f "$MAP_DIR/sfm/cameras.bin" ]; then
    echo "ERROR: Sparse reconstruction not found at $MAP_DIR/sfm/"
    echo "Run build_map.py first to create the sparse reconstruction."
    exit 1
fi

# Step 1: Undistort images using COLMAP
if [ "$SKIP_UNDISTORT" = false ]; then
    echo "=== Step 1: Undistorting images (COLMAP) ==="
    
    # Remove existing dense dir (may need sudo if created by docker as root)
    if [ -d "$DENSE_DIR" ]; then
        rm -rf "$DENSE_DIR" 2>/dev/null || sudo rm -rf "$DENSE_DIR"
    fi
    mkdir -p "$DENSE_DIR"
    
    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        -v "$MAP_DIR:/working" \
        colmap/colmap:latest colmap image_undistorter \
        --image_path /working/frames \
        --input_path /working/sfm \
        --output_path /working/dense \
        --output_type COLMAP
    
    # Convert binary to text format (OpenMVS expects text files)
    echo "Converting COLMAP binary to text format..."
    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        -v "$DENSE_DIR:/working" \
        colmap/colmap:latest colmap model_converter \
        --input_path /working/sparse \
        --output_path /working/sparse \
        --output_type TXT
    
    echo "Undistortion complete."
else
    echo "=== Step 1: Skipped (--skip-undistort) ==="
fi

# Step 2: Convert to OpenMVS format
echo ""
echo "=== Step 2: Converting to OpenMVS format ==="
docker run --gpus all --rm \
    --user "$(id -u):$(id -g)" \
    -v "$DENSE_DIR:/working" \
    -w /working \
    openmvs/openmvs-ubuntu:latest \
    InterfaceCOLMAP -i . -o scene.mvs --image-folder images

# Step 3: Dense point cloud reconstruction
echo ""
echo "=== Step 3: Dense point cloud reconstruction ==="
docker run --gpus all --rm \
    --user "$(id -u):$(id -g)" \
    -v "$DENSE_DIR:/working" \
    -w /working \
    openmvs/openmvs-ubuntu:latest \
    DensifyPointCloud scene.mvs \
    --resolution-level 1 \
    --min-resolution 640 \
    --number-views 5 \
    --number-views-fuse 3

if [ "$SKIP_MESH" = false ]; then
    # Step 4: Mesh reconstruction
    echo ""
    echo "=== Step 4: Mesh reconstruction ==="
    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        -v "$DENSE_DIR:/working" \
        -w /working \
        openmvs/openmvs-ubuntu:latest \
        ReconstructMesh scene_dense.mvs \
        --smooth 2 \
        --decimate 0.5

    # Step 5: Mesh refinement (optional but improves quality)
    echo ""
    echo "=== Step 5: Mesh refinement ==="
    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        -v "$DENSE_DIR:/working" \
        -w /working \
        openmvs/openmvs-ubuntu:latest \
        RefineMesh scene_dense_mesh.mvs \
        --resolution-level 1 \
        --scales 2

    # Step 6: Texture mapping
    echo ""
    echo "=== Step 6: Texture mapping ==="
    docker run --gpus all --rm \
        --user "$(id -u):$(id -g)" \
        -v "$DENSE_DIR:/working" \
        -w /working \
        openmvs/openmvs-ubuntu:latest \
        TextureMesh scene_dense_mesh_refine.mvs \
        --export-type obj
else
    echo ""
    echo "=== Steps 4-6: Skipped (--skip-mesh) ==="
fi

echo ""
echo "=== Done! ==="
echo "Output files in: $DENSE_DIR"
echo ""
ls -lh "$DENSE_DIR"/*.ply "$DENSE_DIR"/*.mvs 2>/dev/null || true
ls -lh "$DENSE_DIR"/*.obj 2>/dev/null || true
