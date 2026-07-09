# Visual Localization Pipeline

Build 3D maps from video and localize query images against them.

## Directory Structure

```
datasets/
└── {scene_name}.mp4      # Input: video files for mapping

inference/
├── build_map.py          # Build map from video
├── inference.py          # Localize queries against map
├── vis.py                # Visualization utilities
├── maps/                 # Output: built maps stored here
│   └── {map_name}/       # Each map in its own folder
└── queries/              # Input: query images to localize
```

## Setup

### 1. Build Docker Image

```bash
# From the root of this project
docker build -t hloc .
```

### 2. Run Container

```bash
docker run --gpus all -it --rm -p 8888:8888 \
  --shm-size=8g \
  -v $(pwd):/app \
  hloc:latest
```

## Usage

### Build a Map

Place your video file in `datasets/`, then run:

```bash
./inference/build_map.py my_office.mp4
```

The map will be saved to `inference/maps/{video_name}/` (e.g., `inference/maps/my_desk/` for `my_desk.mp4`).

**What it does:**
1. Extracts frames from video 
2. Detects SuperPoint keypoints
3. Computes MegaLoc global descriptors
4. Matches frames using LightGlue
5. Runs COLMAP reconstruction
6. Saves 3D visualization to `inference/maps/{map_name}/map_visualization.html`

### Localize Query Images

1. Place query images in `inference/queries/`
2. Run inference against a built map:

```bash
./inference/inference.py my_desk
```

**What it does:**
1. Loads the map from `inference/maps/my_desk/`
2. Extracts features from all images in `inference/queries/`
3. Localizes each query via PnP
4. Saves visualization to `inference/maps/{map_name}/query_outputs/visualization.html`

### Viewing Visualizations

To view the 3D visualizations, copy the path to the `.html` file and open it in your browser:

## Example Workflow

```bash
# Inside Docker container (working directory: /app)

# Build map from video
./inference/build_map.py my_office.mp4

# Run inference
./inference/inference.py my_office
```
