#!/usr/bin/env python3
"""
Visualization utilities for 3D map and query pose visualization.
Provides common functions for build_map.py and inference.py.

Uses hloc's viz_3d module for camera frustum rendering.
"""

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import plotly.graph_objects as go
import pycolmap

# Import hloc visualization utilities
from hloc.utils import viz_3d


def init_figure(title: str = "3D Visualization", height: int = 900) -> go.Figure:
    """
    Initialize a Plotly 3D figure with good navigation defaults.
    Uses turntable rotation for more intuitive interaction.
    """
    fig = go.Figure()
    
    # Axis settings - minimal but visible
    axes = dict(
        visible=True,
        showbackground=True,
        backgroundcolor="rgb(240, 240, 240)",
        showgrid=True,
        gridcolor="rgb(200, 200, 200)",
        showline=True,
        linecolor="rgb(150, 150, 150)",
        showticklabels=True,
        autorange=True,
        zeroline=True,
        zerolinecolor="rgb(100, 100, 100)",
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20)),
        height=height,
        width=1400,
        # Camera settings for intuitive navigation
        scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.0),  # Initial view angle
            up=dict(x=0, y=0, z=1),          # Z-up convention
            projection=dict(type="perspective"),  # Perspective for depth
        ),
        scene=dict(
            xaxis=dict(**axes, title="X"),
            yaxis=dict(**axes, title="Y"),
            zaxis=dict(**axes, title="Z"),
            aspectmode="data",  # Equal aspect ratio
            dragmode="orbit",  # Free rotation 
        ),
        margin=dict(l=0, r=0, b=0, t=40, pad=0),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        ),
    )
    
    return fig


def plot_origin(
    fig: go.Figure,
    scale: float = 0.3,
    show_labels: bool = True
):
    """
    Plot origin axes (X=red, Y=green, Z=blue) at a small scale.
    """
    # X-axis (red)
    fig.add_trace(go.Scatter3d(
        x=[0, scale], y=[0, 0], z=[0, 0],
        mode='lines+text' if show_labels else 'lines',
        line=dict(color='red', width=4),
        text=['', 'X'] if show_labels else None,
        textposition='middle right',
        textfont=dict(size=10, color='red'),
        name='X-axis',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Y-axis (green)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, scale], z=[0, 0],
        mode='lines+text' if show_labels else 'lines',
        line=dict(color='green', width=4),
        text=['', 'Y'] if show_labels else None,
        textposition='middle right',
        textfont=dict(size=10, color='green'),
        name='Y-axis',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Z-axis (blue)
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, scale],
        mode='lines+text' if show_labels else 'lines',
        line=dict(color='blue', width=4),
        text=['', 'Z'] if show_labels else None,
        textposition='top center',
        textfont=dict(size=10, color='blue'),
        name='Z-axis',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Origin point
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=5, color='black', symbol='diamond'),
        name='Origin',
        showlegend=True,
        hovertext='Origin (0,0,0)',
        hoverinfo='text'
    ))


def plot_reconstruction(
    fig: go.Figure,
    model: pycolmap.Reconstruction,
    camera_color: str = "rgb(0, 100, 255)",
    point_color: Optional[str] = None,
    camera_size: float = 1.0,
    show_points: bool = True,
    show_cameras: bool = True,
    use_point_colors: bool = True,
    max_reproj_error: float = 6.0,
    min_track_length: int = 2,
    name: str = "Map",
    highlight_start_end: bool = True
):
    """
    Plot a COLMAP reconstruction with point cloud and camera frustums.
    Uses hloc's viz_3d for proper frustum rendering.
    
    If highlight_start_end=True, first camera is green, last is purple.
    """
    # Filter outliers for points
    bbs = model.compute_bounding_box(0.001, 0.999)
    p3Ds = [
        p3D
        for _, p3D in model.points3D.items()
        if (
            bbs.contains_point(p3D.xyz)
            and p3D.error <= max_reproj_error
            and p3D.track.length() >= min_track_length
        )
    ]
    
    # Plot points
    if show_points and p3Ds:
        xyzs = np.array([p3D.xyz for p3D in p3Ds])
        if use_point_colors:
            pcolor = [p3D.color for p3D in p3Ds]
        else:
            pcolor = camera_color
        viz_3d.plot_points(fig, xyzs, color=pcolor, ps=1, name=name)
    
    # Plot cameras
    if show_cameras:
        # Sort images by name to get temporal order (frame_0000, frame_0001, ...)
        sorted_images = sorted(model.images.items(), key=lambda x: model.images[x[0]].name)
        
        if len(sorted_images) == 0:
            return
        
        first_id = sorted_images[0][0]
        last_id = sorted_images[-1][0]
        
        # Plot each camera
        for image_id, image in sorted_images:
            camera = model.cameras[image.camera_id]
            
            # Determine color based on position
            if highlight_start_end and image_id == first_id:
                color = "rgb(0, 200, 0)"  # Green for start
                cam_name = "Start"
            elif highlight_start_end and image_id == last_id:
                color = "rgb(150, 0, 200)"  # Purple for end
                cam_name = "End"
            else:
                color = camera_color
                cam_name = None
            
            viz_3d.plot_image_colmap(
                fig, 
                image, 
                camera, 
                name=cam_name,
                color=color, 
                legendgroup=name if cam_name is None else cam_name,
                size=camera_size
            )


def plot_query_poses(
    fig: go.Figure,
    query_poses: Dict[str, dict],
    color: str = "red",
    size: float = 0.5
):
    """
    Plot query camera poses as frustums with labels.
    
    query_poses: dict mapping query name to pose dict with 'center' and 'R' keys
    """
    if not query_poses:
        return
    
    # Sort queries by name for consistent img1, img2, ... ordering
    sorted_names = sorted(query_poses.keys())
    
    # Plot each query pose
    for idx, name in enumerate(sorted_names):
        pose = query_poses[name]
        center = pose['center']
        R = pose['R']
        
        # Create a simple intrinsics matrix for frustum visualization
        # (approximate, just for visualization purposes)
        K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ])
        
        label = f"img{idx + 1}"
        
        # Use hloc's plot_camera function
        viz_3d.plot_camera(
            fig=fig,
            R=R,
            t=center,
            K=K,
            color=color,
            name=label,
            legendgroup="Query Poses",
            size=size,
            text=f"{label}: {name}"
        )
    
    # Add query markers with labels
    centers = np.array([query_poses[n]['center'] for n in sorted_names])
    labels = [f"img{i+1}" for i in range(len(sorted_names))]
    
    fig.add_trace(go.Scatter3d(
        x=centers[:, 0],
        y=centers[:, 1],
        z=centers[:, 2],
        mode='markers+text',
        marker=dict(size=3, color=color, symbol='diamond'),
        text=labels,
        textposition='top center',
        textfont=dict(size=12, color=color),
        name='Query Poses',
        showlegend=True,
        hovertext=[f"{l}: {n}" for l, n in zip(labels, sorted_names)],
        hoverinfo='text'
    ))


def compute_scene_scale(model: pycolmap.Reconstruction) -> float:
    """
    Compute an appropriate scale factor based on the reconstruction size.
    """
    # Get camera centers
    centers = []
    for image_id, image in model.images.items():
        world_t_camera = image.cam_from_world().inverse()
        centers.append(world_t_camera.translation)
    
    if len(centers) > 1:
        centers = np.array(centers)
        return float(np.std(centers))
    
    # Fallback to point cloud
    if model.num_points3D() > 0:
        pts = np.array([p.xyz for p in model.points3D.values()])
        return float(np.std(pts))
    
    return 1.0


def visualize_map(
    model: pycolmap.Reconstruction,
    output_path: Path,
    title: str = "3D Map",
    camera_size: float = 1.0,
    show_origin: bool = True
) -> go.Figure:
    """
    Create and save a visualization of a 3D map.
    
    Args:
        model: pycolmap Reconstruction object
        output_path: Path to save HTML file
        title: Title for the visualization
        camera_size: Size multiplier for camera frustums
        show_origin: Whether to show origin axes
    
    Returns:
        Plotly figure object
    """
    # Initialize figure
    fig = init_figure(title=title)
    
    # Compute scene scale for origin sizing
    scene_scale = compute_scene_scale(model)
    
    # Plot the reconstruction (points + cameras with frustums)
    plot_reconstruction(
        fig=fig,
        model=model,
        camera_color="rgb(0, 100, 255)",
        camera_size=camera_size,
        show_points=True,
        show_cameras=True,
        use_point_colors=True
    )
    
    # Plot origin (smaller, proportional to scene)
    if show_origin:
        origin_scale = scene_scale * 0.15  # Small relative to scene
        plot_origin(fig, scale=origin_scale)
    
    # Save to HTML
    fig.write_html(str(output_path))
    
    return fig


def visualize_localization(
    model: pycolmap.Reconstruction,
    query_poses: Dict[str, dict],
    output_path: Path,
    title: str = "Localization Results",
    camera_size: float = 0.8,
    query_size: float = 1.2,
    show_origin: bool = True
) -> go.Figure:
    """
    Create and save a visualization of localization results.
    
    Args:
        model: pycolmap Reconstruction object (the map)
        query_poses: Dict mapping query names to pose dicts with 'center' and 'R'
        output_path: Path to save HTML file
        title: Title for the visualization
        camera_size: Size multiplier for map camera frustums
        query_size: Size multiplier for query camera frustums
        show_origin: Whether to show origin axes
    
    Returns:
        Plotly figure object
    """
    # Initialize figure
    fig = init_figure(title=title)
    
    # Compute scene scale
    scene_scale = compute_scene_scale(model)
    
    # Plot the reconstruction (map cameras in blue)
    plot_reconstruction(
        fig=fig,
        model=model,
        camera_color="rgb(0, 100, 255)",
        camera_size=camera_size,
        show_points=True,
        show_cameras=True,
        use_point_colors=True,
        name="Map"
    )
    
    # Plot query poses (in red, larger)
    plot_query_poses(
        fig=fig,
        query_poses=query_poses,
        color="red",
        size=query_size
    )
    
    # Plot origin (small)
    if show_origin:
        origin_scale = scene_scale * 0.15
        plot_origin(fig, scale=origin_scale)
    
    # Save to HTML
    fig.write_html(str(output_path))
    
    return fig


def show_figure(fig: go.Figure):
    """Try to display figure in browser."""
    try:
        fig.show()
    except Exception:
        pass  # May fail in headless environments
