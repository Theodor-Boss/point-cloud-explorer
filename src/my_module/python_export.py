"""
Point Cloud Exporter for Web Viewer
Save this file alongside your Python code and use it to export point clouds.
"""

import numpy as np
import torch
import json
from pathlib import Path


def export_pointcloud_to_json(xyz_coords, colors, output_path="pointcloud.json"):
    """
    Export point cloud data to JSON format for the web viewer.
    
    Args:
        xyz_coords: numpy array or torch tensor of shape (N, 3) or (H, W, 3)
        colors: numpy array or torch tensor of shape (N, 3) or (H, W, 3), RGB in [0, 1] or [0, 255]
        output_path: path to save the JSON file
    """
    # Convert torch tensors to numpy
    if isinstance(xyz_coords, torch.Tensor):
        xyz = xyz_coords.detach().cpu().numpy()
    else:
        xyz = np.asarray(xyz_coords)
    
    if isinstance(colors, torch.Tensor):
        cols = colors.detach().cpu().numpy()
    else:
        cols = np.asarray(colors)
    
    # Flatten if needed
    if xyz.ndim == 3:
        xyz = xyz.reshape(-1, 3)
    if cols.ndim == 3:
        cols = cols.reshape(-1, 3)
    
    # Normalize colors to [0, 1] if needed
    if cols.max() > 1.0:
        cols = cols / 255.0
    
    # Convert to lists for JSON serialization
    data = {
        "points": xyz.astype(np.float32).tolist(),
        "colors": cols.astype(np.float32).tolist(),
        "count": len(xyz)
    }
    
    # Save to file
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"✓ Exported {len(xyz)} points to {output_file.absolute()}")
    return output_file


def export_lines_to_json(lines_data, output_path="lines.json"):
    """
    Export line data to JSON format.
    
    Args:
        lines_data: list of dicts with 'start', 'end', 'color' keys
        output_path: path to save the JSON file
    """
    # Convert numpy arrays to lists
    lines_serializable = []
    for line in lines_data:
        lines_serializable.append({
            "start": np.asarray(line['start']).tolist(),
            "end": np.asarray(line['end']).tolist(),
            "color": np.asarray(line['color']).tolist()
        })
    
    data = {
        "lines": lines_serializable,
        "count": len(lines_serializable)
    }
    
    output_file = Path(output_path)
    with open(output_file, 'w') as f:
        json.dump(data, f)
    
    print(f"✓ Exported {len(lines_serializable)} lines to {output_file.absolute()}")
    return output_file


# ========== EXAMPLE USAGE WITH YOUR CODE ==========

if __name__ == "__main__":
    # Your existing code here...
    dtype = torch.float64
    device = torch.device('cpu')
    
    # Generate random points (your minecraft stars example)
    num_points_minecraft = 1000
    xyz_rand = np.random.randn(3 * num_points_minecraft).reshape(-1, 3)
    xyz_rand[:, 1] = np.abs(xyz_rand[:, 1])  # stars above ground
    
    colors_rand = np.array([
        np.random.uniform(low=0.5, high=1.0, size=num_points_minecraft),  # RED
        np.random.uniform(low=0.5, high=0.7, size=num_points_minecraft),  # GREEN
        np.random.uniform(low=0.0, high=1.0, size=num_points_minecraft)   # BLUE
    ]).T
    
    # Export point cloud
    export_pointcloud_to_json(xyz_rand, colors_rand, "pointcloud.json")
    
    # Export lines (square on ground)
    lines_data = [
        {"start": np.array([0, -2, 0]), "end": np.array([1, -2, 0]), "color": np.array([1, 1, 1])},
        {"start": np.array([1, -2, 0]), "end": np.array([1, -2, 1]), "color": np.array([0, 1, 0])},
        {"start": np.array([1, -2, 1]), "end": np.array([0, -2, 1]), "color": np.array([1, 0, 0])},
        {"start": np.array([0, -2, 1]), "end": np.array([0, -2, 0]), "color": np.array([0, 0, 1])},
    ]
    export_lines_to_json(lines_data, "lines.json")
    
    print("\nNow open the web viewer in Firefox and:")
    print("1. Click 'Load File' and select pointcloud.json")
    print("2. Or drag and drop pointcloud.json onto the viewer")
