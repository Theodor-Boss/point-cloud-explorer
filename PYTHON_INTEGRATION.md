# Python Integration Guide

## Quick Start

### 1. Export from Python

Add this to your Python code:

```python
from python_export import export_pointcloud_to_json, export_lines_to_json

# After generating your point cloud data:
export_pointcloud_to_json(xyz_coords, colors, "pointcloud.json")

# If you have lines:
lines_data = [
    {"start": [0, -2, 0], "end": [1, -2, 0], "color": [1, 1, 1]},
    # ... more lines
]
export_lines_to_json(lines_data, "lines.json")
```

### 2. View in Browser

1. Start the web viewer:
   ```bash
   npm run dev
   ```

2. Open Firefox: `http://localhost:8080`

3. Drag & drop `pointcloud.json` onto the viewer, or click "Choose File"

## Complete Workflow Example

```python
import numpy as np
import torch
from python_export import export_pointcloud_to_json, export_lines_to_json

# Your existing code
num_points = 1000
xyz_rand = np.random.randn(3 * num_points).reshape(-1, 3)
xyz_rand[:, 1] = np.abs(xyz_rand[:, 1])

colors_rand = np.array([
    np.random.uniform(low=0.5, high=1.0, size=num_points),
    np.random.uniform(low=0.5, high=0.7, size=num_points),
    np.random.uniform(low=0.0, high=1.0, size=num_points)
]).T

# Export to JSON
export_pointcloud_to_json(xyz_rand, colors_rand, "pointcloud.json")

# Export lines (optional)
lines = [
    {"start": np.array([0, -2, 0]), "end": np.array([1, -2, 0]), "color": np.array([1, 1, 1])},
    {"start": np.array([1, -2, 0]), "end": np.array([1, -2, 1]), "color": np.array([0, 1, 0])},
]
export_lines_to_json(lines, "lines.json")

print("✓ Data exported! Open http://localhost:8080 and load the files")
```

## JSON Format

### Point Cloud Format
```json
{
  "points": [[x1, y1, z1], [x2, y2, z2], ...],
  "colors": [[r1, g1, b1], [r2, g2, b2], ...],
  "count": 1000
}
```

### Lines Format
```json
{
  "lines": [
    {
      "start": [x1, y1, z1],
      "end": [x2, y2, z2],
      "color": [r, g, b]
    }
  ],
  "count": 4
}
```

## Tips

- Colors should be in [0, 1] range (will auto-convert from [0, 255])
- Works with NumPy arrays and PyTorch tensors
- Automatically flattens (H, W, 3) shapes to (N, 3)
- Large files (>100k points) may take a moment to load

## Controls in Viewer

- **Click** - Lock pointer to look around
- **WASD** - Move around
- **Space** - Move up
- **Shift** - Move down
- **ESC** - Unlock pointer
