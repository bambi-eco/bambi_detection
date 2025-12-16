# BAMBI Detection

The [BAMBI project](https://www.bambi.eco/) uses camera drones together with artificial intelligence to automatically monitor wildlife. 
Light field technology is used, which for the first time makes it possible to visualize what is happening on the forest floor, and thus to detect animals with a high degree of reliability. 
Based on this technology, an AI-powered system can detect and automatically classify animals on the forest floor and in the open terrain, thus allowing an area-wide and accurate count of wildlife that has not been possible until now.

## Features

- **Video Frame Extraction**: Extract frames from drone video footage with associated pose and GPS metadata
- **Image Projection**: Orthographic projection and light field (ALFS) rendering onto digital elevation models
- **Wildlife Detection**: YOLO-based object detection with tiled inference for high-resolution images
- **Label Georeferencing**: Project detected bounding boxes to real-world coordinates using DEM raytracing
- **Flight Data Export**: Generate GeoJSON files for flight routes and monitored area polygons with area/perimeter statistics
- **Multi-Format Support**: Read and write annotations in YOLO, MOT, Labelbox, and custom BAMBI formats
- **Track Management**: Track objects across video frames with interpolation and simplification utilities
- **Visualization**: Generate annotated videos and track visualizations

## Project Structure

```
bambi_detection/
├── src/
│   └── bambi/
│       ├── ai/                          # AI detection and tracking
│       │   ├── domain/                  # BoundingBox and Track classes
│       │   ├── input/                   # Annotation readers (YOLO, MOT, Labelbox)
│       │   ├── models/                  # Wrapper of Ultralytics YOLO detector and tracker
│       │   ├── output/                  # Annotation writers
│       │   ├── util/                    # Filtering, tracking, interpolation
│       │   └── visualization/           # Bounding box and track visualization
│       ├── airdata/                     # DJI AirData flight log parsing
│       ├── domain/                      # Drone, Camera, Sensor definitions
│       ├── geo/                         # GPS EXIF writing utilities
│       ├── srt/                         # DJI SRT subtitle parsing
│       ├── util/                        # Image, math, and projection utilities
│       ├── video/                       # Video frame access and writing
│       ├── webgl/                       # Pose extraction from flight data
│       ├── bambi_detection.py           # Main processing pipeline
│       ├── comparative_visualization.py # Compare detection results
│       ├── drone_geotiff_generator.py   # Generate GeoTIFF outputs
│       ├── georeferenced_tracking.py    # Georeferenced object tracking
│       ├── georeference_polygons.py     # Georeference polygon annotations
│       ├── orthomosaic.py               # Generate GeoTIFF outputs
│       ├── tracks_to_geojson.py         # Export tracks as GeoJSON
│       └── visualize_tracks_global_and_image.py
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for inference)

### Installation

1. **Install alfs_py dependency**

   This repository requires the [alfs_py project](https://github.com/bambi-eco/alfs_py) which is not available on PyPI:

   ```bash
   pip install <path-to-alfs_py-project>
   ```

2. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   ```


## Configuration

The main script `bambi_detection.py` is configured through variables at the top of the file:

### Processing Steps

```python
steps_to_do = {
    "extract_frames": True,              # Step 1: Extract video frames
    "project_frames": True,              # Step 2: Project frames onto DEM
    "skip_existing_projection": True,    # Skip already projected frames
    "projection_method": ProjectionType.OrthographicProjection,
    "detect_animals": True,              # Step 3: Run YOLO detection
    "skip_already_inferenced_frames": True,
    "project_labels": True,              # Step 4: Georeference detections
    "export_flight_data": True,          # Step 5: Export route and area
    "export_individual_polygons": True   # Export per-frame polygons
}
```

### Projection Types

- `ProjectionType.NoProjection` - Use original frames for detection
- `ProjectionType.OrthographicProjection` - Orthographic projection onto DEM
- `ProjectionType.AlfsProjection` - Light field rendering (ALFS)

### Rendering Settings

```python
sample_rate = 1              # Process every Nth frame
limit = -1                   # Max frames to process (-1 = all)
alfs_number_of_neighbors = 100   # Neighbors for light field
alfs_neighbor_sample_rate = 10   # Sample rate within neighborhood

ortho_width = 70             # Orthographic width in meters
ortho_height = 70            # Orthographic height in meters
render_width = 2048          # Output image width in pixels
render_height = 2048         # Output image height in pixels
fovy = 50                    # Field of view for projection
```

### Detection Settings

```python
model_path = r"path/to/thermal_animal_detector.pt"
labels = ['animal']
min_confidence = 0.5         # Minimum detection confidence
verbose = False              # Ultralytics console output
```

Our YOLO11 model is available on [Hugginface](https://huggingface.co/cpraschl/bambi-thermal-detection).

## Usage

### Processing Pipeline

1. **Prepare input data** (see [Input Data Requirements](#input-data-requirements))

2. **Configure paths** in `bambi_detection.py`:
   ```python
   videos = [
       r"path/to/video1.MP4",
       r"path/to/video2.MP4"
   ]
   srts = [srt.replace(".MP4", ".SRT") for srt in videos]
   air_data_path = r"path/to/air_data.csv"
   target_folder = r"path/to/output"
   path_to_dem = r"path/to/dem.gltf"
   path_to_calibration = r"path/to/calib.json"
   path_to_flight_correction = r"path/to/correction.json"
   camera_name = "T"  # "T" for Thermal, "W" for Wide
   target_crs = CRS.from_epsg(32633)  # Must match DEM CRS
   ```

3. **Enable desired steps** in `steps_to_do`

4. **Run the script**:
   ```bash
   python src/bambi/bambi_detection.py
   ```

### Pipeline Stages

| Step | Description | Input | Output |
|------|-------------|-------|--------|
| 1. Extract Frames | Extract video frames with pose metadata | Videos, SRTs, AirData | `poses.json`, frame images, mask |
| 2. Project Frames | Project frames onto DEM | Frames, DEM, calibration | `*_projected.png` or `*_alfs.png` |
| 3. Detect Animals | Run YOLO inference on frames | Projected frames | `*.txt` (YOLO format) |
| 4. Project Labels | Georeference detections | Labels, DEM, poses | `*.json`, `*.geojson` |
| 5. Export Flight Data | Calculate area and route | All metadata | `route.geojson`, `area.geojson` |

### Additional Scripts

Beyond the main `bambi_detection.py` pipeline, the repository includes several specialized scripts for specific tasks:

#### `comparative_visualization.py`
**MOT Tracking Comparison Visualization Tool**

Creates side-by-side grid visualizations comparing ground truth annotations with multiple tracker outputs for Multi-Object Tracking (MOT) evaluation. Useful for ablation studies and tracker performance comparison.

```bash
python comparative_visualization.py <tracking_results_base> <sequences_base> <output_base>
```

Features:
- Generates comparison grids showing ground truth vs. different tracker configurations
- Supports video output with optional image cleanup
- Parses tracker configuration from folder names (e.g., `modelbotsort_use_embs1_use_velocity0`)
- Consistent color palette for track IDs across visualizations

---

#### `drone_geotiff_generator.py`
**Drone Video Frame to GeoTIFF Generator**

Projects drone video frames onto a Digital Elevation Model and exports georeferenced GeoTIFF files compatible with GIS software.

```bash
python drone_geotiff_generator.py --sequence-id <ID> --images-folder <path> \
    --data-folder <path> --output-folder <path>
```

Features:
- Projects frames using camera intrinsics and extrinsics onto DEM
- Outputs GeoTIFF files with proper CRS metadata (UTM 33N / EPSG:32633)
- Handles coordinate transformation from local mesh coordinates to UTM

---

#### `georeferenced_tracking.py`
**Georeferenced Object Tracking**

Performs object tracking in world coordinates (UTM) rather than pixel coordinates, enabling consistent tracking across frames even with camera movement.

Features:
- Multiple tracker modes: `GREEDY`, `HUNGARIAN`, `CENTER`, `HUNGARIAN_CENTER`
- IoU-based and center-distance-based association
- Class-aware tracking option
- Track interpolation for missed detections
- Outputs tracks with 3D world coordinates (x, y, z)

---

#### `georeference_deepsort_mot.py`
**Georeference DeepSORT MOT Results**

Converts pixel-space tracking results (from DeepSORT or similar trackers) to world coordinates using camera poses and DEM raytracing.

Features:
- Pose smoothing using Savitzky-Golay filter to reduce GPS jitter
- Projects bounding box corners to 3D world coordinates
- Supports batch processing of multiple sequences
- Handles coordinate system transformations (WGS84 ↔ UTM)

---

#### `georeference_polygons.py`
**Georeference SAM3 Polygons**

Converts local (pixel-space) polygon annotations from segmentation models (e.g., SAM3) to georeferenced world coordinates.

```bash
python georeference_polygons.py --source ./local_polygons --target ./georeferenced_polygons \
    --correction-folder ./correction_data --flight-id 223
```

Input format (per line):
```
<object_type> <num_points> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

Output format (per line):
```
<object_type> <num_points> <X1> <Y1> <Z1> <X2> <Y2> <Z2> ... <XN> <YN> <ZN>
```

---

#### `misb_video_converter.py`
**MISB ST 0601 Video Converter**

Creates videos with embedded KLV metadata tracks following the MISB ST 0601.17 standard. Output is compatible with QGIS and other GIS applications that support MISB metadata for drone video overlay.

```bash
python misb_video_converter.py --sequence-id <ID> --images-folder <path> \
    --poses-file <path> --output <path.mp4>
```

Features:
- Generates KLV (Key-Length-Value) metadata packets per MISB ST 0601.17
- Embeds platform position, orientation, sensor parameters, and timestamps
- Requires FFmpeg for video encoding
- Compatible with QGIS video geotagging

---

#### `orthomosaic.py`
**Drone Frames to orthomosaic**

Projects multiple drone video frames onto a Digital Elevation Model and exports one georeferenced orthomosaic as GeoTIFF compatible with GIS software.

```bash
python drone_geotiff_generator.py \
    --sequence-id <ID> \
    --images-folder <path> \
    --data-folder <path> \
    --output-folder <path>
```

Features:
- Projects frames using camera intrinsics and extrinsics onto DEM
- Outputs GeoTIFF file with proper CRS metadata (UTM 33N / EPSG:32633)
- Handles coordinate transformation from local mesh coordinates to UTM

---

#### `tracks_to_geojson.py`
**Export Tracks to GeoJSON**

Converts georeferenced tracking results (CSV format) to GeoJSON for visualization in GIS applications or web maps.

Features:
- Exports track trajectories as LineStrings
- Exports detection bounding boxes as Polygons
- Exports detection centers as Points
- Deterministic color assignment per track ID
- Supports both individual track export and combined flight export
- Transforms UTM coordinates to WGS84 (EPSG:4326)

---

#### `visualize_tracks_global_and_image.py`
**Dual-View Track Visualization**

Creates synchronized visualizations showing tracks both in image space (pixel coordinates) and world space (georeferenced map view).

Features:
- Side-by-side or overlaid visualization of local and global tracks
- Track interpolation for smooth visualization
- Map tile fetching for background context
- Video output with FFmpeg
- Consistent track coloring across views
- Supports multiple tracking algorithms and configurations

## Input Data Requirements

### Video Files
- Format: MP4 (DJI drone recordings)
- Multiple videos from a single flight should be provided in chronological order

### SRT Files
- DJI subtitle files containing frame-by-frame metadata:
  - GPS coordinates (latitude, longitude, altitude)
  - Gimbal angles
  - Timestamps
  - ISO, shutter, aperture settings

### AirData CSV
- Flight log exported from [AirData](https://airdata.com/)
- Contains high-precision flight telemetry

### Digital Elevation Model
- **Format**: GLTF/GLB mesh with accompanying JSON metadata
- **JSON metadata** should include:
  ```json
  {
    "origin": [x_offset, y_offset, z_offset],
    "origin_wgs84": {
      "latitude": 0.0,
      "longitude": 0.0,
      "altitude": 0.0
    },
    ...
  }
  ```
- CRS must match `target_crs` setting

### Camera Calibration
- JSON file with intrinsic camera parameters for distortion correction

### Flight Correction
- JSON file with translation and rotation corrections:
  ```json
  {
    "translation": {"x": 0, "y": 0, "z": 0},
    "rotation": {"x": 0, "y": 0, "z": 0}
  }
  ```

## Output Formats

### Poses JSON (`poses.json`)
Contains frame-by-frame metadata:
```json
{
  "images": [
    {
      "imagefile": "frame_0001.png",
      "location": [x, y, z],
      "rotation": [rx, ry, rz],
      "lat": 47.123,
      "lng": 11.456,
      "fovy": [50.0]
    }
  ]
}
```

### YOLO Annotations (`*.txt`)
Standard YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

### Georeferenced Labels (`*.json`)
```json
{
  "Labels": [
    {
      "Class": "animal",
      "DemCoordinates": [[x, y, z], ...],
      "WGS84Coordinates": [[lon, lat, alt], ...]
    }
  ],
  "EPSG": "EPSG:32633"
}
```

### GeoJSON Outputs
- **`route.geojson`** - Flight path as LineString
- **`area.geojson`** - Monitored area as Polygon with area/perimeter properties
- **`*_area.geojson`** - Per-frame coverage polygons

## Supported Hardware

### Drones
| Model | Manufacturer | Cameras |
|-------|--------------|---------|
| M2EA | DJI | Wide, Thermal |
| M3T/M3TE | DJI | Wide, Thermal, Zoom |
| M30T | DJI | Wide, Thermal, Zoom |
| M300 | DJI | Wide, Thermal, Zoom |

### Cameras
- **T (Thermal)** - Thermal infrared camera
- **W (Wide)** - Wide-angle RGB camera  
- **Z (Zoom)** - Zoom RGB camera (where available)

## Known Issues

### GLTFLib/Trimesh Index Error

## Known Issues

We are using `GLTFLib` for reading the digital elevation models and are converting it to a mesh using `Trimesh` and some internal functions.
However, sometimes the `Trimesh(vertices=mesh_data.vertices, faces=mesh_data.indices)` constructor raises an `IndexError`, when building up this mesh.
Unfortunately, we don't know what is the reason for this and it is non-deterministic.
When running the script multiple times with the exact same input, this error occurs sometimes, but not always.
Since the digital elevation models are loaded multiple times across the script (needed in different steps), this problem may occur at different stages.
However, the script is designed to reuse the results from the previous stages. So, if the error occurs deactivate all previous (successful) stages and just re-run the failed stages.
Be careful, with re-running the `extract_frames` stages, this will clean up the target folder to avoid inconsistencies between the follow up stages.

### Frame Extraction Warning

The `extract_frames` step will **delete the entire target folder** before extracting frames. Ensure you have backups of any important data before enabling this step.
