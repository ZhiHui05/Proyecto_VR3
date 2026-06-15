# V3D - 3D Vision-Based Human-Machine Interface for Virtual Drone Control

> Project 2.4: Gesture-controlled virtual drone using hand tracking, rule-based gesture classification, and real-time 3D visualization.

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Gesture Definitions](#gesture-definitions)
6. [Calibration](#calibration)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Project Structure](#project-structure)

---

## Overview

This project implements a complete 3D vision-based human-machine interface that allows a user to control a virtual drone using hand gestures. The system uses:

- **MediaPipe Hands** for real-time hand landmark tracking
- **Rule-based gesture classifier** mapping hand positions to drone commands
- **Open3D** for 3D visualization of the drone in a virtual environment
- **OpenCV** for camera capture and video feed display
- **Multi-threaded architecture** for parallel processing

### Features

- Real-time hand tracking with temporal smoothing
- 12 distinct gesture commands (TAKEOFF, LAND, MOVE directions, ROTATE, HOVER, EMERGENCY STOP)
- Physics-based drone simulation with velocity/acceleration dynamics
- Rich 3D world with ground plane, obstacles, grid, and reference markers
- Live telemetry display showing position, orientation, velocity, and gesture
- Configurable thresholds and parameters via YAML files
- Camera calibration utility with chessboard pattern
- Structured logging to file and console
- Graceful shutdown and error recovery

---

## System Architecture

```
+----------------+     +----------------+     +----------------+
| Camera Module  | --> |   frame_queue  | --> | Hand Tracker   |
+----------------+     +----------------+     +----------------+
                                                        |
+----------------+     +----------------+     +---------v------+
| Open3D Viz     | <-- | telemetry_queue| <-- | Drone Controller|
+----------------+     +----------------+     +----------------+
                              ^                        ^
                              |                        |
                       gesture_queue           gesture_queue
                              |                        |
                       +----------------+      +----------------+
                       | Gesture Classif| <----| tracking_queue |
                       +----------------+      +----------------+
```

**Data Flow:**
1. Camera acquisition thread polls the webcam and pushes frames to `frame_queue`
2. Hand tracking thread runs MediaPipe Hands, extracts 21 landmarks per hand, and pushes to `tracking_queue`
3. Gesture recognition thread classifies hand positions and pushes to `gesture_queue`
4. Drone simulation thread updates position/orientation/velocity and pushes to `telemetry_queue`
5. Visualization thread renders the Open3D scene with the latest telemetry
6. Main thread displays the live camera feed with overlays and handles user input

**Thread Model:**
- Thread 1: Camera acquisition (polls `CameraModule`, enqueues frames)
- Thread 2: MediaPipe hand landmark processing
- Thread 3: Gesture classification
- Thread 4: Drone physics simulation
- Thread 5: Open3D rendering (render loop)

All threads communicate via bounded `queue.Queue` instances. The `SharedState` object provides thread-safe getters/setters with locks for the latest data snapshots.

---

## Installation

### Prerequisites

- Python 3.11 or higher
- Webcam (USB or built-in)
- pip or conda

### Option A: pip install

```bash
# Clone or extract the project
cd v3dproj2

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B: conda install

```bash
cd v3dproj2

# Create conda environment
conda env create -f environment.yml
conda activate v3d-drone-control
```

### Verify Installation

```bash
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import mediapipe as mp; print('MediaPipe:', mp.__version__)"
python -c "import open3d; print('Open3D:', open3d.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
```

---

## Usage

### Quick Start

```bash
# Run the main application
python -m src.main

# With specific camera
python -m src.main --camera 0

# With debug overlay enabled
python -m src.main --debug

# Headless mode (no OpenCV display window)
python -m src.main --headless

# Specify config directory
python -m src.main --config configs/
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit application |
| `Space` | Pause/Resume processing |
| `d` | Toggle debug overlay |
| `r` | Reset drone and classifier |
| `t` | Manual TAKEOFF trigger |
| `l` | Manual LAND trigger |
| `e` | Emergency STOP |
| `h` | Print help |

### Running Tests

```bash
python -m pytest tests/ -v
# or
python -m unittest discover tests/
```

---

## Gesture Definitions

The system recognizes 12 gestures based on relative hand positions and orientations:

| Gesture | Description | Action |
|---------|-------------|--------|
| **TAKEOFF** | Raise both hands upward from low position | Drone lifts off |
| **LAND** | Lower both hands downward from high position | Drone descends and lands |
| **MOVE_FORWARD** | Push both palms forward, hands extended | Drone moves forward (+Y) |
| **MOVE_BACKWARD** | Pull both palms toward body | Drone moves backward (-Y) |
| **MOVE_LEFT** | Move hands to the left side | Drone moves left (-X) |
| **MOVE_RIGHT** | Move hands to the right side | Drone moves right (+X) |
| **MOVE_UP** | Raise dominant hand upward | Drone increases altitude |
| **MOVE_DOWN** | Lower dominant hand downward | Drone decreases altitude |
| **ROTATE_LEFT** | Left hand lower, right hand higher (tilt) | Drone yaws counter-clockwise |
| **ROTATE_RIGHT** | Right hand lower, left hand higher (tilt) | Drone yaws clockwise |
| **HOVER** | Both hands at neutral position, steady | Drone holds position |
| **EMERGENCY_STOP** | Cross wrists to form X shape | Immediate halt, all velocity zeroed |

### How Gestures Are Detected

The classifier uses these features from MediaPipe landmarks:

1. **Wrist positions** (landmark 0) for both hands - absolute and relative positions
2. **Wrist velocity** - frame-to-frame displacement for movement direction
3. **Hand distance** - Euclidean distance between wrists (crossover indicates STOP)
4. **Wrist roll angle** - angle of the line connecting the two wrists vs horizontal
5. **Palm depth** (Z-axis) - whether hands are pushed forward or pulled back
6. **Lateral/vertical displacement ratios** - proportion of offset relative to hand distance

Temporal stabilization uses a sliding window of 8 classifications with majority voting (5+ matches required for gesture change).

---

## Calibration

Camera calibration improves accuracy by correcting lens distortion.

### Run Calibration

**Live capture from camera:**
```bash
python -m calibration.camera_calibration --device 0 --rows 9 --cols 6
```

**Process existing images:**
```bash
python -m calibration.camera_calibration --images data/calibration_images/
```

### Required Chessboard

You need a chessboard pattern with 9x6 inner corners. Print one from:
- https://docs.opencv.org/4.x/pattern.png (10x7)
- Or generate with: `python -c "import cv2; ..."`

The default square size is 25mm. Adjust with `--square-size`.

### Output Files

Calibration saves three files to the output directory:
- `calibration.npz` - NumPy format for programmatic loading
- `calibration.pkl` - Pickle format with metadata
- `calibration_report.txt` - Human-readable calibration report

### Loading Calibration

Pass the calibration file to the main application:
```bash
python -m src.main --calibration data/calibration/calibration.npz
```

---

## Configuration

All parameters are stored in YAML files under the `configs/` directory:

### `configs/default.yaml`
Main application settings: camera, MediaPipe parameters, tracking options, logging, world dimensions.

### `configs/gestures.yaml`
Gesture definitions with thresholds for each gesture. Includes debounce/cooldown settings.

### `configs/drone.yaml`
Drone model physics (speed, acceleration, damping), initial position/orientation, and 3D model appearance.

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `camera.index` | Webcam device index | 0 |
| `camera.width` | Frame width | 1280 |
| `camera.height` | Frame height | 720 |
| `mediapipe.min_detection_confidence` | Hand detection threshold | 0.7 |
| `mediapipe.min_tracking_confidence` | Tracking quality threshold | 0.5 |
| `drone.physics.translation_speed` | Drone movement speed (m/s) | 3.0 |
| `drone.physics.rotation_speed` | Drone rotation speed (deg/s) | 60.0 |
| `world.dimensions` | World boundary box | -10..10 x -10..10 x 0..15 |
| `gestures.thresholds.*` | Gesture classification thresholds | Various |

---

## Troubleshooting

### Camera not opening

- Check camera index: try `--camera 0`, `--camera 1`, etc.
- Verify no other application is using the camera.
- On Linux, check permissions: `ls /dev/video*`

### Poor tracking quality

- Ensure good lighting conditions.
- Keep hands within camera field of view.
- Reduce `min_detection_confidence` in configs.
- Run the calibration utility to correct lens distortion.

### Open3D window not appearing

- Open3D requires a display server. On headless Linux, set up Xvfb:
  ```bash
  Xvfb :99 -screen 0 1024x768x24 &
  export DISPLAY=:99
  ```
- On macOS, ensure you have the required frameworks.
- On Windows, make sure GPU drivers are up to date.

### Low FPS / stuttering

- Reduce camera resolution in `configs/default.yaml`.
- Set `model_complexity` to 0 in the config file.
- Close other GPU-intensive applications.
- Reduce `max_num_hands` to 1 if only using one hand.

### Import errors

```bash
# If you see "No module named 'mediapipe'":
pip install mediapipe

# If Open3D import fails on macOS with Apple Silicon:
pip install open3d --no-binary open3d
# Or use conda:
conda install -c open3d-admin open3d
```

---

## Project Structure

```
v3dproj2/
│
├── calibration/              # Camera calibration utilities
│   ├── __init__.py
│   └── camera_calibration.py
│
├── configs/                  # YAML configuration files
│   ├── default.yaml          # Main application settings
│   ├── gestures.yaml         # Gesture recognition parameters
│   └── drone.yaml            # Drone physics and model settings
│
├── data/                     # Output data directory
│   └── .gitkeep
│
├── logs/                     # Application log files
│   └── .gitkeep
│
├── src/                      # Main source code
│   ├── __init__.py
│   ├── main.py               # Application entry point
│   │
│   ├── camera/               # Camera capture module
│   │   ├── __init__.py
│   │   └── camera_module.py
│   │
│   ├── tracking/             # MediaPipe hand tracking
│   │   ├── __init__.py
│   │   └── hand_tracker.py
│   │
│   ├── gestures/             # Rule-based gesture classifier
│   │   ├── __init__.py
│   │   └── gesture_classifier.py
│   │
│   ├── drone/                # Virtual drone controller
│   │   ├── __init__.py
│   │   └── drone_controller.py
│   │
│   ├── visualization/        # Open3D 3D scene and rendering
│   │   ├── __init__.py
│   │   └── open3d_visualizer.py
│   │
│   ├── threading/            # Multi-threading management
│   │   ├── __init__.py
│   │   └── thread_manager.py
│   │
│   └── utils/                # Utility modules
│       ├── __init__.py
│       ├── config.py         # YAML configuration loader
│       └── logger.py         # Structured logging
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   ├── test_gestures.py      # Gesture classifier tests
│   └── test_drone.py         # Drone controller tests
│
├── requirements.txt          # Python dependencies (pip)
├── environment.yml           # Conda environment specification
└── README.md                 # This file
```

---

## License

This project is created for academic/university coursework purposes.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime |
| OpenCV | 4.8+ | Camera capture, image processing, display |
| MediaPipe | 0.10+ | Hand landmark detection and tracking |
| Open3D | 0.18+ | 3D scene rendering and visualization |
| NumPy | 1.24+ | Numerical computations and array operations |
| SciPy | 1.11+ | Spatial transformations and utilities |
| PyYAML | 6.0+ | Configuration file parsing |
| PyQt5/PySide6 | 5.15+ / 6.0+ | GUI framework (available for future Qt UI extensions) |
