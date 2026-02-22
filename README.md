# Multi-Camera Pose Estimation with OAK-D PoE

Distributed multi-camera system using OAK-D PoE depth cameras for real-time human pose estimation. Combines stereo vision geometry (epipolar constraints, triangulation) with deep learning-based skeleton detection (OpenVINO).

## Course Information

- **Course:** [AIS2221 - Industriprosjekt](https://www.ntnu.edu/studies/courses/AIS2221)
- **Institution:** NTNU - Norwegian University of Science and Technology
- **Semester:** Fall 2022

## Overview

This industrial project implements a multi-camera computer vision pipeline for human pose detection in 3D space using Luxonis OAK-D PoE cameras. The system addresses three core challenges:

1. **Camera infrastructure** — TCP/IP streaming from OAK-D PoE cameras, calibration extraction (intrinsics, extrinsics, distortion), and multi-device synchronization via timestamp matching
2. **Stereo geometry** — Fundamental/Essential matrix estimation using 4-point and 8-point algorithms, 3D point triangulation from multiple views
3. **Pose estimation** — Real-time human skeleton detection using OpenVINO's `human-pose-estimation-0001` model, extracting 18 keypoints per person with Part Affinity Fields (PAFs) for multi-person association

## Project Structure

```
.
├── src/
│   ├── camera/
│   │   ├── host.py                    # TCP client - receives MJPEG from OAK PoE
│   │   ├── oak.py                     # OAK pipeline - onboard MJPEG encoding + TCP server
│   │   ├── calibration_extract.py     # Extracts intrinsics/extrinsics/distortion to JSON
│   │   ├── multi_device_sync.py       # Multi-camera timestamp synchronization
│   │   └── calib_camera*.json         # Per-camera calibration data
│   ├── pose/
│   │   ├── multidev_pose.py           # Multi-camera pose estimation with OpenVINO
│   │   ├── depth.py                   # Multi-device depth map computation
│   │   ├── main.py                    # Basic multi-device frame synchronization
│   │   ├── pose.py                    # Keypoint extraction and skeleton association
│   │   └── hand_recognition.py        # MediaPipe hand tracking
│   ├── pose_3d/                        # Real-time 3D body pose estimation (v4)
│   │   ├── main.py                    # Main entry point
│   │   ├── camera.py                  # Camera abstraction class
│   │   ├── config.py                  # Device and path configuration
│   │   ├── utils.py                   # Pose utilities and transformations
│   │   ├── realtime_3d_pose.py        # Complete 3D pose pipeline
│   │   └── show_3d_pose.py            # 3D skeleton visualization (matplotlib)
│   ├── stereo_depth/
│   │   ├── depth_map.py               # Stereo depth computation
│   │   └── depth_map_v2.py            # Improved depth map with rectification
│   └── triangulation/
│       ├── triangulate.py             # 3D point triangulation from N views
│       ├── socp.py                    # Second-order cone programming solver
│       └── io.py                      # Data I/O utilities
├── notebooks/
│   ├── four_point_algorithm.ipynb     # Essential matrix from 4 point correspondences
│   ├── eight_point_algorithm.ipynb    # Fundamental matrix from 8 point correspondences
│   ├── triangulation.ipynb            # 3D reconstruction from stereo pairs
│   ├── stereo_calibration.ipynb       # OpenCV stereo calibration pipeline
│   └── depth_triangulation.ipynb      # Depth map triangulation notebook
├── data/
│   ├── camera id.xlsx                # Camera hardware identification
│   └── calibration/                  # Multi-camera calibration parameters
│       ├── camera_matrix_.csv        # Default camera intrinsics
│       ├── camera_distortion.csv     # Default distortion coefficients
│       ├── rotation_vectors.csv      # Rotation parameters
│       ├── translation_vectors.csv   # Translation parameters
│       ├── camera_1/                 # Per-camera calibration CSVs + device JSON
│       ├── camera_2/
│       └── camera_3/
├── requirements.txt
└── .gitignore
```

## Tech Stack

- **Python 3.8+** - All application code
- **DepthAI SDK** (`depthai`) - OAK-D camera interface, pipeline configuration
- **OpenVINO** - Neural network inference (via `blobconverter`)
- **OpenCV** - Frame processing, MJPEG decoding, visualization
- **NumPy** - Linear algebra for epipolar geometry and triangulation
- **Hardware:** 2-6x Luxonis OAK-D PoE cameras (RGB + stereo depth)

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

OAK-D PoE cameras must be connected on the same network. Default IP: `169.254.1.222`.

### Camera Streaming

Start the OAK-side pipeline, then the host receiver:

```bash
# On OAK device (runs automatically via DepthAI)
python src/camera/oak.py

# On host PC
python src/camera/host.py
```

### Calibration

Extract calibration data from a connected OAK device:

```bash
python src/camera/calibration_extract.py
```

Outputs `calib_{device_id}.json` with intrinsic matrices, distortion coefficients, and stereo rectification.

### Multi-Camera Pose Estimation

```bash
python src/pose/multidev_pose.py
```

This connects to multiple OAK devices, runs the pose estimation model on each, and displays detected skeletons. Update device MX IDs in the script to match your hardware.

### Epipolar Geometry Notebooks

```bash
cd notebooks
jupyter notebook
```

The notebooks demonstrate 4-point and 8-point algorithms on synthetic data, and triangulation for 3D reconstruction.

## Architecture

```
OAK-D PoE Cameras (2-6x)
    │ TCP/IP (MJPEG)
    ▼
Host PC
    ├── Frame synchronization (timestamp matching)
    ├── OpenVINO inference (human-pose-estimation-0001)
    ├── Keypoint extraction (18 body points)
    ├── Multi-person association (Part Affinity Fields)
    └── 3D triangulation (from calibrated camera pairs)
```
