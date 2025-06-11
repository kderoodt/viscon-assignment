# Rail Detector & Row Counter  
**Greenhouse rail detection for the Viscon Operational Robotics Engineer Assignment**

A ROS 2 C++ package that:

* Detects greenhouse heating rails in the infrared stream of an Intel RealSense D456 (Q1)
* Counts the rows the robot has passed using the rail detections & odometry (Q2)

The package is self‑contained and reproducible in ≤ 10 minutes on a fresh ROS 2 Jazzy install.

---

## Table of Contents
* [Project overview](#project-overview)
* [Repository layout](#repository-layout)
* [Prerequisites](#prerequisites)
* [Build instructions](#build-instructions)
* [Run instructions](#run-instructions)
* [Detailed solution per question](#detailed-solution-per-assignment-question)
    * [Q 1 – Row detection](#q-1--row-detection)
    * [Q 2 – Row counting](#q-2--row-counting)
    * [Q 3 – Testing & validation plan](#q-3--testing--validation-plan)
* [Launch-file cheatsheet](#launch-file)
* [Troubleshooting & FAQ](#troubleshooting--faq)

---

## Project overview
A side-mounted Intel RealSense D456 camera looks sideways while the robot drives through greenhouse aisles.  
This repository provides two ROS 2 nodes:

| Node | Purpose | Topics (pub →, sub ←) |
|------|---------|------------------------|
| `rail_detector_node` | Pixel-wise rail segmentation using an ONNX deep learning model | ← `/d456_pole/infra1/image_rect_raw`, → `~/preprocessed`, `~/overlay`, `/rail_mask` |
| `row_counter_node` | Counts how many heating-pipe rails have been passed | ← `/rail_mask`, `/odometry/filtered`, → `rows_count` |

Both nodes can be launched together with a single launch file, optionally bringing up *rqt_image_view* for live inspection and selecting GPU vs CPU execution.

---

## Repository layout

```
root/
├── rail_detector/
│   ├── include/
│   ├── launch/
│   ├── models/
│   ├── src/
│   ├── CMakeLists.txt
│   ├── package.xml
│   └── README.md
├── training/
│   ├── preprocess_images.py
│   ├── postprocess_images.py
│   ├── train_rails.py
│   └── dataset/ ...
├── rail_overlay.mp4
├── README.md
└── .gitignore
```

---

## Prerequisites

Tested on Ubuntu 24.04 + ROS 2 Jazzy

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake git curl \
  ros-jazzy-desktop python3-colcon-common-extensions \
  ros-jazzy-image-transport ros-jazzy-rqt-image-view \
  ros-jazzy-cv-bridge ros-jazzy-ament-index-cpp \
  ros-jazzy-rclcpp ros-jazzy-rclcpp-components \
  ros-jazzy-sensor-msgs ros-jazzy-nav-msgs ros-jazzy-std-msgs \
  libopencv-dev
```

### Install ONNX Runtime (required)

**CPU‑only (quick):**

```bash
sudo apt install libonnxruntime-dev       # small CPU‑only library
```

**GPU build (optional but way faster):**

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --update --parallel --build \
           --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda \
           --skip_tests
export ONNXRUNTIME_ROOT=$HOME/onnxruntime
```

> Skip `--use_cuda` to compile a CPU‑only ORT and launch with `use_gpu:=false`.
```bash
# ONNX Runtime 1.18+ with CUDA 12
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --update --parallel --build --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda --skip_tests
export ONNXRUNTIME_ROOT=$HOME/onnxruntime
```

---

## Build instructions

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/kderoodt/viscon-assignment.git rail_detector
cd ..
colcon build --packages-select rail_detector
source install/setup.bash
```

---

## Run instructions

```bash
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=true     use_rqt:=true     use_gpu:=true
```

### Headless / CPU-only
```bash
ros2 launch rail_detector rail_detector_launch.py     use_rqt:=false use_gpu:=false
```

---

## Detailed solution per assignment question

### Q 1 – Row detection

The `rail_detector_node` performs rail segmentation in a three-stage process:

**Pre-processing**
- Rotate 90° clockwise
- Crop bottom 60% of height
- Apply CLAHE for contrast enhancement
- Apply 5×5 Gaussian blur
- Normalise to float32 (0–1)

**Model**
- Using a custom U-Net architecture trained using `train_rails.py`
- Encoder: ResNet-34 (ImageNet weights)
- Output: 2-class segmentation (rail vs background)
- Loss: CrossEntropy + DiceLoss
- Augmentation: flip, brightness, affine
- Trained with torch + albumentations + segmentation_models_pytorch

**Post-processing**
- Remove small noise blobs using `cv2.connectedComponentsWithStats`
- Output binary mask + overlay image

**Published topics**
- `/rail_detector/preprocessed` – Preprocessed mono8 image
- `/rail_mask` – Binary rail segmentation
- `/rail_detector/overlay` – Red overlay on image

**Demo video**  
https://github.com/user-attachments/assets/a0ca3169-3431-446a-8d1a-679908a3250b

**Training/obtaining U-NET ONNX model**  
- Pre-processing image data using `preprocess_images.py` before annotating data.
- Dataset is annotated using CVAT.
- Annotated data is transformed in a mask using `postprocess_images.py`.
- A custom U-Net architecture trained using `train_rails.py`.

---

### Q 2 – Row counting

The `row_counter_node` implements a robust debounced row counter:

- Input: binary mask + `/odometry/filtered`
- Filters:
  - Ignore blobs above ROI Y (default: lower 50%)
  - Ignore area < min_area_px
  - Compare current pose with last counted rail
- Increment count if distance ≥ row_spacing (0.5 m)
- Debounce overlaps to avoid duplicates

Publishes:
- `/rows_count` (UInt32)
- Console output with verbose detection details

---

### Q 3 – Testing & validation plan

1. Visual validation with:
   - rqt_image_view (`~/overlay`)
   - rosbag replay
2. Ground truth comparison (masks vs predictions)
3. Console log with per-row count outputs
4. Manual inspection with saved overlays & masks
5. Optional metrics script XXX

---

## Launch-file

From `rail_detector/launch/rail_detector_launch.py`:

| Arg              | Default | Description |
|------------------|---------|-------------|
| `use_row_counter`| `true`  | Enable row counting node |
| `use_rqt`        | `true`  | Open overlay in rqt |
| `use_gpu`        | `true`  | Use ONNX CUDA EP |
| `model_path`     | auto    | Path to .onnx model |

Examples:

```bash
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=false use_gpu:=false use_rqt:=false
ros2 launch rail_detector rail_detector_launch.py
```

---

## Troubleshooting & FAQ

**Q:** Rqt shows no image  
**A:** Use topic: `rail_detector_node/overlay`

**Q:** ONNX Runtime crashes with CUDA  
**A:** Use `use_gpu:=false` or ensure correct CUDA version

**Q:** Output mask is empty  
**A:** Verify the model is exported properly and pre-processing matches training

---