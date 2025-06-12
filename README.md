# Rail Detector & Row Counter  
**Greenhouse rail detection for the Viscon Operational Robotics Engineer Assignment**

A ROS 2 C++ package `rail_detector` that:

* Detects greenhouse heating rails in the infrared stream of an Intel RealSense D456 (Q1)
* Counts the rows the robot has passed using the rail detections & odometry (Q2)

The package is self‑contained and reproducible in ≤ 10 minutes on a fresh ROS 2 Jazzy install.

The repository also contains a `training` folder. This folder contains files and data for training the U-NET ONNX model. 
> **NOTE!** A trained ONNX model is provided in `rail_detector/models`. So, no training is needed!

---

## Table of Contents
* [Project overview](#project-overview)
* [Repository layout](#repository-layout)
* [Prerequisites](#prerequisites)
* [Build instructions](#build-instructions)
* [Run instructions](#run-instructions)
* [Solution per question](#detailed-solution-per-assignment-question)
    * [Q 1 – Row detection](#q-1--row-detection)
    * [Q 2 – Row counting](#q-2--row-counting)
    * [Q 3 – Testing & validation plan](#q-3--testing--validation-plan)
* [Launch-file](#launch-file)

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
├── Test_and_validation_plan.pdf
├── README.md
└── .gitignore
```

---

## Prerequisites

Tested on:

- Ubuntu 24.04 + ROS 2 Jazzy (recommended)
- Also compatible with ROS 2 Iron and Humble

> If `$ROS_DISTRO` is not defined, set it manually:
> ```bash
> export ROS_DISTRO=jazzy  # or humble, iron, etc.
> ```

```bash
sudo apt update && sudo apt install -y \
  build-essential cmake git curl \
  python3-colcon-common-extensions \
  libopencv-dev \
  ros-$ROS_DISTRO-desktop \
  ros-$ROS_DISTRO-rclcpp \
  ros-$ROS_DISTRO-rclcpp-components \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-image-transport \
  ros-$ROS_DISTRO-sensor-msgs \
  ros-$ROS_DISTRO-std-msgs \
  ros-$ROS_DISTRO-nav-msgs \
  ros-$ROS_DISTRO-rqt-image-view \
  ros-$ROS_DISTRO-ament-cmake \
  ros-$ROS_DISTRO-ament-index-cpp

```


### Install ONNX Runtime (required)

**CPU‑only (quick):**

```bash
# For Ubuntu <= 22.04 (if available in your package manager)
sudo apt install libonnxruntime-dev      
```

```bash
# For Ubuntu ≥ 22.04 (or if the apt package is missing)
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --update --parallel --build --skip_tests
export ONNXRUNTIME_ROOT=$PWD
```

> If you're on Ubuntu 24.04, the `libonnxruntime-dev` package may not exist yet. Use the source build instead.


**GPU build (optional but 5-10min):**

```bash
git clone --recursive https://github.com/microsoft/onnxruntime
cd onnxruntime
./build.sh --config Release --update --parallel --build \
           --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda \
           --skip_tests
export ONNXRUNTIME_ROOT=$HOME/onnxruntime
```
> Only set `ONNXRUNTIME_ROOT` if you built ONNX Runtime manually from source (e.g., for GPU support).

---

> **Note on ONNXRUNTIME_ROOT**  
> The CMake build expects ONNX Runtime to be located in:  
> `${ONNXRUNTIME_ROOT}/build/Linux/Release`  
> 
> If you installed ONNX Runtime to a different location, either:
> - Set `export ONNXRUNTIME_ROOT=/your/custom/path`
> - Or update the `CMakeLists.txt` accordingly:
>   ```cmake
>   link_directories(/your/custom/path/build/Linux/Release)
>   ```

---

## Build instructions

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/kderoodt/viscon-assignment.git
cd ..
export ONNXRUNTIME_ROOT=$HOME/onnxruntime    # only if you built from source
colcon build --packages-select rail_detector
source install/setup.bash
```

---

## Run instructions

```bash
# Launch rail detector and row counter
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=true     use_rqt:=true     use_gpu:=true
```

#### In a second terminal
```bash
# Play rosbag
ros2 bag play ~/rosbags/viscon
```

### CPU-only
```bash
# Launch rail detector and row counter
ros2 launch rail_detector rail_detector_launch.py     use_gpu:=false
```

---

## Detailed solution per assignment question

### Q 1 – Row detection

The `rail_detector_node` runs each incoming IR image through a pre-processing step (resize/contrast-enhance, etc.), feeds the cleaned frame to an ONNX segmentation model, thresholds the output to a binary rail mask, filters out small blobs, and publishes both the mask and a red overlay for visual debug. Parameters let you pick the model file, turn GPU support on/off, and set a minimum blob size. 

#### Training/obtaining U-NET ONNX model

- Pre-processing image data using `preprocess_images.py` before annotating data.
- Dataset is annotated using CVAT.
- Annotated data is transformed in a mask using `postprocess_images.py`.
- A custom U-Net architecture trained using `train_rails.py`.

#### Rail detection 

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
- Remove small noise blobs
- Output binary mask + overlay image

- **Inputs**
  - `/d456_pole/infra1/image_rect_raw` `sensor_msgs/msg/Image` – mono8 image.

**Published topics**
- `/rail_detector/preprocessed` – Preprocessed mono8 image.
- `/rail_mask` – Binary rail segmentation.
- `/rail_detector/overlay` – Red overlay on image.

#### 📹 Demo video

https://github.com/user-attachments/assets/a0ca3169-3431-446a-8d1a-679908a3250b


---

### Q 2 – Row counting

#### OLD APPROACH

The `row_counter_node` uses the binary rail-mask image stream, picks the rail blob closest to the centre of the view, and notes when the robot first “enters” that blob. Once the blob disappears or shifts sideways, it uses odometry to check that the robot has moved at least row_spacing metres; if so, it registers that a rail (row) has been passed and publishes the updated count:

- **Inputs**
  - `/rail_mask` `sensor_msgs/Image` – binary mask (white = rail pixels).
  - `/odometry/filtered` `nav_msgs/Odometry` – robot pose.

- **Published topics**
  - `rows_count` `std_msgs/UInt32` – cumulative number of rails passed.

- **Key Parameters**
  - `row_spacing` (m, default **0.60**) – minimum travel before a new rail can be counted.
  - `min_area_px` (default **1000**) – ignore blobs smaller than this.
  - `min_overlap_px` (default **500**) – overlap threshold to decide we have left the current rail.

- **Algorithm**
  1. Run *connected‑components* on each mask frame.
  2. Pick the blob closest to the image centre (most likely the rail in front).
  3. Stateful logic:  
     - **Enter** rail when a valid blob appears (`active_rail_ = true`).  
     - **Exit** when blob overlap drops below `min_overlap_px` *and* its centre shifts sideways (>100 px).  
  4. On exit, if odometry shows the robot travelled ≥ `row_spacing`, increment and publish `rows_count`.

#### NEW APPROACH (work in progress)

The node continuously segments rails, keeps short-term tracks of every visible rail blob, and increments a row-counter only when a new rail appears after the robot has driven at least row_spacing metres since the previous count.

- **Subscribe** to `/rail_mask` (segmentation) and `/odometry/filtered`.
- **Detect rail blobs** in each frame.
- **Track blobs** across frames: multiple boxes are treated as the same rail when their IoU ≥ 0.05 **or** their centroids shift less than 25 % of the box height.
- **Ignore duplicates**: if a new blob overlaps an existing track with IoU ≥ 0.5 it is considered part of that rails and not counted.
- **Count a rail** only when a rail appears **and** the robot has moved at least `row_spacing` since the last counted rail.
- **Publish** the running count on `rows_count`.


---

### Q 3 – Testing & validation plan

See `Test_and_validation_plan.pdf`

---

## Launch-file

From `rail_detector/launch/rail_detector_launch.py`:

| Arg              | Default | Description |
|------------------|---------|-------------|
| `use_row_counter`| `true`  | Enable row counting node |
| `use_rqt`        | `true`  | Open overlay in rqt |
| `use_gpu`        | `true`  | Use ONNX CUDA EP |

Examples:

```bash
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=false use_gpu:=false use_rqt:=false
ros2 launch rail_detector rail_detector_launch.py
```

---
