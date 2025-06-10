# Rail Detector & Row Counter  
**Greenhouse localisation for the Viscon Operational Robotics Engineer Assignment**

---

## Table of Contents
* [Project overview](#overview)
* [Repository layout](#layout)
* [Prerequisites](#prerequisites)
* [Build instructions](#build)
* [Run instructions](#run)
* [Detailed solution per question](#solutions)
    *    [Q 1 – Row detection](#q1)
    *    [Q 2 – Row counting](#q2)
    *    [Q 3 – Testing & validation plan](#q3)
* [Launch-file cheatsheet](#launch-cheatsheet)
* [Troubleshooting & FAQ](#faq)

---


## 1  Project overview
A side-mounted Intel RealSense D456 camera looks sideways while the robot drives through greenhouse aisles.  
This repository provides two ROS 2 nodes:

| Node | Purpose | Topics (pub →, sub ←) |
|------|---------|------------------------|
| **`rail_detector_node`** | Pixel-wise rail segmentation using an ONNX deep learning model | ← `/d456_pole/infra1/image_rect_raw` (mono-8), `~/preprocessed` (mono-8), `~/overlay` (bgr 8), `/rail_mask` (mono-8) |
| **`row_counter_node`** | Counts how many heating-pipe rails have been passed | ← `/rail_mask`, `/odometry/filtered`, `rows_count` (`std_msgs/UInt32`) |

Both nodes can be launched together with a single launch file, optionally bringing up *rqt_image_view* for live inspection and selecting GPU vs CPU execution.


## 2  Repository layout
```
rail_detector/
├── include/
│   ├── rail_detector_node.hpp
│   └── row_counter_node.hpp                       
├── launch/
│   └── rail_detector_launch.py
├── models/                        
├── src/
│   ├── rail_detector_node.cpp
│   └── row_counter_node.cpp
├── CMakeLists.txt           
├── package.xml
└── README.md
```

## 3  Prerequisites
* **OS** Ubuntu 24.04 LTS (tested)  
* **ROS 2** Humble Hawksbill (desktop-full)  
* **CUDA-capable GPU** *(optional)* Compute ≥ 6.1, CUDA 11.8  
* **ONNX Runtime 1.17+** built **with** or **without** CUDA EP  
* **C/C++ 17 tool-chain** (gcc 11), CMake ≥ 3.16  
* Python 3.10 (for rosbag / evaluation helpers)

<details><summary>One-liner to get the system packages</summary>

```bash
sudo apt update && sudo apt install -y   build-essential cmake git curl   ros-humble-desktop   python3-colcon-common-extensions   python3-pip python3-vcstool   libopencv-dev ros-humble-cv-bridge   ros-humble-image-transport ros-humble-rqt-image-view   ros-humble-nav-msgs ros-humble-ament-cmake   onnxruntime libonnxruntime-dev
```

</details>

> **Note:** if you compiled ONNX Runtime yourself, export the root before building the workspace:
> ```bash
> export ONNXRUNTIME_ROOT=$HOME/onnxruntime  # where include/ and libonnxruntime.so live
> ```

## 4  Build instructions
```bash
# 1. workspace skeleton
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# 2. clone repository
git clone https://github.com/kderoodt/viscon-assignment.git rail_detector

# 3. build
cd ~/ros2_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# 4. source overlay (every new shell)
source install/setup.bash
```
The ```CMakeLists.txt``` provided in the assignment links OpenCV, ONNX Runtime (CPU + CUDA) and registers **`row_counter_node`** both as a component library *and* a standalone executable.

## 5  Run instructions
### 5.1  All-in-one launch
```bash
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=true     use_rqt:=true     use_gpu:=true        
```
Default values are **`use_row_counter=true`**, **`use_rqt=true`**, **`use_gpu=true`**.

### 5.2  Headless / CPU-only example
```bash
ros2 launch rail_detector rail_detector_launch.py     use_rqt:=false use_gpu:=false
```

## 6  Detailed solution per assignment question


### 6.1  Q 1 – Row detection
The **`rail_detector_node`** subscribes to the infra-red image, applies classical CV pre-processing, forwards the 768 × 720 crop through an ONNX segmentation network, and finally publishes a binary mask and colour overlay.

Pipeline (pseudo-code):
```
rotate 90° CW            
crop lower 60 %          
clahe(clip=4.0)          
gaussian_blur(5×5)
↳ ONNX Runtime (sigmoid) → logits
threshold(logits, 0.5)
connected_components()   
```


GPU vs CPU is runtime-selectable by appending or omitting the CUDA EP when the **`use_gpu`** parameter is set.

#### 📹 Demo video

https://github.com/kderoodt/viscon-assignment/blob/main/rail_overlay.mp4



### 6.2  Q 2 – Row counting
The **`row_counter_node`** listens to the binary mask and odometry:
1. Keep only blobs below *roi_y_ratio* (default 0.5 × image height) and larger than *min_area_px* ✕ px.  
2. A tiny state-machine detects the rising edge when a new rail enters the view; possible re-entries are debounced via *min_overlap_px*.
3. The `(x,y)` odom pose at each detection is compared to the last counted pose.  
   If the Euclidean distance ≥ *row_spacing* (default 1 m) we increment `row_count_` and publish it.

Outputs:
* Topic `rows_count` – live counter for other nodes.
* Console log – human readable summary.
* Optional YAML log: `ros2 bag record /rows_count`  → post-processed into *results.yaml*.


### 6.3  Q 3 – Testing & validation plan

## 7  Launch-file cheatsheet
From *launch/rail_detector_launch.py*:

| Argument | Default | Meaning |
|-----------|---------|---------|
| `use_row_counter` | `true` | Whether to start **row_counter_node_exec** |
| `use_rqt`         | `true` | Open *rqt_image_view* on `~/overlay` |
| `use_gpu`         | `true` | Append CUDA EP to ONNX Runtime |
| `model_path`      | *see file* | Absolute or package-relative `.onnx` path |

Example combos:
```bash
# Detector only, headless CPU
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=false use_rqt:=false use_gpu:=false

# Full stack + GUI on GPU
ros2 launch rail_detector rail_detector_launch.py  # no args
```

---

## 8  Troubleshooting & FAQ
> **Q:** ONNX Runtime complains “no CUDA devices found”.  
> **A:** Set `use_gpu:=false` or install the *libonnxruntime-dev* CPU-only package.

> **Q:** Rqt shows an empty image.  
> **A:** Make sure to select `rail_detector_node/overlay`

---
