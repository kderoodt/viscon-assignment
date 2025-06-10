# Rail Detector & Row Counter  
**Greenhouse localisation for the Viscon Operational Robotics Engineer Assignment**

A ROS 2 C++ package that

* Detects greenhouse heating rails in the infrared stream of an Intel RealSense D456 (Q1)

* Counts the rows the robot has passed using the rail detections & odometry (Q2)

The package is self‑contained and reproducible in ≤ 10 minutes on a fresh ROS 2 Jenny install.

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

## 3  Dependencies

> Tested on Ubuntu 24.04 + ROS 2 Jazzy 

* **OS** Ubuntu 24.04 LTS 
* **ROS 2** Jazzy Jalisco 
* **CUDA-capable GPU**  *(optional)* 
* **ONNX Runtime 1.17+** built **with** or **without** CUDA EP  

### ROS 2 & system packages

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

### ONNX Runtime (GPU)

> **Note:** This step is not needed if run using CPU, this process can take up to 10 min. Build ORT 1.18 with CUDA 12 takes ≈10 min:

>```bash
>mkdir -p ~/onnxruntime && cd ~/onnxruntime
> 
>./build.sh --config Release --update --parallel --build --use_cuda >--cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda >--skip_tests
>
>```

> **Note:** if you compiled ONNX Runtime yourself, export the root before building the workspace:

> ```bash
> export ONNXRUNTIME_ROOT=$HOME/onnxruntime  
> ```

## 4  Build instructions
```bash
# 1. workspace skeleton
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src

# 2. clone repository
git clone https://github.com/kderoodt/viscon-assignment.git rail_detector

# 3. build
cd ~/ros2_ws
colcon build --packages-select rail_detector

# 4. source overlay (every new shell)
source install/setup.bash
```
The ```CMakeLists.txt``` provided in the assignment links OpenCV, ONNX Runtime (CPU + CUDA) and registers **`row_counter_node`** both as a component library *and* a standalone executable.

## 5  Run instructions
### All-in-one launch
```bash
ros2 launch rail_detector rail_detector_launch.py     use_row_counter:=true     use_rqt:=true     use_gpu:=true        
```
Default values are **`use_row_counter=true`**, **`use_rqt=true`**, **`use_gpu=true`**.

### Headless / CPU-only example
```bash
ros2 launch rail_detector rail_detector_launch.py     use_rqt:=false use_gpu:=false
```

## 6  Detailed solution per assignment question


### Q 1 – Row detection
The **`rail_detector_node`** subscribes to the infra-red image, applies classical CV pre-processing, forwards the 768 × 720 crop through an ONNX segmentation network, and finally publishes a binary mask and colour overlay.

Pipeline:

* Pre-processing the image 
    * Rotate image 90° CW            
    * Crop lower 60 %          
    * Enhances local contrast - Clahe(clip=4.0)          
    * Suppresses noise - Gaussian_blur(5×5)
* Using trained ONNX Runtime model for detection prediction
* Post-processing  
    * Only keep large blobs  

Outputs:
* Topic `/rail_detector/preprocessed` – 720 × 768 mono8 pre‑processed framedetections.
* Topic `/rail_mask` – Binary mask (0/255).
* Topic `/rail_detector/overlay` – BGR stream with red rail overlay.


GPU vs CPU is runtime-selectable by appending or omitting the CUDA EP when the **`use_gpu`** parameter is set.

#### 📹 Demo video

https://github.com/user-attachments/assets/a0ca3169-3431-446a-8d1a-679908a3250b


### Q 2 – Row counting
The **`row_counter_node`** listens to the binary mask and odometry:
* Keep only blobs below *roi_y_ratio* (default 0.5 × image height) and larger than *min_area_px* ✕ px.  
* A state-machine detects the rising edge when a new rail enters the view; possible re-entries are debounced via *min_overlap_px*.
* The `(x,y)` odom pose at each detection is compared to the last counted pose.  
   If the Euclidean distance ≥ *row_spacing* (default 1 m) we increment `row_count_` and publish it.

Outputs:
* Topic `rows_count` – live counter for other nodes.
* Console log – human readable summary.


### Q 3 – Testing & validation plan

## 7  Launch-file 

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
