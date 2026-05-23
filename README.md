# Vision–Language–Action Control for 5-DOF Robotic Manipulation

> Hierarchical VLA architecture with multi-modal contact sensing and conditioned motion planning, deployed on an affordable 5-DOF serial-bus servo arm.

---

## Overview

This project implements a **Hierarchical Vision-Language-Action (VLA) system** for robotic manipulation on a low-cost 5-DOF arm built around five STS3215 serial-bus servos. The system accepts natural language instructions (e.g. *"place the red block in tray two"*) and executes them using a hierarchical pipeline that decomposes manipulation into interpretable skill tokens — **REACH → GRASP → LIFT → PLACE** — rather than flat end-to-end joint prediction.

The compute stack is split across two processors:
- **ESP32-WROOM-32** — 50Hz deterministic servo control, sensor acquisition, hardware safety enforcement
- **Raspberry Pi 5 (8GB)** — 8Hz VLA inference, YOLOv8-nano perception, language encoding, IK planning

Training requires as few as **30 teleoperation demonstrations**, making this viable for resource-constrained labs and educational settings.

---

## Hardware

| Component | Part | Role |
|---|---|---|
| Arm | 5× STS3215 serial-bus servos | 5-DOF manipulation |
| MCU | ESP32-WROOM-32 @ 240MHz | Real-time control, sensor fusion |
| SBC | Raspberry Pi 5 8GB | VLA inference, perception |
| Camera | Pi Camera Module 3 (fixed overhead) | Object detection, (X, Y) localization |
| ToF | VL53L5CX 8×8 array (wrist-mounted) | Grasp depth (Z) measurement |
| IMU | ISM330DHCX (end-effector) | Contact detection via vibration |

### Joint Mapping

```
J0  — Base yaw          (servo_pos[0])
J1a — Shoulder A        (servo_pos[1])
J1b — Shoulder B        (servo_pos[2])
J2  — Elbow/wrist pitch (servo_pos[3])
J3  — Gripper           (servo_pos[4])
```

---

## Repository Structure

```
.
├── firmware/                   # ESP32 embedded firmware (PlatformIO)
│   ├── src/                    # Main firmware source
│   │   ├── main.cpp            # FreeRTOS dual-core entry point
│   │   ├── comms.*             # USB serial protocol (250-byte telemetry, 20-byte command)
│   │   ├── servo_bus.*         # STS3215 serial bus driver
│   │   ├── tof_driver.*        # VL53L5CX ToF driver
│   │   ├── ism330dhcx_driver.* # IMU driver
│   │   ├── contact_oracle.*    # Gyro RMS contact detection
│   │   ├── safety_layer.*      # Hardware joint limit enforcement
│   │   └── waypoint_interp.*   # Smooth waypoint interpolation
│   ├── include/config.h        # Pin assignments, tuning constants
│   ├── test_sketches/          # Standalone hardware verification sketches
│   │   ├── imu_whoami/         # IMU I2C connectivity test
│   │   ├── servo_ping/         # Servo bus ping test
│   │   └── tof_distance/       # ToF ranging test
│   ├── tools/                  # Python diagnostic tools
│   │   ├── verify_telemetry.py # Validate live telemetry packets
│   │   ├── servo_monitor.py    # Real-time servo state monitor
│   │   └── read_servos.py      # One-shot servo state reader
│   └── platformio.ini
│
├── rpi5_inference/             # Raspberry Pi 5 Python inference stack
│   ├── main.py                 # Inference loop entry point (8Hz)
│   ├── comms/
│   │   ├── teensy_serial.py    # USB serial reader/writer
│   │   └── servo_driver.py     # High-level servo command interface
│   ├── perception/
│   │   ├── yolo_detector.py    # YOLOv8-nano object detection
│   │   ├── pose_estimation.py  # 3D pose from camera + ToF fusion
│   │   └── camera_manager.py   # Pi Camera 3 capture manager
│   ├── language/
│   │   └── language_encoder.py # Text instruction encoder
│   ├── vla/
│   │   ├── vla_policy.py       # SmolVLA-450M / Octo-small policy
│   │   ├── skill_predictor.py  # Discrete skill token predictor
│   │   └── action_generator.py # Skill-conditioned joint delta generator
│   ├── planning/
│   │   ├── ik_solver.py        # Analytical IK for 5-DOF arm
│   │   └── safety_filter.py    # Joint limit + singularity enforcement
│   ├── calibration/
│   │   ├── camera_calibrate.py         # Camera intrinsic calibration
│   │   ├── overhead_height_calib.py    # Z_table measurement
│   │   ├── wrist_tof_calib.py          # Wrist ToF offset calibration
│   │   ├── camera_intrinsics.yaml      # K matrix + distortion coefficients
│   │   ├── camera_extrinsics.yaml      # T_cam_base transform
│   │   └── homography_dots.yaml        # Pixel-to-world homography
│   ├── config/
│   │   ├── arm_config.yaml     # Joint limits, DH params, servo IDs
│   │   └── model_config.yaml   # Model paths, inference settings
│   ├── dashboard/
│   │   └── gui.py              # PyQt6 live monitoring dashboard
│   └── evaluation/
│       ├── run_eval.py         # Full evaluation runner
│       ├── skill_f1.py         # Skill segmentation F1 metric
│       ├── contact_latency.py  # Contact detection latency benchmark
│       └── ablation.py         # Ablation study runner
│
├── dataset/                    # Dataset pipeline (runs on Colab / workstation)
│   ├── hdf5_reader.py          # Reads raw teleoperation HDF5 recordings
│   ├── skill_segmenter.py      # Automatic skill boundary detection
│   ├── augmentation.py         # Data augmentation transforms
│   └── vla_dataset.py          # PyTorch Dataset for VLA fine-tuning
│
├── checkpoints/                # Model weights
│   └── yolov8n_vla/weights/    # Fine-tuned YOLOv8-nano weights
│
├── demos/                      # Raw teleoperation HDF5 recordings (gitkeep)
│
├── docs/superpowers/
│   ├── specs/                  # Design specifications
│   └── plans/                  # Implementation plans
│
├── AI_ML_WORKPLAN.md           # AI/ML engineer workplan
├── HARDWARE_EMBEDDED_WORKPLAN.md  # Hardware/firmware engineer workplan
├── VLA_Robotic_Arm_Project_Report_FINAL.md  # Master project report
├── requirements.txt
└── README.md
```

---

## Communication Protocol

The ESP32 and RPi 5 communicate over USB serial at **2 Mbps**.

### Telemetry Packet — ESP32 → RPi 5 (250 bytes @ 50Hz)

```python
TELEMETRY_DTYPE = np.dtype([
    ('timestamp_us',     np.uint32),         # ESP32 microsecond counter
    ('servo_pos',        np.float32, (5,)),  # degrees, 0.088°/step
    ('servo_load',       np.float32, (5,)),  # normalized 0.0–1.0
    ('servo_speed',      np.float32, (5,)),  # degrees/second
    ('servo_temp',       np.float32, (5,)),  # Celsius
    ('tof_grid',         np.uint16,  (64,)), # 8×8 zone distances in mm
    ('tof_timestamp_us', np.uint32),
    ('tof_resolution',   np.uint8),          # 64 = 8×8 mode
    ('tof_valid',        np.uint8),          # 1 if frame passed validity
    ('imu_gyro',         np.float32, (3,)),  # deg/s (gx, gy, gz)
    ('imu_accel',        np.float32, (3,)),  # m/s² (ax, ay, az)
    ('contact_flag',     np.uint8),          # 1 if contact oracle triggered
    ('contact_rms',      np.float32),        # gyro RMS value
    ('safety_clamped',   np.uint8),          # 1 if hw safety clamped a cmd
    ('checksum',         np.uint16),
])  # Total: 250 bytes
```

### Command Packet — RPi 5 → ESP32 (20 bytes @ 8Hz)

```python
COMMAND_DTYPE = np.dtype([
    ('target_arm',      np.float32, (3,)),  # J0, J1, J2 target degrees
    ('skill_state',     np.uint8),          # 0=REACH 1=GRASP 2=LIFT 3=PLACE
    ('execute',         np.uint8),          # 1=execute, 0=hold
    ('gripper_command', np.float32),        # 0.0=open, 1.0=closed
    ('emergency_stop',  np.uint8),          # 1=halt all servos immediately
    ('checksum',        np.uint8),
])  # Total: 20 bytes
```

---

## Setup

### Prerequisites

- Python 3.10+
- PlatformIO (for firmware)
- Google Colab (for VLA training — A100/T4 GPU required)

### RPi 5 — Python environment

```bash
pip install -r requirements.txt
```

### Firmware — ESP32

```bash
cd firmware
pio run --target upload
```

### Calibration (run once on RPi 5)

```bash
# Camera intrinsics
python rpi5_inference/calibration/camera_calibrate.py

# Table height (Z_table)
python rpi5_inference/calibration/overhead_height_calib.py

# Wrist ToF offset
python rpi5_inference/calibration/wrist_tof_calib.py
```

Outputs are saved to `rpi5_inference/calibration/*.yaml`.

---

## Running Inference

```bash
# Dry run — validates imports and serial connection without moving servos
python rpi5_inference/main.py --dry-run

# Live inference
python rpi5_inference/main.py
```

### Verify telemetry from ESP32

```bash
python firmware/tools/verify_telemetry.py
```

---

## Training (Google Colab)

1. Collect 30 teleoperation demonstrations — saved as HDF5 files in `demos/`
2. Upload `demos/` and `dataset/` to Google Drive
3. Open the training notebook, mount Drive, and run the pipeline:
   - `dataset/hdf5_reader.py` — parse raw recordings
   - `dataset/skill_segmenter.py` — auto-label skill boundaries
   - `dataset/augmentation.py` — apply augmentation
   - `dataset/vla_dataset.py` — build PyTorch dataset
4. Fine-tune SmolVLA-450M with LoRA (Octo-small as fallback baseline)
5. Export weights to `checkpoints/`

### Demo HDF5 format

```
demos/demo_001_pick_red_block.h5
  /telemetry/
      servo_pos     (N, 5)  float32  degrees
      servo_load    (N, 5)  float32  normalized
      servo_speed   (N, 5)  float32  deg/s
      imu_gyro      (N, 3)  float32  deg/s
      imu_accel     (N, 3)  float32  m/s²
      tof_grid      (N, 64) uint16   mm
      contact_flag  (N,)    uint8
      timestamps_us (N,)    uint32
  /video/
      rgb_frames    (N, H, W, 3) uint8
  /metadata/
      task_instruction  str
      demo_id           str
```

---

## Evaluation

```bash
# Full evaluation suite
python rpi5_inference/evaluation/run_eval.py

# Individual metrics
python rpi5_inference/evaluation/skill_f1.py
python rpi5_inference/evaluation/contact_latency.py
python rpi5_inference/evaluation/ablation.py
```

### Targets

| Metric | Target |
|---|---|
| Task success (pick-place, stack, sort) | ≥ 80% |
| Workspace-compliant trajectories | 100% |
| VLA inference latency (P95) | ≤ 110ms |
| Full pipeline latency (P95) | ≤ 125ms |
| Contact detection latency | ≤ 20ms |

---

## Live Dashboard

```bash
python rpi5_inference/dashboard/gui.py
```

PyQt6 dashboard showing real-time servo states, ToF depth map, contact oracle status, skill state, and inference latency.

---

## Architecture

```
Natural Language Instruction
         │
         ▼
  Language Encoder (RPi 5)
         │
         ▼
  Skill Predictor ──── Visual Observation (YOLOv8-nano + ToF)
  (SmolVLA-450M)
         │
    Skill Token
  (REACH/GRASP/LIFT/PLACE)
         │
         ▼
  Action Generator
  (Skill-conditioned joint deltas)
         │
         ▼
  Analytical IK + Safety Filter
         │
         ▼
  Command Packet (20 bytes @ 8Hz)
         │
    USB Serial
         │
         ▼
  ESP32-WROOM-32
  (50Hz servo control + hardware safety)
         │
         ▼
  STS3215 Servo Bus (5 servos)
```

---

## Key Design Decisions

**Why hierarchical (skill tokens) instead of flat VLA?**
A 5-DOF arm has no kinematic redundancy — small joint prediction errors cause large end-effector deviations. Decomposing into discrete skill phases stabilizes regression and makes failures interpretable.

**Why fixed overhead camera + wrist ToF instead of eye-in-hand?**
Eye-in-hand routing requires a CSI ribbon cable through revolute joints, which fatigues within tens of cycles. Fixed overhead avoids occlusion during REACH; wrist ToF provides precise grasp depth (Z) without the cable routing problem.

**Why STS3215 load feedback for contact sensing?**
The STS3215 servo bus provides normalized load current at 50Hz per servo — a natural force proxy. Combined with ISM330DHCX gyro RMS on the end-effector, this gives contact detection without a dedicated force-torque sensor.

**Why SmolVLA-450M?**
Designed for affordable robotics and CPU-capable deployment. Octo-small + LoRA is retained as a lower-latency fallback if SmolVLA cannot meet the 125ms/step budget on RPi 5.

---

## Dependencies

See `requirements.txt`. Key packages:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model training and inference |
| `ultralytics` | YOLOv8-nano object detection |
| `transformers`, `peft` | SmolVLA / Octo fine-tuning with LoRA |
| `h5py` | HDF5 dataset reading/writing |
| `pyserial` | USB serial communication with ESP32 |
| `opencv-python` | Camera capture and image processing |
| `PyQt6`, `pyqtgraph` | Live dashboard |
| `scipy`, `numpy` | IK solver, signal processing |

---

## Project Report

Full technical report including literature review, system architecture, kinematic model, training methodology, and evaluation plan: [`VLA_Robotic_Arm_Project_Report_FINAL.md`](VLA_Robotic_Arm_Project_Report_FINAL.md)
