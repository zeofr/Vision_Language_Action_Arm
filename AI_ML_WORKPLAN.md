# AI/ML Workplan — VLA Robotic Arm Project
### Role: AI/ML Engineer | Collaborator on Vision-Language-Action Manipulation System

> **Read first:** `VLA_Robotic_Arm_Project_Report_FINAL.md` is the master specification.
> Team's hardware/firmware/embedded workplan is in `Team_HARDWARE_EMBEDDED_WORKPLAN.md`.
> This document covers everything **you** own: perception models, VLA training, inference pipeline, dashboard, and evaluation.
> When an AI reads this file alongside the main report, it has enough context to implement every step without ambiguity.

---

## Your Role in the System

The ream owns the Teensy 4.1 firmware, servo bus, sensor acquisition, and hardware safety layer. He produces:
- A **250-byte telemetry packet** at **50Hz** over USB serial (all servo states, IMU data, contact oracle outputs, wrist ToF frames)
- A **20-byte command packet** interface at **8Hz** accepting target joint positions, skill state, and emergency stop

**You own everything that runs on the Raspberry Pi 5**: object detection, 3D pose estimation, language encoding, VLA policy inference, skill prediction, IK planning, safety filtering, the live dashboard, all training on Colab, and all evaluation.

The system boundary between you and The team is the USB serial link. Your Python code reads Team's 250-byte telemetry packets and writes 20-byte command packets. Everything else is yours.

---

## Communication Protocol Reference (Do Not Change)

This is fixed by Team's firmware. Your Python must match exactly.

### Telemetry Packet — Teensy → RPi 5 (250 bytes, 50Hz)

```python
import numpy as np

TELEMETRY_DTYPE = np.dtype([
    ('timestamp_us',     np.uint32),        # 4 bytes
    ('servo_pos',        np.float32, (5,)), # 20 bytes  — degrees, 0.088°/step
    ('servo_load',       np.float32, (5,)), # 20 bytes  — normalized 0.0–1.0
    ('servo_speed',      np.float32, (5,)), # 20 bytes  — degrees/second
    ('servo_temp',       np.float32, (5,)), # 20 bytes  — Celsius
    ('tof_grid',         np.uint16,  (64,)),# 128 bytes — 8×8 zone distances in mm
    ('tof_timestamp_us', np.uint32),        # 4 bytes
    ('tof_resolution',   np.uint8),         # 1 byte    — 64 for 8×8 mode
    ('tof_valid',        np.uint8),         # 1 byte    — 1 if frame passed validity
    ('imu_gyro',         np.float32, (3,)), # 12 bytes  — deg/s (gx, gy, gz)
    ('imu_accel',        np.float32, (3,)), # 12 bytes  — m/s² (ax, ay, az)
    ('contact_flag',     np.uint8),         # 1 byte    — 1 if contact oracle triggered
    ('contact_rms',      np.float32),       # 4 bytes   — gyro RMS for monitoring
    ('safety_clamped',   np.uint8),         # 1 byte    — 1 if hw safety clamped a cmd
    ('checksum',         np.uint16),        # 2 bytes
])
# Total: 250 bytes exactly
```

### Command Packet — RPi 5 → Teensy (20 bytes, 8Hz)

```python
COMMAND_DTYPE = np.dtype([
    ('target_arm',       np.float32, (3,)), # 12 bytes  — J0, J1, J2 target degrees
    ('skill_state',      np.uint8),         # 1 byte    — 0=REACH,1=GRASP,2=LIFT,3=PLACE
    ('execute',          np.uint8),         # 1 byte    — 1=execute, 0=hold
    ('gripper_command',  np.float32),       # 4 bytes   — 0.0=open, 1.0=closed
    ('emergency_stop',   np.uint8),         # 1 byte    — 1=halt all servos
    ('checksum',         np.uint8),         # 1 byte
])
# Total: 20 bytes exactly
```

### Skill State Encoding
```python
SKILL = {'REACH': 0, 'GRASP': 1, 'LIFT': 2, 'PLACE': 3}
SKILL_INV = {0: 'REACH', 1: 'GRASP', 2: 'LIFT', 3: 'PLACE'}
```

### Joint Index Mapping
```
servo_pos[0]  → J0  — Base yaw (degrees)
servo_pos[1]  → J1a — Shoulder servo A (coupled pair, degrees)
servo_pos[2]  → J1b — Shoulder servo B (coupled pair, degrees)
servo_pos[3]  → J2  — Elbow/wrist pitch (degrees)
servo_pos[4]  → J3  — Gripper (degrees → normalize to 0.0–1.0 for gripper_command)
```

**Logical joint state for VLA input (4D):**
```python
def logical_joint_state(telemetry):
    j0  = telemetry['servo_pos'][0]
    j1  = (telemetry['servo_pos'][1] + telemetry['servo_pos'][2]) / 2.0  # average coupled pair
    j2  = telemetry['servo_pos'][3]
    j3  = telemetry['servo_pos'][4]
    return np.array([j0, j1, j2, j3], dtype=np.float32)
```

---

## DH Parameters and Kinematics Reference

```python
# Denavit-Hartenberg parameters — from main report Section 11.1
# [theta_offset, d_mm, a_mm, alpha_deg]
DH = [
    [0, 65,  0,   90],   # Link 1: base yaw   — d1=65mm
    [0,  0, 130,   0],   # Link 2: shoulder   — a2=130mm
    [0,  0, 190,   0],   # Link 3: elbow      — a3=190mm
]
```

---

## Project Timeline (8 Weeks Total)

| Week | Your Deliverable |
|------|-----------------|
| 1    | Environment setup, data collection tool ready |
| 2    | YOLOv8-nano fine-tuned, calibration scripts working |
| 3    | Dataset pipeline: HDF5 loading, segmentation, augmentation |
| 4    | SmolVLA/Octo LoRA training complete, checkpoint exported |
| 5    | RPi5 benchmark done, model format selected, teensy_serial.py working |
| 6    | Full inference pipeline integrated and running at 8Hz |
| 7    | Live dashboard operational, PyQt6 all 5 panels |
| 8    | Evaluation complete, ablation table, demo video |

---

## Phase 0 — Prerequisites and Coordination with Team

**Goal:** Establish clear interfaces with Team's work before writing any AI/ML code.

### 0.1 What to Get from Team Before You Start

1. **Serial port name** of the Teensy on the RPi 5 (usually `/dev/ttyACM0`).
2. **Calibration files** from Team's calibration scripts:
   - `camera_intrinsics.yaml` — camera intrinsic matrix K and distortion coefficients
   - `overhead_height.yaml` — Z_table (meters from camera to table surface)
   - `wrist_tof_offset.yaml` — mechanical wrist-to-sensor offset in mm
   - `camera_to_base_transform.yaml` — 4×4 homogeneous matrix T_cam_base
   - `arm_config.yaml` — DH params, joint limits, workspace bounds
3. **Confirmation** that `telemetry.contact_flag` is firing correctly during a manual contact test.
4. **Confirmation** that the Teensy is transmitting 250-byte packets at 50Hz (run `teensy_serial.py` listener and verify).

**Expected Output of Phase 0:** You can run `python3 teensy_serial.py` and see correctly parsed telemetry structs printing to the terminal in real time. You can also send a test command packet (execute=0, emergency_stop=0) and verify the Teensy acknowledges it without triggering a safety clamp.

---

## Phase 1 — Environment Setup

**Goal:** Both the Raspberry Pi 5 (inference runtime) and Google Colab (training) are fully configured and verified.

### 1.1 Raspberry Pi 5 — Ubuntu 24.04 Setup

```bash
# System base
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git cmake build-essential

# Create project virtualenv
python3.11 -m venv ~/vla_env
source ~/vla_env/bin/activate

# PyTorch for ARM64 (CPU-only inference)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Core ML and vision
pip install ultralytics          # YOLOv8
pip install transformers>=4.40   # Hugging Face transformers (SmolVLA/T5)
pip install peft>=0.10           # LoRA adapters
pip install accelerate           # Training utilities
pip install h5py                 # HDF5 dataset reading
pip install numpy scipy opencv-python pyserial
pip install pyqt6                # Live dashboard

# libcamera Python binding (RPi 5 camera)
sudo apt install -y python3-libcamera python3-picamera2
pip install picamera2

# Verify torch
python3 -c "import torch; print(torch.__version__)"

# Verify libcamera
python3 -c "from picamera2 import Picamera2; print('camera ok')"

# Verify YOLOv8
python3 -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); print('yolo ok')"
```

### 1.2 Google Colab — Training Environment Setup

Create a new Colab notebook titled `VLA_Training.ipynb`. At the top, paste the following setup cell:

```python
# Cell 1 — Runtime: A100 GPU (Runtime → Change runtime type → A100)
!nvidia-smi   # Verify GPU

# Install training dependencies
!pip install -q torch torchvision transformers>=4.40 peft>=0.10 accelerate
!pip install -q ultralytics h5py datasets timm

# Clone/mount your project repo
from google.colab import drive
drive.mount('/content/drive')

# Set paths
import os
PROJECT_DIR = '/content/drive/MyDrive/vla_rob'
RAW_DEMO_DIR = os.path.join(PROJECT_DIR, 'demos')
PROCESSED_DATASET_DIR = os.path.join(PROJECT_DIR, 'dataset', 'processed')
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATASET_DIR, exist_ok=True)

print(f"Project: {PROJECT_DIR}")
print(f"Raw demos: {RAW_DEMO_DIR}")
print(f"Processed dataset: {PROCESSED_DATASET_DIR}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
```

### 1.3 Repository Structure to Create

```
rpi5_inference/
├── main.py                        # Main inference loop entry point
├── comms/
│   └── teensy_serial.py           # USB serial packet encode/decode
├── perception/
│   ├── camera_manager.py          # libcamera ring buffer
│   ├── yolo_detector.py           # YOLOv8-nano wrapper
│   └── pose_estimation.py         # overhead (X,Y) + wrist ToF (Z) → 3D pose
├── language/
│   └── language_encoder.py        # SmolVLA language path or T5-small cache
├── vla/
│   ├── vla_policy.py              # SmolVLA-450M or Octo-small policy wrapper
│   ├── skill_predictor.py         # Skill state machine + transition logic
│   └── action_generator.py        # 8-step action chunk decoder
├── planning/
│   ├── ik_solver.py               # Closed-form 3-DOF IK
│   └── safety_filter.py           # Workspace + singularity check
├── dashboard/
│   └── gui.py                     # PyQt6 live dashboard, 5 panels
├── calibration/
│   ├── camera_calibrate.py        # Checkerboard intrinsic calibration
│   ├── overhead_height_calib.py   # Z_table measurement
│   └── wrist_tof_calib.py         # Wrist offset measurement
└── config/
    ├── arm_config.yaml
    └── model_config.yaml
```

**Expected Output of Phase 1:**
- `python3 main.py --dry-run` imports all modules without error.
- `python3 comms/teensy_serial.py` reads live telemetry from Team's Teensy and prints parsed structs.
- Colab notebook mounts Drive and installs all dependencies without conflicts.
- GPU is available in Colab (`nvidia-smi` shows A100 or T4).

---

## Phase 2 — Object Detection: YOLOv8-nano Fine-Tuning

**Goal:** A fine-tuned YOLOv8-nano checkpoint that detects all manipulation objects (colored blocks, cylinders, trays) with >90% confidence from the overhead camera view.

### 2.1 Data Collection Script

The overhead camera is fixed. All images must be captured from **exactly** the overhead mount position, with the same lighting conditions you will use during demos and evaluation. Do not use images from other angles.

```python
# capture_dataset.py — run on RPi 5 to collect labeled training images
from picamera2 import Picamera2
import cv2, os, time

OUTPUT_DIR = 'yolo_dataset/images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

cam = Picamera2()
cam.configure(cam.create_still_configuration(
    main={"size": (640, 480), "format": "RGB888"}
))
cam.start()

print("Press SPACE to capture, Q to quit")
img_count = 0
while True:
    frame = cam.capture_array()
    cv2.imshow('capture', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    key = cv2.waitKey(1)
    if key == ord(' '):
        path = os.path.join(OUTPUT_DIR, f'img_{img_count:04d}.jpg')
        cv2.imwrite(path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"Saved {path}")
        img_count += 1
    elif key == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
```

**Collection protocol:**
- Capture **200 images per class**: red_block, blue_block, yellow_block, cylinder, tray
- Vary object **position** across the full workspace (not just center)
- Vary object **rotation** (0°, 45°, 90°, 135°) for blocks and cylinders
- Vary **background clutter** (extra objects in frame that are NOT the target)
- Total: ~1,000 images minimum across 5 classes

### 2.2 Labeling with LabelImg

```bash
# Install LabelImg on your laptop (not RPi5)
pip install labelimg
labelimg yolo_dataset/images yolo_dataset/classes.txt
```

Set format to **YOLO**. Label each object with its class name. Export labels to `yolo_dataset/labels/`.

`classes.txt`:
```
red_block
blue_block
yellow_block
cylinder
tray
```

### 2.3 Dataset Split and YAML Config

```python
# split_dataset.py
import os, shutil, random

IMAGES = 'yolo_dataset/images'
LABELS = 'yolo_dataset/labels'

imgs = [f for f in os.listdir(IMAGES) if f.endswith('.jpg')]
random.shuffle(imgs)
n = len(imgs)
train = imgs[:int(0.8*n)]
val   = imgs[int(0.8*n):]

for split, subset in [('train', train), ('val', val)]:
    os.makedirs(f'yolo_dataset/{split}/images', exist_ok=True)
    os.makedirs(f'yolo_dataset/{split}/labels', exist_ok=True)
    for img in subset:
        shutil.copy(f'{IMAGES}/{img}', f'yolo_dataset/{split}/images/{img}')
        lbl = img.replace('.jpg', '.txt')
        if os.path.exists(f'{LABELS}/{lbl}'):
            shutil.copy(f'{LABELS}/{lbl}', f'yolo_dataset/{split}/labels/{lbl}')
```

`yolo_dataset/dataset.yaml`:
```yaml
path: yolo_dataset
train: train/images
val:   val/images
nc: 5
names: ['red_block', 'blue_block', 'yellow_block', 'cylinder', 'tray']
```

### 2.4 Fine-Tuning on Colab

```python
# In VLA_Training.ipynb — Cell: YOLOv8 Fine-Tuning
from ultralytics import YOLO

model = YOLO('yolov8n.pt')   # Start from nano pretrained weights

results = model.train(
    data='/content/drive/MyDrive/vla_rob/yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    device='cuda',
    project='/content/drive/MyDrive/vla_rob/checkpoints',
    name='yolov8n_vla',
    exist_ok=True,
    patience=20,         # early stopping if no mAP50 improvement for 20 epochs
    augment=True,        # default Mosaic/MixUp augmentation
    degrees=15,          # random rotation augmentation
    fliplr=0.5,
    flipud=0.0,          # objects on a table shouldn't be upside down
    hsv_h=0.015,         # slight hue shift for color robustness
    hsv_s=0.7,
    hsv_v=0.4,
)

# Export to TorchScript for RPi 5 deployment
model_best = YOLO('/content/drive/MyDrive/vla_rob/checkpoints/yolov8n_vla/weights/best.pt')
model_best.export(format='torchscript', imgsz=640, optimize=True)

print("mAP50:", results.results_dict['metrics/mAP50(B)'])
print("mAP50-95:", results.results_dict['metrics/mAP50-95(B)'])
```

### 2.5 YOLOv8 Wrapper Module

```python
# perception/yolo_detector.py
import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Optional

CLASS_NAMES = ['red_block', 'blue_block', 'yellow_block', 'cylinder', 'tray']

class Detection:
    def __init__(self, class_name: str, confidence: float, bbox_xyxy: np.ndarray):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox_xyxy  = bbox_xyxy  # [x1, y1, x2, y2]

    @property
    def centroid(self) -> Tuple[float, float]:
        return (
            (self.bbox_xyxy[0] + self.bbox_xyxy[2]) / 2.0,
            (self.bbox_xyxy[1] + self.bbox_xyxy[3]) / 2.0,
        )

class YOLODetector:
    CONF_THRESHOLD = 0.5

    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, rgb_frame: np.ndarray) -> List[Detection]:
        results = self.model(rgb_frame, conf=self.CONF_THRESHOLD, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_idx = int(box.cls[0])
                detections.append(Detection(
                    class_name  = CLASS_NAMES[cls_idx],
                    confidence  = float(box.conf[0]),
                    bbox_xyxy   = box.xyxy[0].cpu().numpy(),
                ))
        return detections

    def match_instruction(self, detections: List[Detection],
                          instruction: str) -> Optional[Detection]:
        """Return highest-confidence detection whose class name appears in instruction."""
        matches = [d for d in detections if d.class_name.replace('_', ' ') in instruction.lower()]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)
```

**Expected Output of Phase 2:**
- `checkpoints/yolov8n_vla/weights/best.pt` saved to Google Drive.
- Validation mAP50 ≥ 0.90 on the 5 object classes.
- `yolo_detector.py` runs on RPi 5 at 18–25ms per frame on a 640×480 overhead image.
- Spot-check: run `detector.detect(frame)` on 10 test images and visually confirm correct bounding boxes.

---

## Phase 3 — Dataset Pipeline: HDF5 Loading, Skill Segmentation, Augmentation

**Goal:** A complete dataset pipeline that reads Team's raw teleoperation recordings, applies skill segmentation, and produces augmented training samples ready for VLA fine-tuning.

### 3.1 Understanding Team's Raw Recording Format

Team records teleoperation demonstrations as grouped HDF5 files. Each file has:
- `/telemetry/*` datasets: servo state, ToF, IMU summary, contact oracle, safety clamp status, checksums, and Teensy timestamps
- `/video/rgb_frames`: overhead RGB frames at 30fps
- `/video/frame_timestamps_us`: uint64 frame timestamps in the same microsecond timebase as telemetry
- `/metadata` attributes: instruction, task_type, demo_id, collection metadata

File naming: `demos/demo_001_pick_red_block.h5`, ..., `demos/demo_030_sort_color.h5`.

### 3.2 HDF5 Dataset Reader

```python
# dataset/hdf5_reader.py
import h5py
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

TELEMETRY_DTYPE = np.dtype([
    ('timestamp_us',     np.uint32),
    ('servo_pos',        np.float32, (5,)),
    ('servo_load',       np.float32, (5,)),
    ('servo_speed',      np.float32, (5,)),
    ('servo_temp',       np.float32, (5,)),
    ('tof_grid',         np.uint16,  (64,)),
    ('tof_timestamp_us', np.uint32),
    ('tof_resolution',   np.uint8),
    ('tof_valid',        np.uint8),
    ('imu_gyro',         np.float32, (3,)),
    ('imu_accel',        np.float32, (3,)),
    ('contact_flag',     np.uint8),
    ('contact_rms',      np.float32),
    ('safety_clamped',   np.uint8),
    ('checksum',         np.uint16),
])

def load_demo(h5_path: str) -> Dict[str, Any]:
    with h5py.File(h5_path, 'r') as f:
        tg = f['telemetry']
        n = len(tg['timestamps_us'])
        telemetry = np.zeros(n, dtype=TELEMETRY_DTYPE)
        telemetry['timestamp_us']     = tg['timestamps_us'][()]
        telemetry['servo_pos']        = tg['servo_pos'][()]
        telemetry['servo_load']       = tg['servo_load'][()]
        telemetry['servo_speed']      = tg['servo_speed'][()]
        telemetry['servo_temp']       = tg['servo_temp'][()]
        telemetry['tof_grid']         = tg['tof_grid'][()]
        telemetry['tof_timestamp_us'] = tg['tof_timestamp_us'][()]
        telemetry['tof_resolution']   = tg['tof_resolution'][()]
        telemetry['tof_valid']        = tg['tof_valid'][()]
        telemetry['imu_gyro']         = tg['imu_gyro'][()]
        telemetry['imu_accel']        = tg['imu_accel'][()]
        telemetry['contact_flag']     = tg['contact_flag'][()]
        telemetry['contact_rms']      = tg['contact_rms'][()]
        telemetry['safety_clamped']   = tg['safety_clamped'][()]
        telemetry['checksum']         = tg['checksum'][()]

        rgb_frames  = f['video/rgb_frames'][()]
        frame_ts    = f['video/frame_timestamps_us'][()]
        meta        = f['metadata'].attrs
        instruction = meta['instruction']
        task_type   = meta['task_type']

    return {
        'telemetry':    telemetry,      # shape (N,)
        'rgb_frames':   rgb_frames,     # shape (N_cam, 480, 640, 3)
        'frame_ts':     frame_ts,       # shape (N_cam,)
        'instruction':  instruction,
        'task_type':    task_type,
        'path':         h5_path,
    }

def load_all_demos(dataset_dir: str) -> List[Dict]:
    paths = sorted(Path(dataset_dir).glob('demo_*.h5'))
    print(f"Found {len(paths)} demonstrations")
    return [load_demo(str(p)) for p in paths]

def get_logical_joints(telemetry_row) -> np.ndarray:
    """Extract 4D logical joint state from a single telemetry row."""
    j0 = telemetry_row['servo_pos'][0]
    j1 = (telemetry_row['servo_pos'][1] + telemetry_row['servo_pos'][2]) / 2.0
    j2 = telemetry_row['servo_pos'][3]
    j3 = telemetry_row['servo_pos'][4]
    return np.array([j0, j1, j2, j3], dtype=np.float32)

def get_nearest_frame(frame_ts: np.ndarray, query_us: int,
                       rgb_frames: np.ndarray) -> np.ndarray:
    """Return RGB frame whose timestamp is nearest to query_us."""
    idx = np.argmin(np.abs(frame_ts.astype(np.int64) - int(query_us)))
    return rgb_frames[idx]
```

### 3.3 Skill Segmentation Algorithm

This is the core algorithm that converts raw demonstration trajectories into labeled segments. **Calibrate the thresholds from your first 5 demonstrations** before running on all 30.

```python
# dataset/skill_segmenter.py
import numpy as np
from typing import List, Dict

# Calibration constants — measure from first 5 demos, then fix
THRESH = {
    'load_contact':  0.30,   # normalized gripper load → contact
    'vel_stop':      5.0,    # deg/s arm velocity below this = arm stationary
    'lift_j1_angle': 45.0,   # degrees — shoulder angle must exceed this for LIFT
    'lift_height':   0.08,   # meters — end-effector height for LIFT confirmation
    'tof_approach':  0.05,   # meters — wrist ToF center zone below this → entering GRASP
    'median_window': 5,      # timestep smoothing window (100ms at 50Hz)
}

DH = [(0, 0.065, 0, 90), (0, 0, 0.130, 0), (0, 0, 0.190, 0)]

def forward_kinematics_z(j0_deg, j1_deg, j2_deg) -> float:
    """Return approximate end-effector height (z) above table."""
    import math
    q1 = math.radians(j1_deg)
    q2 = math.radians(j2_deg)
    # Planar height: d1 + L1*sin(q1) + L2*sin(q1+q2)
    z = DH[0][1] + DH[1][2]*math.sin(q1) + DH[2][2]*math.sin(q1+q2)
    return z

def label_timestep(joint_pos, joint_vel, joint_load, imu_contact,
                   wrist_tof_z_m) -> str:
    j1_angle  = joint_pos[1]
    grip_load = joint_load[3]
    arm_vel   = np.sqrt(np.mean(joint_vel[:4]**2))
    ee_height = forward_kinematics_z(joint_pos[0], joint_pos[1], joint_pos[2])

    if (not imu_contact
            and grip_load < THRESH['load_contact']
            and wrist_tof_z_m > THRESH['tof_approach']):
        return 'REACH'

    elif (j1_angle > THRESH['lift_j1_angle']
          and ee_height > THRESH['lift_height']
          and grip_load > THRESH['load_contact']):
        return 'LIFT'

    elif (imu_contact
          or wrist_tof_z_m <= THRESH['tof_approach']
          or (grip_load > THRESH['load_contact'] and arm_vel < THRESH['vel_stop'])):
        return 'GRASP'

    else:
        return 'PLACE'

def segment_demo(demo: Dict) -> List[Dict]:
    """
    Returns list of timestep dicts with skill label attached.
    Applies median filter to remove single-sample noise at skill boundaries.
    """
    from scipy.signal import medfilt

    tel = demo['telemetry']
    N   = len(tel)
    raw_labels = []

    for i in range(N):
        row       = tel[i]
        joint_pos = get_logical_joints(row)
        joint_vel  = get_logical_speeds(row)
        joint_load = get_logical_loads(row)

        imu_contact  = bool(row['contact_flag'])
        tof_center   = [row['tof_grid'][3*8+3], row['tof_grid'][3*8+4],
                        row['tof_grid'][4*8+3], row['tof_grid'][4*8+4]]
        tof_valid    = [z for z in tof_center if 20 < z < 600]
        tof_z_m      = float(np.mean(tof_valid)) / 1000.0 if row['tof_valid'] and tof_valid else 0.3

        raw_labels.append(label_timestep(joint_pos, joint_vel, joint_load,
                                          imu_contact, tof_z_m))

    SKILL_INT = {'REACH': 0, 'GRASP': 1, 'LIFT': 2, 'PLACE': 3}
    INT_SKILL = {0: 'REACH', 1: 'GRASP', 2: 'LIFT', 3: 'PLACE'}
    int_labels   = np.array([SKILL_INT[s] for s in raw_labels])
    smooth_labels = medfilt(int_labels.astype(float), kernel_size=THRESH['median_window'])
    smooth_labels = np.round(smooth_labels).astype(int)

    timesteps = []
    for i in range(N):
        row       = tel[i]
        rgb_frame = get_nearest_frame(demo['frame_ts'], row['timestamp_us'],
                                       demo['rgb_frames'])
        timesteps.append({
            'joint_pos':    get_logical_joints(row),
            'joint_vel':    get_logical_speeds(row),
            'joint_load':   get_logical_loads(row),
            'contact_flag': bool(row['contact_flag']),
            'contact_rms':  float(row['contact_rms']),
            'tof_grid':     row['tof_grid'].copy(),
            'imu_gyro':     row['imu_gyro'].copy(),
            'imu_accel':    row['imu_accel'].copy(),
            'rgb_frame':    rgb_frame,
            'skill_label':  INT_SKILL[smooth_labels[i]],
            'skill_int':    smooth_labels[i],
            'instruction':  demo['instruction'],
            'timestamp_us': int(row['timestamp_us']),
        })

    return timesteps

def get_logical_joints(row) -> np.ndarray:
    j0 = row['servo_pos'][0]
    j1 = (row['servo_pos'][1] + row['servo_pos'][2]) / 2.0
    j2 = row['servo_pos'][3]
    j3 = row['servo_pos'][4]
    return np.array([j0, j1, j2, j3], dtype=np.float32)

def get_logical_speeds(row) -> np.ndarray:
    return np.array([
        row['servo_speed'][0],
        (row['servo_speed'][1] + row['servo_speed'][2]) / 2.0,
        row['servo_speed'][3],
        row['servo_speed'][4],
    ], dtype=np.float32)

def get_logical_loads(row) -> np.ndarray:
    return np.array([
        row['servo_load'][0],
        (row['servo_load'][1] + row['servo_load'][2]) / 2.0,
        row['servo_load'][3],
        row['servo_load'][4],  # gripper load, critical for GRASP/LIFT labels
    ], dtype=np.float32)
```

### 3.4 Data Augmentation

```python
# dataset/augmentation.py
import numpy as np
import cv2
from typing import Dict

RNG = np.random.default_rng(42)

def augment_sample(sample: Dict, aug_id: int) -> Dict:
    """
    Applies one of four augmentation strategies to a single timestep sample.
    aug_id 0: original (no augmentation)
    aug_id 1: joint noise
    aug_id 2: load noise + joint noise
    aug_id 3: horizontal image flip + joint sign flip for J0
    """
    s = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in sample.items()}

    if aug_id == 0:
        return s

    # Always add small joint noise for aug_id >= 1
    s['joint_pos'] = s['joint_pos'] + RNG.normal(0, 0.15, size=4).astype(np.float32)

    if aug_id >= 2:
        s['joint_load'] = np.clip(
            s['joint_load'] + RNG.normal(0, 0.03, size=4).astype(np.float32),
            0.0, 1.0
        )

    if aug_id == 3:
        # Horizontal flip: mirror workspace left-right
        # J0 (base yaw) flips sign; other joints unchanged
        s['rgb_frame'] = cv2.flip(s['rgb_frame'], 1)
        s['joint_pos'][0] = -s['joint_pos'][0]

    return s

def build_training_set(all_timesteps, augmentation_factor: int = 4):
    """
    Augments dataset to produce augmentation_factor × original size.
    Expected: 12,000 raw timesteps × 4 aug = ~48,000 samples before subsampling.
    Subsample by keeping every 6th timestep: ~800 unique temporal positions × 4 aug = 3,200 samples.
    """
    subsampled = all_timesteps[::6]   # ~2,000 samples from 30 demos
    augmented  = []
    for sample in subsampled:
        for aug_id in range(augmentation_factor):
            augmented.append(augment_sample(sample, aug_id))
    return augmented
```

### 3.5 Verify Segmentation Quality

Before training, manually verify skill boundaries from the first 5 demos:

```python
# verify_segmentation.py
import matplotlib.pyplot as plt
from dataset.hdf5_reader import load_all_demos
from dataset.skill_segmenter import segment_demo

COLORS = {'REACH': 'blue', 'GRASP': 'orange', 'LIFT': 'green', 'PLACE': 'red'}

demos = load_all_demos('demos/')
for demo_idx in range(min(5, len(demos))):
    steps = segment_demo(demos[demo_idx])
    labels = [s['skill_label'] for s in steps]
    times  = [s['timestamp_us']/1e6 for s in steps]

    fig, ax = plt.subplots(figsize=(14, 2))
    for i, (t, lbl) in enumerate(zip(times, labels)):
        ax.bar(t, 1, width=0.02, color=COLORS[lbl], alpha=0.7)
    ax.set_title(f"Demo {demo_idx}: {demos[demo_idx]['instruction']}")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'verify_seg_demo{demo_idx}.png')
    plt.close()
    print(f"Demo {demo_idx}: {len(steps)} timesteps, saved verify_seg_demo{demo_idx}.png")
```

**Acceptance criterion:** Skill boundaries make physical sense — REACH covers arm motion toward object, GRASP covers gripper closing, LIFT covers arm rising with object, PLACE covers transport and release. Boundary placement error < 10% by visual inspection on demos 6–15.

**Expected Output of Phase 3:**
- `segment_demo()` runs on all 30 demos without errors.
- `verify_seg_demo0.png` through `verify_seg_demo4.png` show plausible skill boundaries.
- `build_training_set()` produces ~3,000–4,000 augmented training samples.
- Skill distribution is reasonably balanced (none of the 4 skills is <10% of samples).

---

## Phase 4 — VLA Fine-Tuning: SmolVLA-450M + LoRA

**Goal:** A fine-tuned SmolVLA-450M (or Octo-small fallback) checkpoint that predicts correct skill tokens and joint deltas from overhead RGB + instruction + sensor state. Checkpoint exported to Google Drive for RPi 5 deployment.

### 4.1 Model Architecture Decision Tree

Before starting training, benchmark **SmolVLA-450M inference time on RPi 5 CPU** (Section 5 of this workplan). If it exceeds 110ms per step, switch to Octo-small. Do not commit to SmolVLA without a latency measurement.

### 4.2 PyTorch Dataset Class

```python
# dataset/vla_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class VLADataset(Dataset):
    """
    Wraps augmented training samples for SmolVLA/Octo fine-tuning.
    Each sample produces one (observation, action_chunk) pair.
    Action chunk: 8 steps of 4D joint delta from current timestep.
    """
    CHUNK_SIZE = 8

    def __init__(self, timesteps, language_encoder, normalize_joints=True):
        self.timesteps  = timesteps
        self.encoder    = language_encoder
        self.normalize  = normalize_joints

        # Joint normalization bounds — from arm_config.yaml
        self.joint_min = np.array([-150., -90., -120.,  0.], dtype=np.float32)
        self.joint_max = np.array([ 150.,  90.,  120., 90.], dtype=np.float32)

    def __len__(self):
        return len(self.timesteps) - self.CHUNK_SIZE

    def __getitem__(self, idx):
        current = self.timesteps[idx]
        future  = self.timesteps[idx:idx + self.CHUNK_SIZE]

        # RGB: resize to 256×256, normalize to [0,1]
        rgb = cv2.resize(current['rgb_frame'], (256, 256)).astype(np.float32) / 255.0
        rgb = np.transpose(rgb, (2, 0, 1))   # CHW

        # Joint state (normalized)
        joint_state = current['joint_pos'].copy()
        if self.normalize:
            joint_state = (joint_state - self.joint_min) / (self.joint_max - self.joint_min)
            joint_state = np.clip(joint_state, 0.0, 1.0)

        # Skill (one-hot)
        skill_onehot = np.zeros(4, dtype=np.float32)
        skill_onehot[current['skill_int']] = 1.0

        # Contact quality and wrist ToF scalar features
        contact_rms  = np.array([current['contact_rms']], dtype=np.float32)
        tof_center = [current['tof_grid'][3*8+3], current['tof_grid'][3*8+4],
                      current['tof_grid'][4*8+3], current['tof_grid'][4*8+4]]
        tof_valid = [z for z in tof_center if 20 < z < 600]
        tof_center_mm = float(np.mean(tof_valid)) if tof_valid else 300.0
        tof_scalar = np.array([tof_center_mm / 1000.0], dtype=np.float32)  # convert to meters

        # Language embedding (cached)
        lang_emb = self.encoder.encode(current['instruction'])

        # Action chunk: 8-step joint delta sequence, shape (8, 4)
        chunk_joints = np.stack([s['joint_pos'] for s in future], axis=0)  # (8, 4)
        delta_joints = chunk_joints - current['joint_pos'][np.newaxis, :]  # (8, 4)

        # Target skill label (single integer for current step)
        skill_label = current['skill_int']

        return {
            'rgb':          torch.from_numpy(rgb),
            'joint_state':  torch.from_numpy(joint_state),
            'skill_onehot': torch.from_numpy(skill_onehot),
            'contact_rms':  torch.from_numpy(contact_rms),
            'tof_scalar':   torch.from_numpy(tof_scalar),
            'lang_emb':     torch.from_numpy(lang_emb),
            'delta_joints': torch.from_numpy(delta_joints),
            'skill_label':  torch.tensor(skill_label, dtype=torch.long),
        }
```

### 4.3 Language Encoder (Shared by Dataset and Inference)

```python
# language/language_encoder.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

class LanguageEncoder:
    """
    Instruction embedding cache.
    SmolVLA deployments may use the backbone's native tokenizer directly; this
    T5-small path is retained for Octo fallback and any auxiliary dataset features.
    Repeated instructions are cached so runtime encoding cost is effectively zero.
    """

    MODEL_NAME = 'google/t5-small'

    def __init__(self, device='cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model     = AutoModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()
        self.device = device
        self._cache = {}

    @torch.no_grad()
    def encode(self, instruction: str) -> np.ndarray:
        if instruction in self._cache:
            return self._cache[instruction]

        tokens  = self.tokenizer(instruction, return_tensors='pt',
                                  padding=True, truncation=True, max_length=64)
        tokens  = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.model.encoder(**tokens)
        emb     = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        self._cache[instruction] = emb
        return emb
```

### 4.4 LoRA Configuration and Model Wrapping

```python
# In VLA_Training.ipynb — Cell: Model Setup
from transformers import AutoModelForSeq2SeqLM, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn

# ---- Option A: SmolVLA-450M ----
# SmolVLA is available via HuggingFace lerobot library
# pip install lerobot
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig

smolvla_config = SmolVLAConfig()
backbone = SmolVLAPolicy(smolvla_config)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'v_proj', 'fc1', 'fc2'],
    lora_dropout=0.05,
    bias='none',
)
backbone = get_peft_model(backbone, lora_config)
backbone.print_trainable_parameters()
# Expected: ~2-5M trainable out of ~450M total

# ---- Option B: Octo-small (fallback) ----
# Use if SmolVLA RPi5 latency > 110ms
# pip install octo
# from octo.model import OctoModel
# backbone = OctoModel.load_pretrained('hf://rail-berkeley/octo-small')
```

### 4.5 Dual-Head Policy Wrapper

```python
# In VLA_Training.ipynb — Cell: Dual-Head Policy
class VLAPolicy(nn.Module):
    """
    Wraps SmolVLA (or Octo) backbone with:
      - 4-class skill head (REACH/GRASP/LIFT/PLACE)
      - 8×4 action chunk head (joint deltas)
    """
    SKILL_CLASSES = 4
    CHUNK_SIZE    = 8
    JOINT_DIM     = 4
    HIDDEN_DIM    = 512   # adjust to backbone output dim

    def __init__(self, backbone, backbone_output_dim: int):
        super().__init__()
        self.backbone   = backbone
        self.skill_head = nn.Sequential(
            nn.Linear(backbone_output_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.SKILL_CLASSES),
        )
        self.action_head = nn.Sequential(
            nn.Linear(backbone_output_dim + self.SKILL_CLASSES, 512),
            nn.GELU(),
            nn.Linear(512, self.CHUNK_SIZE * self.JOINT_DIM),
        )

    def forward(self, batch):
        features    = self.backbone(batch)          # (B, backbone_output_dim)
        skill_logits = self.skill_head(features)    # (B, 4)
        action_in   = torch.cat([features, skill_logits.detach()], dim=-1)
        delta_flat  = self.action_head(action_in)   # (B, 32)
        delta_joints = delta_flat.view(-1, self.CHUNK_SIZE, self.JOINT_DIM)  # (B, 8, 4)
        return skill_logits, delta_joints
```

### 4.6 Training Loop

```python
# In VLA_Training.ipynb — Cell: Training Loop
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss weights (from main report Section 13.3)
LAMBDA_SKILL   = 1.0
LAMBDA_ACTION  = 0.5
LAMBDA_CONTACT = 0.3   # contact quality prediction (optional extension)

def compute_loss(skill_logits, delta_joints_pred, skill_labels, delta_joints_gt):
    L_skill  = F.cross_entropy(skill_logits, skill_labels)
    L_action = F.mse_loss(delta_joints_pred, delta_joints_gt)
    return LAMBDA_SKILL * L_skill + LAMBDA_ACTION * L_action

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        skill_logits, delta_pred = model(batch)
        loss = compute_loss(skill_logits, delta_pred,
                            batch['skill_label'], batch['delta_joints'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total, val_loss = 0, 0, 0.0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        skill_logits, delta_pred = model(batch)
        loss = compute_loss(skill_logits, delta_pred,
                            batch['skill_label'], batch['delta_joints'])
        val_loss += loss.item()
        pred_skill = skill_logits.argmax(dim=-1)
        correct   += (pred_skill == batch['skill_label']).sum().item()
        total     += batch['skill_label'].size(0)
    return val_loss / len(loader), correct / total

# --- Instantiate ---
lang_enc  = LanguageEncoder(device=DEVICE)
all_data  = build_training_set(all_timesteps, augmentation_factor=4)
dataset   = VLADataset(all_data, lang_enc)
n_val     = int(0.15 * len(dataset))
train_ds, val_ds = random_split(dataset, [len(dataset)-n_val, n_val])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=2)

model     = VLAPolicy(backbone, backbone_output_dim=512).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)

best_skill_acc = 0.0
for epoch in range(1, 6):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss, skill_acc = evaluate(model, val_loader)
    scheduler.step()
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  skill_acc={skill_acc:.3f}")

    if skill_acc > best_skill_acc:
        best_skill_acc = skill_acc
        model.backbone.save_pretrained(f'{CHECKPOINT_DIR}/smolvla_lora_best')
        torch.save(model.skill_head.state_dict(),  f'{CHECKPOINT_DIR}/skill_head_best.pt')
        torch.save(model.action_head.state_dict(), f'{CHECKPOINT_DIR}/action_head_best.pt')
        print(f"  → Saved best checkpoint (skill_acc={skill_acc:.3f})")
```

**Training targets:**
- Skill classification accuracy ≥ 75% on validation set
- Action MSE loss < 5.0 degrees² per step
- Training time < 2 hours on Colab A100
- If validation skill accuracy plateaus below 60% after 5 epochs, expand to 100 demonstrations before tuning hyperparameters

**Expected Output of Phase 4:**
- `checkpoints/smolvla_lora_best/` saved to Google Drive with LoRA adapter weights.
- `skill_head_best.pt` and `action_head_best.pt` saved.
- Training log showing skill_acc ≥ 0.75 by epoch 5.
- If SmolVLA is unavailable, the same pipeline runs with Octo-small as the backbone.

---

## Phase 5 — RPi 5 Benchmarking and Model Export

**Goal:** Measure actual inference latency on the RPi 5 for the complete pipeline. Select export format (TorchScript vs ONNX). Confirm the system can run at 8Hz (≤125ms per step).

### 5.1 Export Checkpoint to TorchScript

```python
# In VLA_Training.ipynb — Cell: Export
import torch

# Load best checkpoint
model.load_state_dict(...)   # reload best weights
model.eval()

# Trace with representative input shapes
dummy_rgb        = torch.zeros(1, 3, 256, 256)
dummy_joint      = torch.zeros(1, 4)
dummy_skill      = torch.zeros(1, 4)
dummy_lang       = torch.zeros(1, 512)
dummy_contact    = torch.zeros(1, 1)
dummy_tof        = torch.zeros(1, 1)

# Export full policy
traced = torch.jit.trace(model.cpu(), (
    {'rgb': dummy_rgb, 'joint_state': dummy_joint, 'skill_onehot': dummy_skill,
     'lang_emb': dummy_lang, 'contact_rms': dummy_contact, 'tof_scalar': dummy_tof}
,))
traced.save(f'{CHECKPOINT_DIR}/vla_policy_traced.pt')
print("TorchScript export complete")
```

### 5.2 RPi 5 Latency Benchmark Script

```python
# benchmark_rpi5.py — run this ON the RPi 5
import torch
import time
import numpy as np

model = torch.jit.load('checkpoints/vla_policy_traced.pt')
model.eval()

def make_dummy():
    return {
        'rgb':          torch.zeros(1, 3, 256, 256),
        'joint_state':  torch.zeros(1, 4),
        'skill_onehot': torch.zeros(1, 4),
        'lang_emb':     torch.zeros(1, 512),
        'contact_rms':  torch.zeros(1, 1),
        'tof_scalar':   torch.zeros(1, 1),
    }

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = model(make_dummy())

# Measure 50 iterations
latencies = []
for _ in range(50):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(make_dummy())
    latencies.append((time.perf_counter() - t0) * 1000)

print(f"VLA inference latency (ms):")
print(f"  Mean:    {np.mean(latencies):.1f}")
print(f"  Median:  {np.median(latencies):.1f}")
print(f"  P95:     {np.percentile(latencies, 95):.1f}")
print(f"  Max:     {np.max(latencies):.1f}")

# Decision rule:
# P95 ≤ 100ms → SmolVLA OK for 8Hz loop with comfortable headroom.
# 100ms < P95 ≤ 110ms → SmolVLA only if full pipeline P95 still ≤ 125ms.
# P95 > 110ms → switch to Octo-small or a faster quantized checkpoint.
```

### 5.3 Full Pipeline Budget Measurement

```python
# benchmark_full_pipeline.py — measures each stage separately
import time, numpy as np
from ultralytics import YOLO
from language.language_encoder import LanguageEncoder

def measure(fn, n=20):
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter()-t)*1000)
    return np.mean(times), np.percentile(times, 95)

# Stage timings — target budget (ms):
# USB serial read:      ≤ 3ms
# Camera frame grab:    ≤ 0.5ms
# YOLOv8-nano:          ≤ 25ms
# Pose estimation:      ≤ 3ms
# Language encoding:    ≤ 1ms (cached)
# VLA inference:        ≤ 100ms
# Skill transition:     ≤ 0.1ms
# IK + safety:          ≤ 7ms
# USB serial write:     ≤ 1.5ms
# TOTAL TARGET:         ≤ 125ms
```

**Decision table:**

| VLA P95 latency | Decision |
|-----------------|----------|
| ≤ 100ms | Deploy SmolVLA-450M + LoRA |
| 100–110ms | Deploy SmolVLA only if full pipeline P95 is still ≤ 125ms |
| > 110ms | Try INT8 quantization once; if still > 110ms, switch to Octo-small + LoRA |

### 5.4 INT8 Quantization (if needed)

```python
# quantize.py
import torch

model = torch.jit.load('checkpoints/vla_policy_traced.pt')
quantized = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
torch.jit.save(torch.jit.script(quantized), 'checkpoints/vla_policy_int8.pt')
print("INT8 export complete")
```

**Expected Output of Phase 5:**
- Benchmark report printed: VLA P95 latency on RPi 5.
- Final deployment checkpoint selected: either `vla_policy_traced.pt` or `vla_policy_int8.pt` or Octo fallback.
- Full pipeline budget measured: total ≤ 125ms confirmed.
- `model_config.yaml` updated with final model path and confirmed latency numbers.

---

## Phase 6 — Full Inference Pipeline

**Goal:** A complete, running Python inference pipeline on the RPi 5 that reads telemetry from Team's Teensy, runs all perception and planning stages, and sends valid command packets at 8Hz.

### 6.1 USB Serial Communication Module

```python
# comms/teensy_serial.py
import serial
import struct
import numpy as np
import threading
import time
from typing import Optional

TELEMETRY_DTYPE = np.dtype([
    ('timestamp_us',     np.uint32),
    ('servo_pos',        np.float32, (5,)),
    ('servo_load',       np.float32, (5,)),
    ('servo_speed',      np.float32, (5,)),
    ('servo_temp',       np.float32, (5,)),
    ('tof_grid',         np.uint16,  (64,)),
    ('tof_timestamp_us', np.uint32),
    ('tof_resolution',   np.uint8),
    ('tof_valid',        np.uint8),
    ('imu_gyro',         np.float32, (3,)),
    ('imu_accel',        np.float32, (3,)),
    ('contact_flag',     np.uint8),
    ('contact_rms',      np.float32),
    ('safety_clamped',   np.uint8),
    ('checksum',         np.uint16),
])
TELEMETRY_SIZE = 250

COMMAND_DTYPE = np.dtype([
    ('target_arm',      np.float32, (3,)),
    ('skill_state',     np.uint8),
    ('execute',         np.uint8),
    ('gripper_command', np.float32),
    ('emergency_stop',  np.uint8),
    ('checksum',        np.uint8),
])
COMMAND_SIZE = 20

class TeensySerial:
    def __init__(self, port: str = '/dev/ttyACM0', baud: int = 2_000_000):
        self.ser  = serial.Serial(port, baud, timeout=0.1)
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = True
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def _rx_loop(self):
        buf = bytearray()
        while self._running:
            data = self.ser.read(TELEMETRY_SIZE)
            if not data:
                continue
            buf.extend(data)
            while len(buf) >= TELEMETRY_SIZE:
                chunk = bytes(buf[:TELEMETRY_SIZE])
                buf   = buf[TELEMETRY_SIZE:]
                parsed = np.frombuffer(chunk, dtype=TELEMETRY_DTYPE)
                if self._verify_checksum(parsed[0]):
                    with self._lock:
                        self._latest = parsed[0]

    def _verify_checksum(self, t) -> bool:
        raw = t.tobytes()
        computed = sum(raw[:-2]) & 0xFFFF
        return computed == int(t['checksum'])

    def latest_telemetry(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    def send_command(self, target_arm: np.ndarray, skill_state: int,
                     execute: int, gripper_cmd: float,
                     emergency_stop: int = 0):
        cmd = np.zeros(1, dtype=COMMAND_DTYPE)
        cmd['target_arm'][0]      = target_arm[:3].astype(np.float32)
        cmd['skill_state'][0]     = skill_state
        cmd['execute'][0]         = execute
        cmd['gripper_command'][0] = gripper_cmd
        cmd['emergency_stop'][0]  = emergency_stop
        raw = cmd.tobytes()[:COMMAND_SIZE-1]
        checksum = sum(raw) & 0xFF
        payload  = raw + bytes([checksum])
        with self._lock:
            self.ser.write(payload)

    def close(self):
        self._running = False
        self.ser.close()
```

### 6.2 Camera Manager

```python
# perception/camera_manager.py
import threading
import numpy as np
from picamera2 import Picamera2

class CameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"},
            controls={"FrameRate": fps}
        ))
        self._frame = None
        self._lock  = threading.Lock()
        self._running = True
        self.cam.start()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        while self._running:
            frame = self.cam.capture_array()
            with self._lock:
                self._frame = frame

    def latest_frame(self) -> np.ndarray:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def close(self):
        self._running = False
        self.cam.stop()
```

### 6.3 Pose Estimation Module

```python
# perception/pose_estimation.py
import numpy as np
import yaml
from typing import Optional, Tuple

class PoseEstimator:
    def __init__(self, calib_dir: str):
        with open(f'{calib_dir}/camera_intrinsics.yaml') as f:
            cam = yaml.safe_load(f)
        with open(f'{calib_dir}/overhead_height.yaml') as f:
            height = yaml.safe_load(f)
        with open(f'{calib_dir}/wrist_tof_offset.yaml') as f:
            tof_off = yaml.safe_load(f)
        with open(f'{calib_dir}/camera_to_base_transform.yaml') as f:
            T = yaml.safe_load(f)

        self.K           = np.array(cam['K'], dtype=np.float64).reshape(3, 3)
        self.Z_table     = float(height['Z_table_m'])
        self.wrist_off   = float(tof_off['wrist_to_sensor_offset_mm'])
        self.T_cam_base  = np.array(T['T_cam_base'], dtype=np.float64).reshape(4, 4)

    def overhead_xy(self, centroid: Tuple[float, float]) -> Tuple[float, float]:
        u_c, v_c = centroid
        fx, fy   = self.K[0, 0], self.K[1, 1]
        cx, cy   = self.K[0, 2], self.K[1, 2]
        X = (u_c - cx) * self.Z_table / fx
        Y = (v_c - cy) * self.Z_table / fy
        return X, Y

    def wrist_tof_z(self, tof_grid: np.ndarray, tof_valid: bool) -> Optional[float]:
        if not tof_valid:
            return None
        center = [tof_grid[3*8+3], tof_grid[3*8+4],
                  tof_grid[4*8+3], tof_grid[4*8+4]]
        valid  = [z for z in center if 20 < z < 600]
        if not valid:
            return None
        Z_mm = np.mean(valid) - self.wrist_off
        return Z_mm / 1000.0

    def compute_pick_pose(self, centroid, tof_grid, tof_valid) -> np.ndarray:
        X_cam, Y_cam = self.overhead_xy(centroid)
        Z_grasp = self.wrist_tof_z(tof_grid, tof_valid)
        if Z_grasp is None:
            Z_grasp = 0.02   # fallback: assume 20mm object height

        pose_cam  = np.array([X_cam, Y_cam, self.Z_table - Z_grasp, 1.0])
        pose_base = self.T_cam_base @ pose_cam
        return pose_base[:3]   # (X, Y, Z) in robot base frame, meters
```

### 6.4 IK Solver

```python
# planning/ik_solver.py
import numpy as np
from typing import Optional

DH_D1 = 0.065   # base height (m)
DH_A2 = 0.130   # shoulder link length (m)
DH_A3 = 0.190   # elbow link length (m)

def inverse_kinematics(x: float, y: float, z: float) -> Optional[np.ndarray]:
    """
    Closed-form 3-DOF IK for the spatial arm chain (J0, J1, J2).
    Returns [J0, J1, J2] in degrees, or None if out of reach.
    """
    q0 = np.arctan2(y, x)
    r  = np.sqrt(x**2 + y**2)
    z_from_base = z - DH_D1

    D = (r**2 + z_from_base**2 - DH_A2**2 - DH_A3**2) / (2 * DH_A2 * DH_A3)
    if abs(D) > 1.0:
        return None   # out of reach

    q2 = np.arctan2(-np.sqrt(1.0 - D**2), D)   # elbow-down configuration
    q1 = (np.arctan2(z_from_base, r)
          - np.arctan2(DH_A3 * np.sin(q2), DH_A2 + DH_A3 * np.cos(q2)))

    return np.array([np.rad2deg(q0), np.rad2deg(q1), np.rad2deg(q2)], dtype=np.float32)
```

### 6.5 Safety Filter

```python
# planning/safety_filter.py
import numpy as np
from typing import Optional
import yaml

class SafetyFilter:
    WORKSPACE = {
        'x': (-0.38, 0.38),
        'y': (-0.38, 0.38),
        'z': (0.02, 0.35),
    }
    SINGULARITY_THRESH = 1e-4

    def __init__(self, config_path: str):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.joint_min = np.array(cfg['joint_limits_min_deg'], dtype=np.float32)
        self.joint_max = np.array(cfg['joint_limits_max_deg'], dtype=np.float32)

    def check_workspace(self, pos: np.ndarray) -> np.ndarray:
        mins = np.array([self.WORKSPACE['x'][0], self.WORKSPACE['y'][0], self.WORKSPACE['z'][0]])
        maxs = np.array([self.WORKSPACE['x'][1], self.WORKSPACE['y'][1], self.WORKSPACE['z'][1]])
        return np.clip(pos, mins, maxs)

    def check_singularity(self, joints_deg: np.ndarray) -> np.ndarray:
        q1 = np.deg2rad(joints_deg[1])
        q2 = np.deg2rad(joints_deg[2])
        det = abs(np.sin(q2))   # simplified Jacobian determinant for 2-link planar
        if det < self.SINGULARITY_THRESH:
            joints_deg = joints_deg.copy()
            joints_deg[1] += 2.0   # nudge shoulder to escape singularity
        return joints_deg

    def clamp_joints(self, joints_deg: np.ndarray) -> np.ndarray:
        return np.clip(joints_deg, self.joint_min[:len(joints_deg)],
                       self.joint_max[:len(joints_deg)])

    def apply(self, joints_deg: np.ndarray) -> np.ndarray:
        j = self.check_singularity(joints_deg)
        return self.clamp_joints(j)
```

### 6.6 VLA Policy Runtime Wrapper

```python
# vla/vla_policy.py
import torch
import numpy as np
from language.language_encoder import LanguageEncoder
import cv2

class VLARuntime:
    CHUNK_SIZE = 8

    def __init__(self, model_path: str, lang_encoder: LanguageEncoder):
        self.model    = torch.jit.load(model_path)
        self.model.eval()
        self.encoder  = lang_encoder
        self._chunk_buffer = None   # last predicted action chunk

    def predict(self, rgb_frame: np.ndarray, joint_state_4d: np.ndarray,
                skill_onehot: np.ndarray, instruction: str,
                contact_rms: float, tof_z_m: float):
        rgb   = cv2.resize(rgb_frame, (256, 256)).astype(np.float32) / 255.0
        rgb_t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)

        lang_emb = self.encoder.encode(instruction)

        batch = {
            'rgb':          rgb_t,
            'joint_state':  torch.from_numpy(joint_state_4d).unsqueeze(0),
            'skill_onehot': torch.from_numpy(skill_onehot).unsqueeze(0),
            'lang_emb':     torch.from_numpy(lang_emb).unsqueeze(0),
            'contact_rms':  torch.tensor([[contact_rms]]),
            'tof_scalar':   torch.tensor([[tof_z_m]]),
        }

        with torch.no_grad():
            skill_logits, delta_joints = self.model(batch)

        skill_pred    = int(skill_logits.argmax(dim=-1).item())
        delta_step0   = delta_joints[0, 0].numpy()   # first step of 8-step chunk

        self._chunk_buffer = delta_joints[0].numpy()  # (8, 4) for Teensy interpolation
        return skill_pred, delta_step0, self._chunk_buffer
```

### 6.7 Skill State Machine

```python
# vla/skill_predictor.py
SKILL = {'REACH': 0, 'GRASP': 1, 'LIFT': 2, 'PLACE': 3}
SKILL_INV = {v: k for k, v in SKILL.items()}

class SkillStateMachine:
    def __init__(self):
        self.state = SKILL['REACH']
        self.legal_next = {
            SKILL['REACH']: {SKILL['REACH'], SKILL['GRASP']},
            SKILL['GRASP']: {SKILL['GRASP'], SKILL['LIFT']},
            SKILL['LIFT']:  {SKILL['LIFT'],  SKILL['PLACE']},
            SKILL['PLACE']: {SKILL['PLACE']},
        }

    def update(self, predicted_skill: int, contact_flag: bool) -> int:
        """
        Forced transitions take precedence over model predictions.
        IMU contact flag forces GRASP → LIFT immediately.
        """
        # IMU contact triggers forced transition
        if contact_flag and self.state == SKILL['GRASP']:
            self.state = SKILL['LIFT']
            return self.state

        # Accept only one-step legal progressions; never jump REACH directly to PLACE.
        if predicted_skill in self.legal_next[self.state]:
            self.state = predicted_skill

        return self.state

    def reset(self):
        self.state = SKILL['REACH']
```

### 6.8 Main Inference Loop

```python
# main.py
import time, yaml, numpy as np
from comms.teensy_serial       import TeensySerial
from perception.camera_manager import CameraManager
from perception.yolo_detector  import YOLODetector
from perception.pose_estimation import PoseEstimator
from language.language_encoder import LanguageEncoder
from vla.vla_policy            import VLARuntime
from vla.skill_predictor       import SkillStateMachine, SKILL, SKILL_INV
from planning.ik_solver        import inverse_kinematics
from planning.safety_filter    import SafetyFilter

INSTRUCTION = "pick the red block and place it in the tray"
PERIOD_S    = 1.0 / 8.0   # 8Hz inference target

def main():
    teensy  = TeensySerial('/dev/ttyACM0', 2_000_000)
    camera  = CameraManager(640, 480, 30)
    yolo    = YOLODetector('checkpoints/yolov8n_vla/weights/best.pt')
    pose    = PoseEstimator('calibration/')
    lang    = LanguageEncoder()
    vla     = VLARuntime('checkpoints/vla_policy_traced.pt', lang)
    safety  = SafetyFilter('config/arm_config.yaml')
    fsm     = SkillStateMachine()

    skill_onehot = np.zeros(4, dtype=np.float32)
    skill_onehot[0] = 1.0   # start in REACH

    print("Inference loop starting. Press Ctrl+C to stop.")
    try:
        while True:
            t0 = time.monotonic()

            # --- READ ---
            telemetry = teensy.latest_telemetry()
            rgb_frame = camera.latest_frame()
            if telemetry is None or rgb_frame is None:
                time.sleep(0.01)
                continue

            # --- DETECT ---
            detections  = yolo.detect(rgb_frame)
            target_det  = yolo.match_instruction(detections, INSTRUCTION)

            # --- 3D POSE ---
            pick_pose = None
            if target_det is not None:
                pick_pose = pose.compute_pick_pose(
                    target_det.centroid,
                    telemetry['tof_grid'],
                    bool(telemetry['tof_valid'])
                )

            # --- JOINT STATE ---
            j0  = telemetry['servo_pos'][0]
            j1  = (telemetry['servo_pos'][1] + telemetry['servo_pos'][2]) / 2.0
            j2  = telemetry['servo_pos'][3]
            j3  = telemetry['servo_pos'][4]
            joint_state_4d = np.array([j0, j1, j2, j3], dtype=np.float32)

            tof_center = [telemetry['tof_grid'][3*8+3],
                          telemetry['tof_grid'][3*8+4],
                          telemetry['tof_grid'][4*8+3],
                          telemetry['tof_grid'][4*8+4]]
            tof_valid = [z for z in tof_center if 20 < z < 600]
            tof_center_mm = float(np.mean(tof_valid)) if telemetry['tof_valid'] and tof_valid else 300.0
            tof_z_m = tof_center_mm / 1000.0

            # --- VLA INFERENCE ---
            skill_pred, delta_step0, chunk = vla.predict(
                rgb_frame, joint_state_4d, skill_onehot,
                INSTRUCTION, float(telemetry['contact_rms']), tof_z_m
            )

            # --- SKILL TRANSITION ---
            new_skill = fsm.update(skill_pred, bool(telemetry['contact_flag']))
            skill_onehot[:] = 0.0
            skill_onehot[new_skill] = 1.0

            # --- VLA DELTA + OPTIONAL IK ANCHOR + SAFETY ---
            policy_target = joint_state_4d[:3] + delta_step0[:3]
            ik_target = inverse_kinematics(*pick_pose) if pick_pose is not None else None

            if ik_target is not None and new_skill in (SKILL['REACH'], SKILL['PLACE']):
                # Blend learned deltas with geometric pose only when a reliable pose exists.
                target_joints = 0.5 * policy_target + 0.5 * ik_target
            else:
                target_joints = policy_target

            target_joints = safety.apply(target_joints)

            place_reached = (
                new_skill == SKILL['PLACE']
                and np.linalg.norm(target_joints - joint_state_4d[:3]) < 3.0
            )
            if new_skill == SKILL['REACH']:
                gripper_cmd = 0.0      # open while approaching
            elif new_skill == SKILL['PLACE'] and place_reached:
                gripper_cmd = 0.0      # release only at the place target
            else:
                gripper_cmd = 1.0      # hold closed through GRASP/LIFT and travel-to-place

            # --- TRANSMIT ---
            teensy.send_command(
                target_arm    = target_joints,
                skill_state   = new_skill,
                execute       = 1,
                gripper_cmd   = gripper_cmd,
                emergency_stop = 0,
            )

            elapsed = (time.monotonic() - t0) * 1000
            if elapsed > 125:
                print(f"[WARN] Inference overrun: {elapsed:.1f}ms")

            remaining = PERIOD_S - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("Stopping. Sending emergency stop.")
        teensy.send_command(np.zeros(3), 0, 0, 0.0, emergency_stop=1)
    finally:
        teensy.close()
        camera.close()

if __name__ == '__main__':
    main()
```

**Expected Output of Phase 6:**
- `python3 main.py` runs for 60 seconds without crashing.
- Inference loop completes each cycle in ≤ 125ms (check logs for overrun warnings).
- Robot arm responds to `send_command` — servos move toward detected objects.
- `contact_flag` from telemetry correctly reaches the skill state machine and triggers GRASP→LIFT.
- Emergency stop sends correctly when Ctrl+C is pressed.

---

## Phase 7 — Live Dashboard (PyQt6)

**Goal:** A 5-panel real-time dashboard running on the RPi 5 at 10Hz that displays all sensor signals, detections, skill state, and system health.

### 7.1 Dashboard Architecture

The GUI runs in the main thread (required by Qt). All data is accessed through thread-safe shared state updated by the inference thread.

```python
# dashboard/gui.py
import sys, time, collections
import numpy as np
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QGridLayout,
                               QLabel, QPushButton, QVBoxLayout, QHBoxLayout)
from PyQt6.QtCore    import QTimer, Qt
from PyQt6.QtGui     import QImage, QPixmap, QPainter, QPen, QColor, QFont
import pyqtgraph as pg

SKILL_COLORS = {0: '#4488FF', 1: '#FF8800', 2: '#44CC44', 3: '#FF4444'}
SKILL_NAMES  = {0: 'REACH',   1: 'GRASP',   2: 'LIFT',    3: 'PLACE'}

class SharedState:
    """Thread-safe container passed between inference thread and GUI thread."""
    def __init__(self):
        import threading
        self._lock      = threading.Lock()
        self.rgb_frame  = None          # numpy (480, 640, 3)
        self.detections = []
        self.tof_grid   = np.zeros(64, dtype=np.uint16)
        self.contact_rms = 0.0
        self.contact_rms_history = collections.deque(maxlen=200)
        self.servo_load  = np.zeros(5)
        self.servo_load_history = collections.deque(maxlen=200)
        self.skill_state = 0
        self.skill_history = collections.deque(maxlen=200)
        self.inference_ms  = 0.0
        self.safety_clamp  = False
        self.contact_flag  = False
        self.contact_events = []   # list of x indices where contact fired
        self.contact_threshold = 3.5

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self):
        with self._lock:
            return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class VLADashboard(QMainWindow):
    UPDATE_HZ = 10

    def __init__(self, shared: SharedState):
        super().__init__()
        self.shared = shared
        self.setWindowTitle("VLA Robotic Arm — Live Dashboard")
        self.resize(1280, 800)
        self._build_ui()
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(int(1000 / self.UPDATE_HZ))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setSpacing(6)

        # Panel 1: Overhead camera (top-left)
        self.cam_label = QLabel("Overhead Camera")
        self.cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cam_label.setMinimumSize(480, 360)
        self.cam_label.setStyleSheet("background:#111; border:1px solid #444; color:#888;")
        grid.addWidget(self.cam_label, 0, 0, 2, 1)

        # Panel 2: Skill timeline (top-right)
        self.skill_plot = pg.PlotWidget(title="Skill Timeline (REACH/GRASP/LIFT/PLACE)")
        self.skill_plot.setYRange(-0.5, 3.5)
        self.skill_plot.setLabel('left', 'Skill')
        self.skill_plot.setLabel('bottom', 'Time steps')
        self.skill_bars = pg.BarGraphItem(x=[], height=[], width=0.9, brush='#4488FF')
        self.skill_plot.addItem(self.skill_bars)
        grid.addWidget(self.skill_plot, 0, 1)

        # Panel 3: ToF heatmap (middle-left)
        self.tof_plot = pg.ImageView(name="Wrist ToF 8x8 Depth (mm)")
        self.tof_plot.setColorMap(pg.colormap.get('CET-R2'))
        self.tof_plot.ui.roiBtn.hide()
        self.tof_plot.ui.menuBtn.hide()
        grid.addWidget(self.tof_plot, 1, 1)

        # Panel 4: Contact oracle signals (middle-right)
        self.contact_plot = pg.PlotWidget(title="Contact Oracle: IMU RMS & Gripper Load")
        self.contact_plot.addLegend()
        self.imu_curve  = self.contact_plot.plot(pen=pg.mkPen('#AA44FF', width=2), name='IMU RMS (deg/s)')
        self.load_curve = self.contact_plot.plot(pen=pg.mkPen('#FF8800', width=2), name='Gripper Load')
        self.thresh_line = pg.InfiniteLine(pos=3.5, angle=0,
                                           pen=pg.mkPen('#FF4444', style=Qt.PenStyle.DashLine))
        self.contact_plot.addItem(self.thresh_line)
        grid.addWidget(self.contact_plot, 2, 0)

        # Panel 5: System status (bottom)
        self.status_widget = self._build_status_panel()
        grid.addWidget(self.status_widget, 2, 1)

    def _build_status_panel(self):
        w = QWidget()
        w.setStyleSheet("background:#1a1a2e; border:1px solid #444; border-radius:4px;")
        layout = QHBoxLayout(w)
        self.lbl_latency  = self._status_label("Latency", "---ms")
        self.lbl_hz       = self._status_label("Loop Hz", "---")
        self.lbl_skill    = self._status_label("Skill", "REACH")
        self.lbl_safety   = self._status_label("Safety Clamp", "OFF")
        self.lbl_tof_dist = self._status_label("Wrist Z", "---mm")
        self.btn_estop    = QPushButton("EMERGENCY STOP")
        self.btn_estop.setStyleSheet("background:#CC0000; color:white; font-weight:bold; padding:8px;")
        for lbl in [self.lbl_latency, self.lbl_hz, self.lbl_skill,
                    self.lbl_safety, self.lbl_tof_dist]:
            layout.addWidget(lbl)
        layout.addWidget(self.btn_estop)
        return w

    def _status_label(self, title, value):
        lbl = QLabel(f"{title}\n{value}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont("Monospace", 10))
        lbl.setStyleSheet("color:#AAFFAA; padding:4px;")
        return lbl

    def _refresh(self):
        s = self.shared.snapshot()

        # Panel 1: Camera with YOLOv8 boxes
        if s['rgb_frame'] is not None:
            frame = s['rgb_frame'].copy()
            for det in s['detections']:
                x1, y1, x2, y2 = det.bbox_xyxy.astype(int)
                color = (0, 255, 0) if 'red' in det.class_name else (255, 200, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                            (x1, max(y1-4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            h, w, ch = frame.shape
            qimg = QImage(frame.data, w, h, ch*w, QImage.Format.Format_RGB888)
            self.cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(
                self.cam_label.width(), self.cam_label.height(),
                Qt.AspectRatioMode.KeepAspectRatio))

        # Panel 2: Skill timeline bars
        hist = list(s['skill_history'])
        if hist:
            x = np.arange(len(hist))
            brushes = [SKILL_COLORS[sk] for sk in hist]
            self.skill_plot.clear()
            for i, (xi, sk) in enumerate(zip(x, hist)):
                bar = pg.BarGraphItem(x=[xi], height=[sk+0.8], width=0.9,
                                       brush=SKILL_COLORS[sk])
                self.skill_plot.addItem(bar)

        # Panel 3: ToF heatmap 8×8
        tof = s['tof_grid'].reshape(8, 8).astype(np.float32)
        self.tof_plot.setImage(tof, autoLevels=False, levels=(0, 600))

        # Panel 4: IMU + load time series
        imu_hist  = list(s['contact_rms_history'])
        load_hist = list(s['servo_load_history'])
        if imu_hist:
            self.imu_curve.setData(imu_hist)
        if load_hist:
            self.load_curve.setData(load_hist)

        # Panel 5: Status
        self.lbl_latency.setText(f"Latency\n{s['inference_ms']:.1f}ms")
        self.lbl_skill.setText(f"Skill\n{SKILL_NAMES.get(s['skill_state'], '?')}")
        self.lbl_safety.setText(f"Safety\n{'CLAMPED' if s['safety_clamp'] else 'OK'}")
        tof_center = float(s['tof_grid'][3*8+3] if len(s['tof_grid']) >= 64 else 0)
        self.lbl_tof_dist.setText(f"Wrist Z\n{tof_center:.0f}mm")

def launch_dashboard(shared: SharedState):
    app = QApplication(sys.argv)
    win = VLADashboard(shared)
    win.show()
    app.exec()
```

**Expected Output of Phase 7:**
- Dashboard window opens on RPi 5 with all 5 panels visible.
- Panel 1 shows live 640×480 overhead feed with YOLO bounding boxes at ~10Hz.
- Panel 2 shows scrolling skill timeline in correct colors (blue=REACH, orange=GRASP, green=LIFT, red=PLACE).
- Panel 3 shows 8×8 ToF heatmap updating at 15Hz with blue (far) to red (near) gradient.
- Panel 4 shows IMU RMS and gripper load time series with threshold dashed line. Contact events appear as vertical lines.
- Panel 5 shows inference latency, loop Hz, current skill name, safety clamp status, and emergency stop button.
- Emergency stop button sends halt command to Teensy when clicked.

---

## Phase 8 — Evaluation and Ablation Studies

**Goal:** Quantitative success rates across 3 tasks, IMU contact detection latency measurement, skill F1-score, and 4 ablation conditions. Produce a results table suitable for the project report.

### 8.1 Evaluation Setup

```python
# evaluation/run_eval.py
import time, csv, numpy as np
from comms.teensy_serial import TeensySerial
from main import setup_pipeline   # reuse from main.py

TASKS = {
    'task1_pick_place': {
        'instruction': 'pick the {color} block and place it in the tray',
        'colors': ['red', 'blue', 'yellow'],
        'trials_per_color': 20,
        'success_criterion': 'placement_within_3cm',
    },
    'task2_stacking': {
        'instruction': 'stack the {color1} block on top of the {color2} block',
        'pairs': [('red','blue'), ('blue','yellow'), ('yellow','red')],
        'trials_per_pair': 10,
        'success_criterion': 'stable_3s',
    },
    'task3_sorting': {
        'instruction': 'pick the {color} block',
        'colors': ['red', 'blue'],
        'trials_per_color': 20,
        'success_criterion': 'correct_object_placed',
    },
}

class EvaluationRecorder:
    def __init__(self, output_path: str):
        self.path = output_path
        self.rows = []

    def record(self, task, trial, instruction, success,
               skill_f1, contact_latency_ms, inference_ms):
        self.rows.append({
            'task': task, 'trial': trial, 'instruction': instruction,
            'success': int(success), 'skill_f1': skill_f1,
            'contact_latency_ms': contact_latency_ms,
            'inference_ms': inference_ms,
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        })

    def save(self):
        with open(self.path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Results saved to {self.path}")

    def report(self):
        import pandas as pd
        df = pd.DataFrame(self.rows)
        for task in df['task'].unique():
            sub = df[df['task'] == task]
            print(f"\n{task}: success={sub['success'].mean():.1%} "
                  f"({sub['success'].sum()}/{len(sub)}) "
                  f"| mean_latency={sub['inference_ms'].mean():.1f}ms "
                  f"| mean_skill_f1={sub['skill_f1'].mean():.3f}")
```

### 8.2 Skill F1 Measurement

```python
# evaluation/skill_f1.py
from sklearn.metrics import f1_score
import numpy as np

SKILL_INT = {'REACH': 0, 'GRASP': 1, 'LIFT': 2, 'PLACE': 3}

def compute_skill_f1(predicted_labels: list, ground_truth_labels: list) -> float:
    """
    Compare auto-segmentation labels against human-annotated ground truth.
    Human annotation: label every 10th timestep in 15 held-out demos
    then interpolate. Auto labels from segment_demo().
    """
    pred = [SKILL_INT[s] if isinstance(s, str) else s for s in predicted_labels]
    gt   = [SKILL_INT[s] if isinstance(s, str) else s for s in ground_truth_labels]
    return f1_score(gt, pred, average='macro')

def annotate_demo_manually(demo_path: str) -> list:
    """
    Interactive CLI for human annotation: print timestep index and IMU/load values,
    ask human to type REACH/GRASP/LIFT/PLACE for every 10th timestep.
    """
    from dataset.hdf5_reader import load_demo, get_logical_joints
    demo = load_demo(demo_path)
    tel  = demo['telemetry']
    labels = []
    for i in range(0, len(tel), 10):
        row = tel[i]
        j   = get_logical_joints(row)
        print(f"t={i/50:.2f}s | joints={j.round(1)} | load={row['servo_load'][4]:.2f} | "
              f"contact={row['contact_flag']} | tof_center={row['tof_grid'][27]}mm")
        lbl = input("  Skill [R/G/L/P]: ").strip().upper()
        MAP = {'R': 'REACH', 'G': 'GRASP', 'L': 'LIFT', 'P': 'PLACE'}
        labels.extend([MAP.get(lbl, 'REACH')] * 10)
    return labels[:len(tel)]
```

### 8.3 IMU Contact Detection Latency Measurement

```python
# evaluation/contact_latency.py
"""
Protocol:
1. Hold arm stationary in GRASP position.
2. Manually tap the gripper with a calibration object.
3. Record the telemetry timestamp when contact_flag goes HIGH.
4. Record the actual contact timestamp from a floor pressure pad or high-speed camera.
5. Difference = contact detection latency.

Simpler alternative (software-only):
- Play back a recorded demo HDF5.
- Find the first frame where contact_flag = 1.
- Find the first frame where servo_load[4] > 0.35 (load-based confirmation).
- Difference between these two timestamps = IMU advantage over load-based detection.
"""
import numpy as np
from dataset.hdf5_reader import load_demo, TELEMETRY_DTYPE

def measure_imu_vs_load_latency(h5_path: str) -> float:
    demo = load_demo(h5_path)
    tel  = demo['telemetry']

    imu_trigger_idx  = None
    load_trigger_idx = None

    for i, row in enumerate(tel):
        if imu_trigger_idx is None and row['contact_flag']:
            imu_trigger_idx = i
        if load_trigger_idx is None and row['servo_load'][4] > 0.35:
            load_trigger_idx = i

    if imu_trigger_idx is None or load_trigger_idx is None:
        return float('nan')

    delta_samples = load_trigger_idx - imu_trigger_idx
    delta_ms = delta_samples * (1000.0 / 50.0)   # 50Hz telemetry
    return delta_ms

def batch_latency(demo_dir: str):
    from pathlib import Path
    paths = list(Path(demo_dir).glob('demo_*.h5'))
    latencies = [measure_imu_vs_load_latency(str(p)) for p in paths]
    latencies  = [l for l in latencies if not np.isnan(l)]
    print(f"IMU contact detection advantage over load-based:")
    print(f"  Mean:   {np.mean(latencies):.1f}ms earlier")
    print(f"  Median: {np.median(latencies):.1f}ms earlier")
    print(f"  Max:    {np.max(latencies):.1f}ms earlier")
```

### 8.4 Ablation Study Protocol

Run each ablation as a **separate evaluation run** of Task 1 (20 trials × 3 colors = 60 trials each). Record success rate.

| Ablation | Code Change Required | Expected Impact |
|----------|---------------------|-----------------|
| **A — No wrist ToF** | In `pose_estimation.py`, return `None` from `wrist_tof_z()` always. Fallback to Z=0.02m assumption. | Task 1 success drops 10–20% from Z error |
| **B — No IMU contact** | In `main.py`, set `contact_flag = False` always before passing to FSM. Skill transitions driven by model only. | GRASP→LIFT 100–200ms slower; more over-squeezing |
| **C — Flat VLA** | Train Octo-small without skill decomposition. Direct joint delta outputs only, no skill head. | Lower task success, no interpretable timeline |
| **D — Behind camera** | Mount camera behind/above arm base instead of overhead. Rerun 20 Task 1 trials. | Detection failures during REACH phase from arm occlusion |

```python
# evaluation/ablation.py

def ablation_A_no_tof(pose_estimator):
    """Monkey-patch wrist_tof_z to always return None."""
    pose_estimator.wrist_tof_z = lambda tof_grid, tof_valid: None
    print("Ablation A active: wrist ToF disabled, using Z=0.02m fallback")

def ablation_B_no_imu(teensy_bridge):
    """Intercept telemetry and zero out contact_flag."""
    original = teensy_bridge.latest_telemetry
    def patched():
        t = original()
        if t is not None:
            t = t.copy()
            t['contact_flag'] = 0
            t['contact_rms']  = 0.0
        return t
    teensy_bridge.latest_telemetry = patched
    print("Ablation B active: IMU contact oracle disabled")
```

### 8.5 Results Table Template

Fill this after running all evaluations:

```
| Condition                      | Task 1 (%) | Task 2 (%) | Task 3 (%) | Inf. Latency (ms) | Skill F1 |
|--------------------------------|------------|------------|------------|-------------------|----------|
| Full system                    |            |            |            |                   |          |
| Ablation A — no wrist ToF      |            |     N/A    |     N/A    |                   |   N/A    |
| Ablation B — load-only contact |            |     N/A    |     N/A    |                   |   N/A    |
| Ablation C — flat VLA          |            |            |            |                   |   N/A    |
| Ablation D — behind camera     |            |     N/A    |     N/A    |                   |   N/A    |

Targets: Task 1 ≥85%, Task 2 ≥75%, Task 3 ≥80%, Latency ≤125ms, Skill F1 ≥75%
```

**Expected Output of Phase 8:**
- `results/eval_full_system.csv` with all trial rows.
- Printed report showing success rates for all 3 tasks.
- Ablation table filled with measured numbers.
- `contact_latency_report.txt` showing mean IMU advantage over load-based detection.
- Skill F1 ≥ 0.75 on 15 manually annotated demos.

---

## Appendix A — `model_config.yaml` Template

```yaml
# config/model_config.yaml

yolo:
  model_path: "checkpoints/yolov8n_vla/weights/best.pt"
  input_size: 640
  conf_threshold: 0.5
  device: "cpu"

vla:
  model_path: "checkpoints/vla_policy_traced.pt"
  backbone: "smolvla"    # or "octo_small"
  chunk_size: 8
  inference_hz_target: 8
  device: "cpu"

language:
  mode: "native_smolvla_or_t5_fallback"
  fallback_model_name: "google/t5-small"
  embedding_dim: 512
  cache_instructions: true

serial:
  port: "/dev/ttyACM0"
  baud: 2000000
  telemetry_size_bytes: 250
  command_size_bytes: 20
  telemetry_hz: 50
  command_hz: 8
```

---

## Appendix B — `arm_config.yaml` Template (Get Actual Values from Team)

```yaml
# config/arm_config.yaml — fill actual calibrated values from Team

dh_params:
  d1_mm: 65
  a2_mm: 130
  a3_mm: 190

joint_limits_min_deg: [-150, -90, -120, 0]
joint_limits_max_deg: [ 150,  90,  120, 90]

workspace:
  x_m: [-0.38, 0.38]
  y_m: [-0.38, 0.38]
  z_m: [0.02, 0.35]

contact_oracle:
  rms_threshold_dps: 3.5
  slip_variance_threshold: 2.0

servos:
  count: 5
  ids: [1, 2, 3, 4, 5]    # Get from Team's firmware
  j0_id: 1                 # base yaw
  j1a_id: 2                # shoulder A
  j1b_id: 3                # shoulder B (coupled)
  j2_id: 4                 # elbow
  j3_id: 5                 # gripper
```

---

## Appendix C — Key Dependencies and Versions

```
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0           # YOLOv8
transformers>=4.40.0         # SmolVLA / T5
peft>=0.10.0                 # LoRA adapters
accelerate>=0.28.0
h5py>=3.10.0
numpy>=1.24.0
scipy>=1.10.0
opencv-python>=4.8.0
pyserial>=3.5
PyQt6>=6.6.0
pyqtgraph>=0.13.0
picamera2>=0.3.12
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
lerobot>=0.1.0               # SmolVLA tooling (if using SmolVLA)
```

---

## Appendix D — Verification Checklist Before Demo

- [ ] Team's Teensy transmits 250-byte packets continuously — verify with `teensy_serial.py`
- [ ] `camera_intrinsics.yaml`, `overhead_height.yaml`, `wrist_tof_offset.yaml`, `camera_to_base_transform.yaml`, and `arm_config.yaml` all present in `calibration/`
- [ ] `arm_config.yaml` has correct joint limits (get from Team, not estimates)
- [ ] YOLOv8 detects all 5 object classes at >90% confidence in current lighting
- [ ] VLA inference P95 ≤ 110ms and full pipeline P95 ≤ 125ms on RPi 5
- [ ] Skill segmentation F1 ≥ 0.75 on manually annotated validation demos
- [ ] Gripper opens during REACH, closes/holds through GRASP and LIFT, then releases only after PLACE target is reached
- [ ] Emergency stop button in dashboard sends halt command to Teensy
- [ ] Dashboard all 5 panels update at 10Hz without lag
- [ ] All evaluation CSV files saved with timestamps before demo recording
