# VLA Inference Pipeline Completion — Design Spec

**Date:** 2026-05-22
**Approach:** Approach A — Repo restructure first, then implement in four logical commits

---

## Goal

Complete the ~70% done AI/ML inference pipeline so that running the five test-command imports on the RPi5 succeeds without `ImportError` or `AttributeError`, and `main.py` runs an end-to-end 8 Hz loop (with mock VLA model) before the trained checkpoint exists.

---

## Current State

| File | State |
|------|-------|
| `vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference/vla/vla_policy.py` | 0 bytes |
| `vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference/vla/action_generator.py` | 0 bytes |
| `vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference/perception/camera_manager.py` | 0 bytes |
| `vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference/config/model_config.yaml` | 0 bytes |
| `vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference/evaluation/run_eval.py` | 0 bytes |
| `rpi5_inference/main.py` | 304 lines, working but dead inference loop (no camera, no VLA calls) |

All other modules (`yolo_detector.py`, `pose_estimation.py`, `teensy_serial.py`, `skill_predictor.py`, `ik_solver.py`, `safety_filter.py`, `language_encoder.py`, `gui.py`) are complete and must not be modified.

---

## Constraints

- Target hardware: Raspberry Pi 5, Python, CPU-only inference at 8 Hz
- Packet schema ground truth: `firmware/src/comms.h` (float32 servo fields, 250-byte telemetry, 20-byte command)
- `teensy_serial.py` is correct and must not be changed
- `CHUNK_SIZE = 8` throughout VLA modules
- No new dependencies beyond `requirements.txt`
- Commit messages must not include `Co-Authored-By:` trailers

---

## Commit Plan

| # | Commit | Files |
|---|--------|-------|
| 1 | `refactor: flatten repo structure — move AI/ML code to root level` | `git mv` only |
| 2 | `feat: add core VLA inference modules` | `vla_policy.py`, `action_generator.py`, `camera_manager.py`, `model_config.yaml` |
| 3 | `feat: add run_eval pipeline and fix main.py inference loop` | `run_eval.py`, `main.py` |
| 4 | `feat: add calibration scripts, stub YAMLs, and evaluation utilities` | `calibration/`, `evaluation/skill_f1.py`, `evaluation/contact_latency.py`, `evaluation/ablation.py` |

`python3 -m py_compile` runs on every Python file before its commit group.

---

## Section 1 — Repo Restructure

### Git commands

```bash
cd /home/m0mspagetthi/vla_rob

git mv vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference  rpi5_inference
git mv vla-robotic-arm-main/vla-robotic-arm-main/dataset          dataset
git mv vla-robotic-arm-main/vla-robotic-arm-main/checkpoints      checkpoints
git mv vla-robotic-arm-main/vla-robotic-arm-main/requirements.txt requirements.txt

mkdir -p demos && touch demos/.gitkeep

git rm -r vla-robotic-arm-main/

git add demos/
git commit -m "refactor: flatten repo structure — move AI/ML code to root level"
```

### Post-move import verification

- `rpi5_inference/main.py` — no hardcoded `vla-robotic-arm-main/` paths present; no `sys.path` manipulation
- `rpi5_inference/perception/yolo_detector.py` — `DEFAULT_CHECKPOINT = "checkpoints/yolov8n_vla/weights/best.pt"` is relative to CWD; resolves correctly when running from repo root
- All other existing modules use relative imports only — no changes needed

### Target layout after restructure

```
vla_rob/
├── firmware/                  # unchanged
├── rpi5_inference/            # moved from vla-robotic-arm-main/vla-robotic-arm-main/
│   ├── comms/
│   ├── config/
│   ├── calibration/           # new (created in Commit 4)
│   ├── dashboard/
│   ├── evaluation/
│   ├── language/
│   ├── perception/
│   ├── planning/
│   ├── vla/
│   └── main.py
├── dataset/                   # moved
├── checkpoints/               # moved (contains yolov8n_vla/weights/best.pt)
├── demos/                     # new empty dir with .gitkeep
├── docs/                      # unchanged
├── requirements.txt           # moved
├── .gitignore
├── CLAUDE.md
└── rules.md
```

---

## Section 2 — Core VLA Inference Modules

### `rpi5_inference/vla/vla_policy.py`

**Class:** `VLARuntime`
**Constant:** `CHUNK_SIZE = 8`

#### `_MockModel` (module-level private class)

An `nn.Module` used when the checkpoint file does not exist. Returns random-but-correctly-shaped outputs. Deltas are scaled by `0.01` so IK-primary steering remains stable during development:

```python
class _MockModel(torch.nn.Module):
    def forward(self, batch):
        skill_logits = torch.randn(1, 4)
        delta_joints = torch.randn(1, 8, 4) * 0.01
        return skill_logits, delta_joints
```

#### `__init__(self, model_path: str, lang_encoder)`

- If `os.path.exists(model_path)`: `self.model = torch.jit.load(model_path); self.model.eval()`
- Else: `self.model = _MockModel(); self.model.eval()` + warning log
- Stores `self.encoder = lang_encoder`
- Initializes `self._chunk_buffer = None`

#### `predict(self, rgb_frame, joint_state_4d, skill_onehot, instruction, contact_rms, tof_z_m) → tuple[int, ndarray[4], ndarray[8,4]]`

1. Resize `rgb_frame` to 224×224
2. Normalize with ImageNet stats: mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`
3. Convert to `(1, 3, 224, 224)` float32 tensor
4. Encode instruction via `self.encoder.encode(instruction)` (returns ndarray → unsqueeze to batch)
5. Build batch dict with all inputs
6. Run `torch.no_grad()` forward pass
7. Return `(skill_pred: int, delta_step0: ndarray[4], chunk_buffer: ndarray[8,4])`

---

### `rpi5_inference/vla/action_generator.py`

**Class:** `ActionGenerator`
**Constant:** `CHUNK_SIZE = 8`

| Method | Behavior |
|--------|----------|
| `__init__()` | `_chunk = None`, `_step = 0` |
| `set_chunk(chunk: ndarray[8,4])` | stores chunk, resets `_step = 0` |
| `step(joint_state_4d: ndarray[4]) → ndarray[4]` | returns `joint_state + _chunk[_step]`, increments `_step % CHUNK_SIZE`; if chunk is None returns `joint_state` unchanged |
| `reset()` | `_chunk = None`, `_step = 0` |

---

### `rpi5_inference/perception/camera_manager.py`

**Class:** `CameraManager`

#### Platform detection

```python
try:
    from picamera2 import Picamera2
    _HAVE_PICAMERA2 = True
except ImportError:
    _HAVE_PICAMERA2 = False
```

#### `__init__(self, width=640, height=480, fps=30)`

- If `_HAVE_PICAMERA2`: configure and start `Picamera2` with RGB888, `FrameRate=fps`
- Else: `self._cap = cv2.VideoCapture(0)` with width/height hints
- Start background daemon thread calling `_capture_loop()`
- Ring buffer: `_frame = None`, `_lock = threading.Lock()`

#### `_capture_loop(self)`

- picamera2 path: `cam.capture_array()` → store in `_frame` under lock
- cv2 path: `cap.read()` → bgr frame → store under lock
- Runs until `_running = False`

#### `latest_frame(self) → ndarray | None`

Thread-safe read. Returns copy of latest BGR frame, or None if no frame yet.

#### `close(self)`

Sets `_running = False`, stops camera / releases cap.

---

### `rpi5_inference/config/model_config.yaml`

```yaml
yolo:
  model_path: "checkpoints/yolov8n_vla/weights/best.pt"
  input_size: 640
  conf_threshold: 0.5
  device: "cpu"

vla:
  model_path: "checkpoints/vla_policy_traced.pt"
  backbone: "smolvla"
  chunk_size: 8
  inference_hz_target: 8
  device: "cpu"

language:
  mode: "native_smolvla_or_t5_fallback"
  fallback_model_name: "google/flan-t5-small"
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

## Section 3 — `run_eval.py` + `main.py` Fix

### `rpi5_inference/evaluation/run_eval.py`

#### `setup_pipeline() → dict`

Creates and returns all inference components. Imported by `main.py` to avoid duplication.

```python
def setup_pipeline() -> dict:
    from rpi5_inference.comms.teensy_serial        import TeensySerial
    from rpi5_inference.perception.camera_manager  import CameraManager
    from rpi5_inference.perception.yolo_detector   import YOLODetector
    from rpi5_inference.perception.pose_estimation import PoseEstimator
    from rpi5_inference.language.language_encoder  import LanguageEncoder
    from rpi5_inference.vla.vla_policy             import VLARuntime
    from rpi5_inference.vla.action_generator       import ActionGenerator
    from rpi5_inference.vla.skill_predictor        import SkillStateMachine
    from rpi5_inference.planning.safety_filter     import SafetyFilter
    ...
    return {"camera": cam, "teensy": ts, "det": det, "pose": pe,
            "lang": lang, "vla": vla, "action_gen": ag, "sm": sm, "sf": sf}
```

Note: `setup_pipeline()` does NOT open the serial port or camera in evaluation mode — callers pass port/checkpoint overrides via optional parameters.

#### `TASKS` dict

```python
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
```

Total trials: task1=60, task2=30, task3=40.

#### `EvaluationRecorder`

| Method | Signature |
|--------|-----------|
| `__init__` | `(output_path: str)` |
| `record` | `(task, trial, instruction, success, skill_f1, contact_latency_ms, inference_ms)` |
| `save` | writes CSV to `output_path`; raises `ValueError` if no rows recorded |
| `report` | prints per-task success rates, mean latency, mean skill F1 using pandas |

#### `run_eval(task_name, n_trials, pipeline)`

Main loop: iterates trials, prompts for manual success/failure input, calls `recorder.record()`.

---

### `rpi5_inference/main.py` — Targeted Fixes

The existing 304-line file is kept. The following changes are made surgically:

**1. New imports (top of file, after existing imports):**
```python
from rpi5_inference.perception.camera_manager  import CameraManager
from rpi5_inference.vla.vla_policy             import VLARuntime
from rpi5_inference.vla.action_generator       import ActionGenerator
from rpi5_inference.evaluation.run_eval        import setup_pipeline
```

**2. `run_loop()` — new component instantiation (after existing component init block):**
```python
camera     = CameraManager()
vla        = VLARuntime(DEFAULT_CHECKPOINT.replace("yolov8n_vla/weights/best.pt",
                        "vla_policy_traced.pt"), enc)
action_gen = ActionGenerator()
```

**3. Replace dead frame line** (`frame = np.zeros(...)`) **with:**
```python
frame = camera.latest_frame()
if frame is None:
    time.sleep(0.005)
    continue
```

**4. VLA inference call** (after IK block, before safety filter):
```python
skill_onehot = np.zeros(4, dtype=np.float32)
skill_onehot[int(sm.state)] = 1.0
tof_z_m = float(tof_grid.mean()) / 1000.0  # rough scalar from grid
skill_pred, delta_step0, chunk = vla.predict(
    frame, joints_4[:4].astype(np.float32), skill_onehot,
    args.instruction, float(telem["contact_rms"]) if telem is not None else 0.0,
    tof_z_m,
)
action_gen.set_chunk(chunk)
# IK-primary: VLA delta is a small additive correction
joints_4[:4] = joints_4[:4] + delta_step0 * 0.1
```

**5. FSM advance wiring** (immediately after existing `sm.notify_contact(contact)` call):
```python
sm.notify_contact(contact)
if skill_pred > int(sm.state) and not sm.done:
    sm.advance()
```

Note: `sm.done` guard prevents `advance()` being called at terminal state (PLACE), which would raise `ValueError` in the actual `SkillStateMachine` implementation.

**6. `setup_pipeline()` function** added at module level — delegates to `run_eval.setup_pipeline()`:
```python
def setup_pipeline():
    from rpi5_inference.evaluation.run_eval import setup_pipeline as _sp
    return _sp()
```

**7. `close()` on camera** in `finally` block:
```python
finally:
    ts.close()
    camera.close()
```

**What is NOT changed:** `TeensySerial` init, `SafetyFilter`, `IKSolver`, `YOLODetector`, `PoseEstimator`, `LanguageEncoder`, `_gripper_pct`, timing loop, `_dry_run`, CLI parser.

---

## Section 4 — Calibration Scripts + Stub YAMLs

### Directory: `rpi5_inference/calibration/`

#### `camera_calibrate.py`

OpenCV checkerboard calibration. Captures frames from `CameraManager`, finds corners, runs `cv2.calibrateCamera()`, writes `camera_intrinsics.yaml` with keys `K` (3×3 flattened list) and `dist` (5-element distortion list). Target: reprojection error < 0.5 px.

#### `overhead_height_calib.py`

Manual measurement helper. Prompts user to place ruler at table surface and enter the measured height in mm. Computes average over 3 measurements. Writes `overhead_height.yaml` with key `Z_table_m`.

#### `wrist_tof_calib.py`

Places arm at known distances (100mm, 200mm, 300mm from a flat surface). Reads ToF center zone average at each distance. Computes systematic offset via linear regression. Writes `wrist_tof_offset.yaml` with key `wrist_to_sensor_offset_mm`.

---

### Stub YAML files (placeholders so `PoseEstimator` loads without error)

**Note:** `PoseEstimator()` takes no constructor args and falls back to hardcoded dummy values if any YAML is missing (warns but doesn't crash). Stubs prevent the warning on clean boot.

Key names come from the actual `pose_estimation.py`, not the workplan.

#### `rpi5_inference/calibration/camera_intrinsics.yaml`
```yaml
# Placeholder — run camera_calibrate.py to get real values
# Keys must match PoseEstimator._load_intrinsics: fx, fy, cx, cy, dist_coeffs
fx: 600.0
fy: 600.0
cx: 320.0
cy: 240.0
dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

#### `rpi5_inference/calibration/camera_extrinsics.yaml`
```yaml
# Placeholder — run compute_extrinsics after camera_calibrate.py
rvec: [0.0, 0.0, 0.0]
tvec: [0.0, 0.0, 0.5]
```

#### `rpi5_inference/calibration/homography_dots.yaml`
```yaml
# Placeholder — measure actual A/B/C/D dot pixel coords on workspace mat
# Keys must match PoseEstimator._load_homography: A_px, B_px, C_px, D_px
# World positions are fixed: A=(-0.18,0.10) B=(+0.18,0.10) C=(+0.18,0.22) D=(-0.18,0.22)
A_px: [100.0, 400.0]
B_px: [540.0, 400.0]
C_px: [480.0, 220.0]
D_px: [160.0, 220.0]
```

---

## Section 5 — Evaluation Utilities

All three files are extracted verbatim from `FRIEND_AI_ML_WORKPLAN.md` Sections 8.2–8.4, with minor path corrections for the new flat layout.

### `rpi5_inference/evaluation/skill_f1.py`
- `compute_skill_f1(predicted, ground_truth) → float` — macro F1 via `sklearn.metrics.f1_score`
- `annotate_demo_manually(demo_path) → list` — interactive CLI annotation at every 10th timestep

### `rpi5_inference/evaluation/contact_latency.py`
- `measure_imu_vs_load_latency(h5_path) → float` — IMU vs load-based detection delta in ms
- `batch_latency(demo_dir)` — prints mean/median/max across all demo HDF5s

### `rpi5_inference/evaluation/ablation.py`
- `ablation_A_no_tof(pose_estimator)` — monkey-patches `wrist_tof_z` to always return None
- `ablation_B_no_imu(teensy_bridge)` — wraps `latest_telemetry` to zero out contact fields

---

## Success Criteria

After Commit 3, the following must all pass on the development machine:

```bash
cd /home/m0mspagetthi/vla_rob

python3 -c "from rpi5_inference.vla.vla_policy import VLARuntime; print('OK')"
python3 -c "from rpi5_inference.vla.action_generator import ActionGenerator; print('OK')"
python3 -c "from rpi5_inference.perception.camera_manager import CameraManager; print('OK')"
python3 -c "from rpi5_inference.evaluation.run_eval import setup_pipeline; print('OK')"
python3 -m py_compile rpi5_inference/main.py && echo "main.py syntax OK"
```

After Commit 4, additionally:
```bash
python3 -m py_compile rpi5_inference/evaluation/skill_f1.py && echo "OK"
python3 -m py_compile rpi5_inference/evaluation/contact_latency.py && echo "OK"
python3 -m py_compile rpi5_inference/evaluation/ablation.py && echo "OK"
python3 -m py_compile rpi5_inference/calibration/camera_calibrate.py && echo "OK"
```

---

## Key Decisions Recorded

| Decision | Choice | Reason |
|----------|--------|--------|
| Image resize in VLARuntime | 224×224, ImageNet norm | User spec takes precedence over workplan's 256×256 |
| `_MockModel` delta scale | ×0.01 | Keeps IK-primary loop stable; random deltas become ~0.01° corrections |
| `sm.advance()` trigger | `skill_pred > int(sm.state)` | VLA-driven progression; IMU contact still handled by `notify_contact()` |
| IK vs VLA for joint targets | IK-primary, VLA as additive correction ×0.1 | Random mock deltas won't destabilize IK; weight increases once real model loads |
| `setup_pipeline()` location | `run_eval.py` | Required by test-command: `from rpi5_inference.evaluation.run_eval import setup_pipeline` |
| `fallback_model_name` | `google/flan-t5-small` | Matches dry-run assertion: `"flan-t5" in MODEL_NAME.lower()` |
| Repo restructure timing | Commit 1, before any code | Clean git history; all subsequent work in final locations |
