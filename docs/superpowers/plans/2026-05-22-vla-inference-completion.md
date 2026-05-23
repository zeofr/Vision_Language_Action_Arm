# VLA Inference Pipeline Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the empty/stub AI/ML inference modules so that all five test-command imports pass and `main.py` runs an end-to-end 8 Hz loop with a mock VLA model before training is complete.

**Architecture:** Four independent commits: (1) `git mv` repo flatten, (2) core VLA modules (`vla_policy.py`, `action_generator.py`, `camera_manager.py`, `model_config.yaml`), (3) `run_eval.py` + surgical `main.py` fix, (4) calibration scripts/stubs and evaluation utilities. IK remains primary for joint targets; VLA delta is a small (×0.1) additive correction. `setup_pipeline()` lives in `run_eval.py` and is imported by `main.py`.

**Tech Stack:** Python 3.11, PyTorch (TorchScript), OpenCV, picamera2 (with cv2 fallback), numpy, yaml, threading

**Spec:** `docs/superpowers/specs/2026-05-22-vla-inference-completion-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Move | `vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference/` → `rpi5_inference/` | Flatten repo layout |
| Move | `vla-robotic-arm-main/vla-robotic-arm-main/dataset/` → `dataset/` | |
| Move | `vla-robotic-arm-main/vla-robotic-arm-main/checkpoints/` → `checkpoints/` | |
| Move | `vla-robotic-arm-main/vla-robotic-arm-main/requirements.txt` → `requirements.txt` | |
| Create | `rpi5_inference/vla/vla_policy.py` | `VLARuntime` + `_MockModel` |
| Create | `rpi5_inference/vla/action_generator.py` | `ActionGenerator` chunk stepper |
| Create | `rpi5_inference/perception/camera_manager.py` | `CameraManager` with picamera2/cv2 fallback |
| Create | `rpi5_inference/config/model_config.yaml` | Inference config |
| Create | `rpi5_inference/evaluation/run_eval.py` | `setup_pipeline()`, `TASKS`, `EvaluationRecorder`, `run_eval()` |
| Modify | `rpi5_inference/main.py` | Wire camera + VLA + ActionGenerator + `sm.advance()` |
| Create | `rpi5_inference/calibration/camera_intrinsics.yaml` | Placeholder stub |
| Create | `rpi5_inference/calibration/camera_extrinsics.yaml` | Placeholder stub |
| Create | `rpi5_inference/calibration/homography_dots.yaml` | Placeholder stub |
| Create | `rpi5_inference/calibration/camera_calibrate.py` | Checkerboard intrinsics calibration |
| Create | `rpi5_inference/calibration/overhead_height_calib.py` | Z_table measurement |
| Create | `rpi5_inference/calibration/wrist_tof_calib.py` | ToF offset measurement |
| Create | `rpi5_inference/evaluation/skill_f1.py` | Skill F1 measurement |
| Create | `rpi5_inference/evaluation/contact_latency.py` | IMU contact latency measurement |
| Create | `rpi5_inference/evaluation/ablation.py` | Ablation patch functions |

---

## Task 1: Repo Restructure (Commit 1)

**Files:** git mv operations only — no code written.

- [ ] **Step 1: Verify current state**

```bash
cd /home/m0mspagetthi/vla_rob
ls vla-robotic-arm-main/vla-robotic-arm-main/
```

Expected: `checkpoints/  dataset/  README.md  requirements.txt  rpi5_inference/  yolov8n.pt  ...`

- [ ] **Step 2: Move all AI/ML directories to repo root**

```bash
git mv vla-robotic-arm-main/vla-robotic-arm-main/rpi5_inference  rpi5_inference
git mv vla-robotic-arm-main/vla-robotic-arm-main/dataset          dataset
git mv vla-robotic-arm-main/vla-robotic-arm-main/checkpoints      checkpoints
git mv vla-robotic-arm-main/vla-robotic-arm-main/requirements.txt requirements.txt
```

- [ ] **Step 3: Create demos placeholder**

```bash
mkdir -p demos && touch demos/.gitkeep
```

- [ ] **Step 4: Remove the now-empty wrapper dir**

```bash
git rm -r vla-robotic-arm-main/
```

Expected: lists of deleted files ending with `vla-robotic-arm-main/vla-robotic-arm-main/yolov8n.pt`.

- [ ] **Step 5: Verify the new layout**

```bash
ls -1
```

Expected to contain: `checkpoints/  dataset/  demos/  docs/  firmware/  requirements.txt  rpi5_inference/  rules.md  ...`

```bash
ls rpi5_inference/
```

Expected: `__init__.py  comms/  config/  dashboard/  evaluation/  language/  main.py  perception/  planning/  vla/`

```bash
ls checkpoints/yolov8n_vla/weights/
```

Expected: `best.pt`

- [ ] **Step 6: Check pose_estimation.py calibration path resolves correctly**

After the move, `rpi5_inference/perception/pose_estimation.py` contains:
```python
_CALIB_DIR = Path(__file__).parents[2] / "rpi5_inference" / "calibration"
```

`parents[2]` from the new location `rpi5_inference/perception/` resolves to the repo root. Verify:

```bash
python3 -c "
from pathlib import Path
p = Path('rpi5_inference/perception/pose_estimation.py').resolve()
print(p.parents[2] / 'rpi5_inference' / 'calibration')
"
```

Expected: `...vla_rob/rpi5_inference/calibration` ✓

- [ ] **Step 7: Verify existing dry-run still passes**

```bash
cd /home/m0mspagetthi/vla_rob
python3 -m rpi5_inference.main --dry-run
```

Expected: all `[ok]` lines, final `ok`. If any `[FAIL]` appears, fix before proceeding.

- [ ] **Step 8: Commit**

```bash
git add demos/
git commit -m "refactor: flatten repo structure — move AI/ML code to root level"
```

---

## Task 2: `vla_policy.py` — VLARuntime + MockModel (Commit 2 starts)

**Files:**
- Create: `rpi5_inference/vla/vla_policy.py`

- [ ] **Step 1: Confirm the file is empty (0 bytes)**

```bash
wc -c rpi5_inference/vla/vla_policy.py
```

Expected: `0 rpi5_inference/vla/vla_policy.py`

- [ ] **Step 2: Write the file**

Create `rpi5_inference/vla/vla_policy.py` with this exact content:

```python
from __future__ import annotations

import logging
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

CHUNK_SIZE = 8

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _MockModel(nn.Module):
    """Stub used when the TorchScript checkpoint does not exist yet.

    Returns random-but-correctly-shaped outputs. Deltas are scaled ×0.01
    so the IK-primary steering loop stays stable during development.
    """

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        skill_logits = torch.randn(1, 4)
        delta_joints = torch.randn(1, CHUNK_SIZE, 4) * 0.01
        return skill_logits, delta_joints


class VLARuntime:
    """Thin wrapper around a TorchScript VLA policy checkpoint.

    Falls back to _MockModel when the checkpoint file does not exist,
    allowing end-to-end pipeline testing before training is complete.
    """

    CHUNK_SIZE = CHUNK_SIZE

    def __init__(self, model_path: str, lang_encoder) -> None:
        if os.path.exists(model_path):
            self.model: nn.Module = torch.jit.load(model_path)
            log.info("VLARuntime: loaded checkpoint %s", model_path)
        else:
            warnings.warn(
                f"VLARuntime: checkpoint '{model_path}' not found — using _MockModel.",
                UserWarning,
                stacklevel=2,
            )
            self.model = _MockModel()
        self.model.eval()
        self.encoder = lang_encoder
        self._chunk_buffer: np.ndarray | None = None

    def predict(
        self,
        rgb_frame: np.ndarray,
        joint_state_4d: np.ndarray,
        skill_onehot: np.ndarray,
        instruction: str,
        contact_rms: float,
        tof_z_m: float,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run one inference step.

        Returns:
            skill_pred  : int — argmax of skill logits (0=REACH … 3=PLACE)
            delta_step0 : ndarray[4] — first step of the action chunk
            chunk_buffer: ndarray[8, 4] — full 8-step action chunk
        """
        # 1. Preprocess image → (1, 3, 224, 224) float32, ImageNet-normalised
        img = cv2.resize(rgb_frame, (224, 224)).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        rgb_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,224,224)

        # 2. Language embedding (LanguageEncoder caches repeated instructions)
        lang_emb = self.encoder.encode(instruction)  # ndarray (512,)

        # 3. Build batch dict
        batch = {
            "rgb":          rgb_t,
            "joint_state":  torch.from_numpy(
                                np.asarray(joint_state_4d, dtype=np.float32)
                            ).unsqueeze(0),
            "skill_onehot": torch.from_numpy(
                                np.asarray(skill_onehot, dtype=np.float32)
                            ).unsqueeze(0),
            "lang_emb":     torch.from_numpy(
                                np.asarray(lang_emb, dtype=np.float32)
                            ).unsqueeze(0),
            "contact_rms":  torch.tensor([[float(contact_rms)]], dtype=torch.float32),
            "tof_scalar":   torch.tensor([[float(tof_z_m)]],     dtype=torch.float32),
        }

        # 4. Forward pass (no gradient needed)
        with torch.no_grad():
            skill_logits, delta_joints = self.model(batch)

        # 5. Decode
        skill_pred  = int(skill_logits.argmax(dim=-1).item())
        chunk_buf   = delta_joints[0].numpy()          # (8, 4)
        delta_step0 = chunk_buf[0].copy()              # (4,)

        self._chunk_buffer = chunk_buf
        return skill_pred, delta_step0, chunk_buf
```

- [ ] **Step 3: Syntax-check**

```bash
python3 -m py_compile rpi5_inference/vla/vla_policy.py && echo "OK"
```

Expected: `OK`

- [ ] **Step 4: Import test**

```bash
python3 -c "from rpi5_inference.vla.vla_policy import VLARuntime; print('OK')"
```

Expected: `OK`

---

## Task 3: `action_generator.py` — ActionGenerator

**Files:**
- Create: `rpi5_inference/vla/action_generator.py`

- [ ] **Step 1: Write the file**

Create `rpi5_inference/vla/action_generator.py`:

```python
from __future__ import annotations

import numpy as np

CHUNK_SIZE = 8


class ActionGenerator:
    """Steps through an 8-step action chunk one tick at a time.

    Decouples VLA inference cadence from command output cadence.
    Call set_chunk() when a new prediction arrives; call step() every tick.
    """

    CHUNK_SIZE = CHUNK_SIZE

    def __init__(self) -> None:
        self._chunk: np.ndarray | None = None
        self._step: int = 0

    def set_chunk(self, chunk: np.ndarray) -> None:
        """Store a new (CHUNK_SIZE, 4) delta array and reset the step counter."""
        self._chunk = np.asarray(chunk, dtype=np.float32)
        self._step = 0

    def step(self, joint_state_4d: np.ndarray) -> np.ndarray:
        """Return joint_state + chunk[i] and advance the internal counter.

        Wraps at CHUNK_SIZE. Returns joint_state unchanged if no chunk is set.
        """
        if self._chunk is None:
            return np.asarray(joint_state_4d, dtype=np.float32).copy()
        delta = self._chunk[self._step % CHUNK_SIZE]
        self._step += 1
        return np.asarray(joint_state_4d, dtype=np.float32) + delta

    def reset(self) -> None:
        """Clear the chunk and reset the counter."""
        self._chunk = None
        self._step = 0
```

- [ ] **Step 2: Syntax-check + import test**

```bash
python3 -m py_compile rpi5_inference/vla/action_generator.py && echo "syntax OK"
python3 -c "from rpi5_inference.vla.action_generator import ActionGenerator; print('import OK')"
```

Expected: both lines print OK.

- [ ] **Step 3: Smoke test (inline)**

```bash
python3 -c "
import numpy as np
from rpi5_inference.vla.action_generator import ActionGenerator

ag = ActionGenerator()

# No chunk set: step returns joint_state unchanged
js = np.array([10.0, 20.0, 30.0, 40.0])
out = ag.step(js)
assert np.allclose(out, js), f'Expected unchanged, got {out}'
print('no-chunk: OK')

# set_chunk, step through 8 steps, verify wrap
chunk = np.ones((8, 4), dtype=np.float32)
ag.set_chunk(chunk)
for i in range(9):  # 9 steps: should wrap on step 8
    out = ag.step(js)
assert ag._step == 9, f'step counter should be 9, got {ag._step}'
print('wrap: OK')

# reset clears state
ag.reset()
assert ag._chunk is None
assert ag._step == 0
print('reset: OK')

print('All smoke tests passed')
"
```

Expected: `no-chunk: OK` / `wrap: OK` / `reset: OK` / `All smoke tests passed`

---

## Task 4: `camera_manager.py` — CameraManager

**Files:**
- Create: `rpi5_inference/perception/camera_manager.py`

- [ ] **Step 1: Write the file**

Create `rpi5_inference/perception/camera_manager.py`:

```python
from __future__ import annotations

import threading
from typing import Optional

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
    _HAVE_PICAMERA2 = True
except ImportError:
    _HAVE_PICAMERA2 = False


class CameraManager:
    """Continuous background frame capture with thread-safe latest-frame access.

    Uses picamera2 on Raspberry Pi 5; falls back to cv2.VideoCapture(0) on other
    platforms so the module imports and runs in development environments.
    All frames are returned as BGR uint8 (H, W, 3).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self._frame: Optional[np.ndarray] = None
        self._lock    = threading.Lock()
        self._running = True

        if _HAVE_PICAMERA2:
            self._cam = Picamera2()
            cfg = self._cam.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                controls={"FrameRate": fps},
            )
            self._cam.configure(cfg)
            self._cam.start()
            self._mode = "picamera2"
        else:
            self._cap = cv2.VideoCapture(0)
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)
            self._mode = "cv2"

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CameraCapture"
        )
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running:
            if self._mode == "picamera2":
                frame_rgb = self._cam.capture_array()          # RGB888
                bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ok, bgr = self._cap.read()
                if not ok:
                    continue
            with self._lock:
                self._frame = bgr

    def latest_frame(self) -> Optional[np.ndarray]:
        """Return a copy of the most recent BGR frame, or None if not yet available."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def close(self) -> None:
        """Stop the capture thread and release the camera resource."""
        self._running = False
        if self._mode == "picamera2":
            self._cam.stop()
        else:
            self._cap.release()
```

- [ ] **Step 2: Syntax-check + import test**

```bash
python3 -m py_compile rpi5_inference/perception/camera_manager.py && echo "syntax OK"
python3 -c "from rpi5_inference.perception.camera_manager import CameraManager; print('import OK')"
```

Expected: both print OK. The import succeeds even without picamera2 (falls back to cv2).

---

## Task 5: `model_config.yaml` + Commit 2

**Files:**
- Create: `rpi5_inference/config/model_config.yaml`

- [ ] **Step 1: Write the config**

Create `rpi5_inference/config/model_config.yaml`:

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

- [ ] **Step 2: Verify yaml parses cleanly**

```bash
python3 -c "import yaml; d=yaml.safe_load(open('rpi5_inference/config/model_config.yaml')); print('vla.model_path =', d['vla']['model_path'])"
```

Expected: `vla.model_path = checkpoints/vla_policy_traced.pt`

- [ ] **Step 3: Run all four import tests for this commit group**

```bash
python3 -c "from rpi5_inference.vla.vla_policy import VLARuntime; print('vla_policy OK')"
python3 -c "from rpi5_inference.vla.action_generator import ActionGenerator; print('action_generator OK')"
python3 -c "from rpi5_inference.perception.camera_manager import CameraManager; print('camera_manager OK')"
```

Expected: all three print OK.

- [ ] **Step 4: Commit**

```bash
git add rpi5_inference/vla/vla_policy.py \
        rpi5_inference/vla/action_generator.py \
        rpi5_inference/perception/camera_manager.py \
        rpi5_inference/config/model_config.yaml
git commit -m "feat: add core VLA inference modules (VLARuntime, ActionGenerator, CameraManager)"
```

---

## Task 6: `run_eval.py` — Pipeline Factory + Evaluation Recorder

**Files:**
- Create: `rpi5_inference/evaluation/run_eval.py`

- [ ] **Step 1: Confirm the file is empty**

```bash
wc -c rpi5_inference/evaluation/run_eval.py
```

Expected: `0 rpi5_inference/evaluation/run_eval.py`

- [ ] **Step 2: Write the file**

Create `rpi5_inference/evaluation/run_eval.py`:

```python
"""
Evaluation harness for the VLA robotic arm pipeline.

Public API
----------
setup_pipeline()          -- create all inference components, return dict
EvaluationRecorder        -- record trial outcomes, save CSV, print report
TASKS                     -- task configs (name → trial params)
run_eval(name, n, pipe)   -- interactive trial loop
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


# ── task configs ──────────────────────────────────────────────────────────────

TASKS: dict[str, dict[str, Any]] = {
    "task1_pick_place": {
        "instruction": "pick the {color} block and place it in the tray",
        "colors": ["red", "blue", "yellow"],
        "trials_per_color": 20,
        "total_trials": 60,
        "success_criterion": "placement_within_3cm",
    },
    "task2_stacking": {
        "instruction": "stack the {color1} block on top of the {color2} block",
        "pairs": [("red", "blue"), ("blue", "yellow"), ("yellow", "red")],
        "trials_per_pair": 10,
        "total_trials": 30,
        "success_criterion": "stable_3s",
    },
    "task3_sorting": {
        "instruction": "pick the {color} block",
        "colors": ["red", "blue"],
        "trials_per_color": 20,
        "total_trials": 40,
        "success_criterion": "correct_object_placed",
    },
}


# ── pipeline factory ──────────────────────────────────────────────────────────

def setup_pipeline(
    port: str = "/dev/ttyACM0",
    yolo_checkpoint: str = "checkpoints/yolov8n_vla/weights/best.pt",
    vla_checkpoint:  str = "checkpoints/vla_policy_traced.pt",
) -> dict:
    """Instantiate all inference components and return them as a dict.

    All imports are lazy so that importing this module does not trigger
    hardware initialisation (required for the dry-run import test).

    Keys in the returned dict:
        camera, teensy, det, pose, lang, vla, action_gen, sm, sf
    """
    from rpi5_inference.comms.teensy_serial        import TeensySerial
    from rpi5_inference.perception.camera_manager  import CameraManager
    from rpi5_inference.perception.yolo_detector   import YOLODetector
    from rpi5_inference.perception.pose_estimation import PoseEstimator
    from rpi5_inference.language.language_encoder  import LanguageEncoder
    from rpi5_inference.vla.vla_policy             import VLARuntime
    from rpi5_inference.vla.action_generator       import ActionGenerator
    from rpi5_inference.vla.skill_predictor        import SkillStateMachine
    from rpi5_inference.planning.safety_filter     import SafetyFilter

    lang       = LanguageEncoder()
    camera     = CameraManager()
    det        = YOLODetector(yolo_checkpoint)
    pe         = PoseEstimator()
    vla        = VLARuntime(vla_checkpoint, lang)
    action_gen = ActionGenerator()
    sm         = SkillStateMachine()
    sf         = SafetyFilter()
    ts         = TeensySerial(port)

    return {
        "camera":     camera,
        "teensy":     ts,
        "det":        det,
        "pose":       pe,
        "lang":       lang,
        "vla":        vla,
        "action_gen": action_gen,
        "sm":         sm,
        "sf":         sf,
    }


# ── evaluation recorder ───────────────────────────────────────────────────────

class EvaluationRecorder:
    """Accumulate trial results in memory, save to CSV, print summary report."""

    _FIELDS = (
        "task", "trial", "instruction", "success",
        "skill_f1", "contact_latency_ms", "inference_ms", "timestamp",
    )

    def __init__(self, output_path: str) -> None:
        self.path = output_path
        self.rows: list[dict] = []

    def record(
        self,
        task: str,
        trial: int,
        instruction: str,
        success: bool,
        skill_f1: float,
        contact_latency_ms: float,
        inference_ms: float,
    ) -> None:
        self.rows.append({
            "task":               task,
            "trial":              trial,
            "instruction":        instruction,
            "success":            int(success),
            "skill_f1":           skill_f1,
            "contact_latency_ms": contact_latency_ms,
            "inference_ms":       inference_ms,
            "timestamp":          time.strftime("%Y%m%d_%H%M%S"),
        })

    def save(self) -> None:
        if not self.rows:
            raise ValueError("No rows to save — record at least one trial first.")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Results saved to {self.path}")

    def report(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available — raw row dump:")
            for row in self.rows:
                print(row)
            return

        df = pd.DataFrame(self.rows)
        for task in df["task"].unique():
            sub = df[df["task"] == task]
            print(
                f"\n{task}: success={sub['success'].mean():.1%} "
                f"({int(sub['success'].sum())}/{len(sub)}) "
                f"| mean_latency={sub['inference_ms'].mean():.1f}ms "
                f"| mean_skill_f1={sub['skill_f1'].mean():.3f}"
            )


# ── trial loop ────────────────────────────────────────────────────────────────

def _build_instruction(task_cfg: dict, trial: int) -> str:
    if "pairs" in task_cfg:
        pair = task_cfg["pairs"][trial % len(task_cfg["pairs"])]
        return task_cfg["instruction"].format(color1=pair[0], color2=pair[1])
    if "colors" in task_cfg:
        color = task_cfg["colors"][trial % len(task_cfg["colors"])]
        return task_cfg["instruction"].format(color=color)
    return task_cfg["instruction"]


def run_eval(
    task_name: str,
    n_trials: int,
    pipeline: dict,
    output_path: str = "results/eval_results.csv",
) -> EvaluationRecorder:
    """Interactive evaluation loop.

    The inference pipeline (pipeline dict from setup_pipeline()) should already
    be running in a background thread or called externally. This function
    collects trial outcomes via manual user input.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Valid: {list(TASKS)}")

    task_cfg = TASKS[task_name]
    recorder = EvaluationRecorder(output_path)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}  ({n_trials} trials)")
    print(f"Criterion: {task_cfg['success_criterion']}")
    print(f"{'='*60}")

    for trial in range(n_trials):
        instruction = _build_instruction(task_cfg, trial)
        print(f"\nTrial {trial + 1}/{n_trials}: '{instruction}'")
        input("  Set up the scene, then press Enter to start…")

        t0 = time.monotonic()
        input("  Press Enter when the trial is complete…")
        inference_ms = (time.monotonic() - t0) * 1000.0

        s = input("  Success? [y/N]: ").strip().lower()
        success = s == "y"

        recorder.record(
            task=task_name,
            trial=trial,
            instruction=instruction,
            success=success,
            skill_f1=0.0,           # filled post-hoc via skill_f1.py
            contact_latency_ms=0.0,  # filled post-hoc via contact_latency.py
            inference_ms=inference_ms,
        )
        print(f"  Recorded: {'✓ success' if success else '✗ failure'}")

    recorder.save()
    recorder.report()
    return recorder
```

- [ ] **Step 3: Syntax-check + import test**

```bash
python3 -m py_compile rpi5_inference/evaluation/run_eval.py && echo "syntax OK"
python3 -c "from rpi5_inference.evaluation.run_eval import setup_pipeline; print('import OK')"
```

Expected: both print OK. `setup_pipeline` is importable without opening any serial port.

---

## Task 7: Fix `main.py` + Commit 3

**Files:**
- Modify: `rpi5_inference/main.py`

The existing 304-line file is surgically modified. Eight targeted edits — no sections are rewritten wholesale.

- [ ] **Step 1: Add `DEFAULT_VLA_CHECKPOINT` constant**

Find the line:
```python
DEFAULT_CHECKPOINT  = "checkpoints/yolov8n_vla/weights/best.pt"
```

Add one line immediately after it:
```python
DEFAULT_VLA_CHECKPOINT = "checkpoints/vla_policy_traced.pt"
```

- [ ] **Step 2: Add `setup_pipeline` wrapper at module level**

Find the `_gripper_pct` function (around line 183). Add this function immediately after it, before `run_loop`:

```python
def setup_pipeline():
    """Delegate to run_eval.setup_pipeline(). Imported by evaluation scripts."""
    from rpi5_inference.evaluation.run_eval import setup_pipeline as _sp
    return _sp()
```

- [ ] **Step 3: Add new component imports inside `run_loop`**

Find this line inside `run_loop`:
```python
    from rpi5_inference.comms.teensy_serial       import TeensySerial
```

Add three lines immediately after it:
```python
    from rpi5_inference.perception.camera_manager import CameraManager
    from rpi5_inference.vla.vla_policy            import VLARuntime
    from rpi5_inference.vla.action_generator      import ActionGenerator
```

- [ ] **Step 4: Instantiate new components**

Find this line inside `run_loop`:
```python
    ts  = TeensySerial(args.port)
```

Add three lines immediately after it:
```python
    camera     = CameraManager()
    vla        = VLARuntime(DEFAULT_VLA_CHECKPOINT, enc)
    action_gen = ActionGenerator()
```

- [ ] **Step 5: Fix tof_grid shape + replace dead frame line**

The existing code at line 232 produces a `(64,)` flat array from telemetry, but `PoseEstimator.wrist_tof_z` requires a 2-D `(N, N)` array. This bug is hidden in the current dead loop (YOLO never detects anything on `np.zeros` frames) but will crash once a real camera is connected. Fix it in the same edit.

Find this block:
```python
            # ── 2. Camera → detection ─────────────────────────────────
            # TODO: replace with camera_manager.get_frame() once integrated
            frame      = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = det.detect(frame)
```

Also find (two lines earlier, in the telemetry section):
```python
                tof_grid = telem["tof_grid"][0].astype(np.float64)
```

Replace the tof_grid line with:
```python
                tof_grid = telem["tof_grid"][0].astype(np.float64).reshape(8, 8)
```

Replace the dead frame block with:
```python
            # ── 2. Camera → detection ─────────────────────────────────
            frame = camera.latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            detections = det.detect(frame)
```

- [ ] **Step 6: Add VLA predict block + FSM advance**

Find this line:
```python
            # ── 3. Pose → IK → safety ────────────────────────────────
```

Insert the following block immediately before it (before the `# ── 3.` comment):

```python
            # ── 3. VLA predict ────────────────────────────────────────
            skill_onehot = np.zeros(4, dtype=np.float32)
            skill_onehot[int(sm.state)] = 1.0

            if telem is not None:
                tof_raw = np.asarray(telem["tof_grid"]).flatten()
                center_vals = [float(tof_raw[3*8+3]), float(tof_raw[3*8+4]),
                               float(tof_raw[4*8+3]), float(tof_raw[4*8+4])]
                valid_vals  = [z for z in center_vals if 20 < z < 600]
                tof_z_m     = float(np.mean(valid_vals)) / 1000.0 if valid_vals else 0.3
                contact_rms_v = float(np.asarray(telem["contact_rms"]).flat[0])
                j_raw = np.asarray(telem["servo_pos"]).flatten()
                joint_state_vla = np.array([
                    j_raw[0],
                    (j_raw[1] + j_raw[2]) / 2.0,
                    j_raw[3],
                    j_raw[4],
                ], dtype=np.float32)
            else:
                tof_z_m         = 0.3
                contact_rms_v   = 0.0
                joint_state_vla = _HOLD_JOINTS[:4].astype(np.float32)

            skill_pred, delta_step0, chunk = vla.predict(
                frame, joint_state_vla, skill_onehot,
                args.instruction, contact_rms_v, tof_z_m,
            )
            action_gen.set_chunk(chunk)

            if skill_pred > int(sm.state) and not sm.done:
                sm.advance()

```

- [ ] **Step 7: Apply VLA delta correction before safety filter**

Find this block:
```python
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                safe_joints = sf.filter(joints_4)
```

Add the VLA delta line immediately before it:
```python
            # IK-primary: VLA delta is a small additive correction (×0.1 while mock)
            joints_4[:4] = joints_4[:4] + delta_step0 * 0.1

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                safe_joints = sf.filter(joints_4)
```

- [ ] **Step 8: Add `camera.close()` to the finally block**

Find:
```python
    finally:
        ts.close()
```

Replace with:
```python
    finally:
        ts.close()
        camera.close()
```

- [ ] **Step 9: Verify the dry-run still passes (all smoke tests must stay green)**

```bash
python3 -m rpi5_inference.main --dry-run
```

Expected: all `[ok]` lines, final `ok`. If any `[FAIL]` appears, the edit introduced a regression — revert the problematic step and fix it.

- [ ] **Step 10: Run all five test-command imports**

```bash
python3 -c "from rpi5_inference.vla.vla_policy import VLARuntime; print('OK')"
python3 -c "from rpi5_inference.vla.action_generator import ActionGenerator; print('OK')"
python3 -c "from rpi5_inference.perception.camera_manager import CameraManager; print('OK')"
python3 -c "from rpi5_inference.evaluation.run_eval import setup_pipeline; print('OK')"
python3 -m py_compile rpi5_inference/main.py && echo "main.py syntax OK"
```

Expected: all five lines print OK.

- [ ] **Step 11: Commit**

```bash
git add rpi5_inference/evaluation/run_eval.py rpi5_inference/main.py
git commit -m "feat: add run_eval pipeline factory and wire VLA+camera into main loop"
```

---

## Task 8: Calibration Stub YAMLs (Commit 4 starts)

**Files:**
- Create: `rpi5_inference/calibration/camera_intrinsics.yaml`
- Create: `rpi5_inference/calibration/camera_extrinsics.yaml`
- Create: `rpi5_inference/calibration/homography_dots.yaml`

These stubs allow `PoseEstimator` to load without printing dummy-value warnings. Key names match what `pose_estimation.py` actually reads. If the files are absent the module still works (falls back to hardcoded dummies), but real measurements should replace these before demo.

- [ ] **Step 1: Create the calibration directory**

```bash
mkdir -p rpi5_inference/calibration
```

- [ ] **Step 2: Write `camera_intrinsics.yaml`**

```yaml
# Placeholder — run camera_calibrate.py to get real values.
# Keys match PoseEstimator._load_intrinsics (pose_estimation.py line ~116).
fx: 600.0
fy: 600.0
cx: 320.0
cy: 240.0
dist_coeffs: [0.0, 0.0, 0.0, 0.0, 0.0]
```

Save as `rpi5_inference/calibration/camera_intrinsics.yaml`.

- [ ] **Step 3: Write `camera_extrinsics.yaml`**

```yaml
# Placeholder — run compute_extrinsics after camera_calibrate.py.
# Keys match PoseEstimator._load_extrinsics.
rvec: [0.0, 0.0, 0.0]
tvec: [0.0, 0.0, 0.5]
```

Save as `rpi5_inference/calibration/camera_extrinsics.yaml`.

- [ ] **Step 4: Write `homography_dots.yaml`**

```yaml
# Placeholder — measure actual pixel coordinates of the 4 workspace dots.
# Keys must be A_px, B_px, C_px, D_px — as read by PoseEstimator._load_homography.
# World positions are fixed: A=(-0.180,0.100) B=(+0.180,0.100) C=(+0.180,0.220) D=(-0.180,0.220).
A_px: [100.0, 400.0]
B_px: [540.0, 400.0]
C_px: [480.0, 220.0]
D_px: [160.0, 220.0]
```

Save as `rpi5_inference/calibration/homography_dots.yaml`.

- [ ] **Step 5: Verify PoseEstimator loads without warnings**

```bash
python3 -c "
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    from rpi5_inference.perception.pose_estimation import PoseEstimator
    pe = PoseEstimator()
print(f'Warnings emitted: {len(w)}')
for warning in w:
    print(' -', warning.message)
"
```

Expected: `Warnings emitted: 0` (the stub files are found, no fallback to dummies).
If warnings appear, check that the YAML key names match exactly (`fx`, `fy`, `cx`, `cy`, `dist_coeffs`, `rvec`, `tvec`, `A_px`, `B_px`, `C_px`, `D_px`).

---

## Task 9: Calibration Scripts

**Files:**
- Create: `rpi5_inference/calibration/camera_calibrate.py`
- Create: `rpi5_inference/calibration/overhead_height_calib.py`
- Create: `rpi5_inference/calibration/wrist_tof_calib.py`

- [ ] **Step 1: Write `camera_calibrate.py`**

Create `rpi5_inference/calibration/camera_calibrate.py`:

```python
#!/usr/bin/env python3
"""
Camera intrinsics calibration using an OpenCV checkerboard target.

Usage
-----
  python3 camera_calibrate.py

Procedure
---------
  1. Print a 9×6 inner-corner checkerboard with known square size (default 25 mm).
  2. Hold it flat at various angles/distances covering the full frame.
  3. Press SPACE to capture a frame when the board is detected (green corners drawn).
  4. Capture at least 10 frames for good coverage.
  5. Press Q to compute calibration and save results.

Output
------
  rpi5_inference/calibration/camera_intrinsics.yaml
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

BOARD_W: int   = 9       # number of inner corners, width
BOARD_H: int   = 6       # number of inner corners, height
SQUARE_MM: float = 25.0  # measured square size in mm — adjust to match your board
MIN_CAPTURES: int = 10

OUTPUT_PATH = Path(__file__).parent / "camera_intrinsics.yaml"
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _open_camera():
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        cam.configure(cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"}
        ))
        cam.start()
        return cam, "picamera2"
    except ImportError:
        cap = cv2.VideoCapture(0)
        return cap, "cv2"


def _read_frame(cam, mode: str):
    if mode == "picamera2":
        rgb = cam.capture_array()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, bgr = cam.read()
    return bgr if ok else None


def _release(cam, mode: str) -> None:
    if mode == "picamera2":
        cam.stop()
    else:
        cam.release()


def main() -> None:
    objp = np.zeros((BOARD_H * BOARD_W, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_W, 0:BOARD_H].T.reshape(-1, 2) * SQUARE_MM

    objpoints: list = []
    imgpoints: list = []
    gray_shape = None

    cam, mode = _open_camera()
    print(f"Camera: {mode}")
    print(f"Board: {BOARD_W}×{BOARD_H} inner corners, {SQUARE_MM} mm squares")
    print("SPACE = capture frame when board is detected  |  Q = compute & save")

    while True:
        frame = _read_frame(cam, mode)
        if frame is None:
            continue

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (BOARD_W, BOARD_H), None)

        display = frame.copy()
        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CRITERIA)
            cv2.drawChessboardCorners(display, (BOARD_W, BOARD_H), corners2, found)

        cv2.putText(
            display,
            f"Captures: {len(objpoints)}/{MIN_CAPTURES}  {'[FOUND]' if found else ''}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2,
        )
        cv2.imshow("Calibration — SPACE=capture  Q=done", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            objpoints.append(objp)
            imgpoints.append(corners2)
            gray_shape = gray.shape
            print(f"  Captured {len(objpoints)} frames")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    _release(cam, mode)

    if len(objpoints) < MIN_CAPTURES:
        print(f"Not enough captures ({len(objpoints)} < {MIN_CAPTURES}). Aborted.")
        sys.exit(1)

    h, w = gray_shape
    ret, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )
    print(f"\nCalibration RMS reprojection error: {ret:.4f} px")
    if ret > 0.5:
        print("WARNING: RMS > 0.5 px — consider recapturing with more diverse angles.")

    data = {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "dist_coeffs": dist.flatten().tolist(),
        "rms_reprojection_error_px": float(ret),
        "image_size_wh": [w, h],
    }
    OUTPUT_PATH.write_text(yaml.dump(data, default_flow_style=False))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `overhead_height_calib.py`**

Create `rpi5_inference/calibration/overhead_height_calib.py`:

```python
#!/usr/bin/env python3
"""
Overhead camera height calibration.

Measures the perpendicular distance from the camera lens to the workspace surface.
Takes three manual tape-measure readings and saves the average.

Output
------
  rpi5_inference/calibration/overhead_height.yaml  (key: Z_table_m)
"""

import yaml
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "overhead_height.yaml"


def main() -> None:
    print("Overhead camera height calibration")
    print("Measure from the camera lens down to the workspace mat surface.")
    print("Take 3 tape-measure readings for averaging.\n")

    readings: list[float] = []
    for i in range(3):
        while True:
            try:
                val = float(input(f"  Measurement {i + 1}/3 (mm): ").strip())
                if val <= 0:
                    print("  Value must be positive.")
                    continue
                readings.append(val)
                break
            except ValueError:
                print("  Enter a numeric value.")

    avg_mm = sum(readings) / len(readings)
    avg_m  = avg_mm / 1000.0
    std_mm = (sum((x - avg_mm) ** 2 for x in readings) / len(readings)) ** 0.5

    print(f"\n  Mean: {avg_mm:.1f} mm = {avg_m:.4f} m")
    print(f"  Std:  {std_mm:.2f} mm")

    confirm = input("\nSave this value? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted — no file written.")
        return

    data = {
        "Z_table_m":              avg_m,
        "Z_table_mm":             avg_mm,
        "n_measurements":         len(readings),
        "raw_measurements_mm":    readings,
        "std_mm":                 std_mm,
    }
    OUTPUT_PATH.write_text(yaml.dump(data, default_flow_style=False))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `wrist_tof_calib.py`**

Create `rpi5_inference/calibration/wrist_tof_calib.py`:

```python
#!/usr/bin/env python3
"""
Wrist VL53L5CX ToF sensor offset calibration.

Holds the arm wrist at known distances from a flat surface, reads live ToF
centre-zone averages via Teensy telemetry, and computes the systematic offset.

Output
------
  rpi5_inference/calibration/wrist_tof_offset.yaml  (key: wrist_to_sensor_offset_mm)

Requirements
------------
  Teensy must be connected and transmitting 50 Hz telemetry.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

OUTPUT_PATH = Path(__file__).parent / "wrist_tof_offset.yaml"
KNOWN_DISTANCES_MM = [100, 200, 300]   # hold wrist at these heights above a flat surface
N_SAMPLES = 20                          # samples to average at each distance


def _read_tof_centre(ts) -> float | None:
    telem = ts.latest_telemetry
    if telem is None:
        return None
    tof_raw = np.asarray(telem["tof_grid"]).flatten()
    centre  = [
        float(tof_raw[3 * 8 + 3]), float(tof_raw[3 * 8 + 4]),
        float(tof_raw[4 * 8 + 3]), float(tof_raw[4 * 8 + 4]),
    ]
    valid = [z for z in centre if 20 < z < 1000]
    return float(np.mean(valid)) if valid else None


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    port = input("Teensy serial port [/dev/ttyACM0]: ").strip() or "/dev/ttyACM0"

    from rpi5_inference.comms.teensy_serial import TeensySerial

    print(f"Connecting to {port}…")
    ts = TeensySerial(port)
    time.sleep(2.0)   # allow rx thread to buffer at least one packet

    offsets: list[float] = []

    for known_mm in KNOWN_DISTANCES_MM:
        input(f"\nHold wrist {known_mm} mm above a flat surface. Press Enter when stable…")

        readings: list[float] = []
        for _ in range(N_SAMPLES):
            r = _read_tof_centre(ts)
            if r is not None:
                readings.append(r)
            time.sleep(0.02)

        if not readings:
            print(f"  No valid ToF readings at {known_mm} mm — skipping.")
            continue

        mean_mm = float(np.mean(readings))
        offset  = mean_mm - known_mm
        offsets.append(offset)
        print(f"  ToF reads {mean_mm:.1f} mm at {known_mm} mm true → offset = {offset:+.1f} mm")

    ts.close()

    if not offsets:
        print("No usable measurements collected. Aborted.")
        sys.exit(1)

    mean_offset = float(np.mean(offsets))
    print(f"\nMean offset: {mean_offset:+.2f} mm")

    confirm = input("Save? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    data = {
        "wrist_to_sensor_offset_mm": mean_offset,
        "n_measurements":            len(offsets),
        "raw_offsets_mm":            offsets,
        "known_distances_mm":        KNOWN_DISTANCES_MM,
    }
    OUTPUT_PATH.write_text(yaml.dump(data, default_flow_style=False))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Syntax-check all three scripts**

```bash
python3 -m py_compile rpi5_inference/calibration/camera_calibrate.py    && echo "camera_calibrate OK"
python3 -m py_compile rpi5_inference/calibration/overhead_height_calib.py && echo "overhead_height OK"
python3 -m py_compile rpi5_inference/calibration/wrist_tof_calib.py      && echo "wrist_tof OK"
```

Expected: all three print OK.

---

## Task 10: Evaluation Utilities + Commit 4

**Files:**
- Create: `rpi5_inference/evaluation/skill_f1.py`
- Create: `rpi5_inference/evaluation/contact_latency.py`
- Create: `rpi5_inference/evaluation/ablation.py`

- [ ] **Step 1: Write `skill_f1.py`**

Create `rpi5_inference/evaluation/skill_f1.py`:

```python
"""
Skill F1 measurement utilities.

compute_skill_f1  -- macro F1 between auto-segmented and human-annotated labels
annotate_demo_manually -- interactive CLI to label every 10th timestep of a demo
"""

from __future__ import annotations

SKILL_INT = {"REACH": 0, "GRASP": 1, "LIFT": 2, "PLACE": 3}
SKILL_STR = {v: k for k, v in SKILL_INT.items()}


def compute_skill_f1(
    predicted_labels: list,
    ground_truth_labels: list,
) -> float:
    """Return macro-averaged F1 between predicted and ground-truth skill labels.

    Both lists may contain int (0–3) or str ('REACH'/'GRASP'/'LIFT'/'PLACE').
    """
    from sklearn.metrics import f1_score

    def to_int(labels: list) -> list[int]:
        return [SKILL_INT[s] if isinstance(s, str) else int(s) for s in labels]

    pred = to_int(predicted_labels)
    gt   = to_int(ground_truth_labels)
    return float(f1_score(gt, pred, average="macro", zero_division=0))


def annotate_demo_manually(demo_path: str) -> list[str]:
    """Interactive CLI for human annotation of a demo HDF5 file.

    Prints sensor summary for every 10th timestep and prompts the user to
    label it REACH / GRASP / LIFT / PLACE. Returns a full-length list
    (labels for every timestep, populated by forward-filling from annotated
    keyframes).

    Usage::

        labels = annotate_demo_manually("demos/demo_001_pick_red_block.h5")
        # labels is a list of str of length == len(telemetry)
    """
    import sys
    import numpy as np

    sys.path.insert(0, __file__.split("rpi5_inference")[0])
    from dataset.hdf5_reader import load_demo

    _MAP = {"R": "REACH", "G": "GRASP", "L": "LIFT", "P": "PLACE"}

    demo = load_demo(demo_path)
    tel  = demo["telemetry"]
    n    = len(tel)

    keyframe_labels: dict[int, str] = {}
    print(f"\nAnnotating {demo_path} ({n} timesteps, labelling every 10th)")
    print("Input: R=REACH  G=GRASP  L=LIFT  P=PLACE\n")

    for i in range(0, n, 10):
        row       = tel[i]
        j_pos     = [float(row["servo_pos"][k]) for k in range(5)]
        load_grip = float(row["servo_load"][4])
        contact   = int(row["contact_flag"])
        tof_c     = int(row["tof_grid"][3 * 8 + 3])

        print(
            f"t={i / 50:.2f}s | pos={[round(v,1) for v in j_pos]} "
            f"| grip_load={load_grip:.2f} | contact={contact} | tof_c={tof_c}mm"
        )
        while True:
            raw = input("  [R/G/L/P]: ").strip().upper()
            if raw in _MAP:
                keyframe_labels[i] = _MAP[raw]
                break
            print("  Enter R, G, L, or P.")

    # Forward-fill from keyframes to produce a label per timestep
    labels: list[str] = ["REACH"] * n
    last = "REACH"
    for i in range(n):
        if i in keyframe_labels:
            last = keyframe_labels[i]
        labels[i] = last

    return labels
```

- [ ] **Step 2: Write `contact_latency.py`**

Create `rpi5_inference/evaluation/contact_latency.py`:

```python
"""
IMU contact detection latency measurement.

Quantifies how many milliseconds earlier the IMU-based contact oracle
detects a grasp event compared to a load-threshold approach on the same demo.

measure_imu_vs_load_latency  -- single demo, returns ms advantage (positive = IMU faster)
batch_latency                -- all demos in a directory, prints summary stats
"""

from __future__ import annotations

import numpy as np

LOAD_THRESHOLD = 0.35   # normalised gripper load that indicates contact
TELEMETRY_HZ   = 50     # samples per second


def measure_imu_vs_load_latency(h5_path: str) -> float:
    """Return how many ms earlier the IMU contact flag fires vs. load threshold.

    Positive value = IMU fires first (expected).
    NaN = one or both triggers never fired in this demo.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(h5_path).parents[1]))
    from dataset.hdf5_reader import load_demo

    demo = load_demo(h5_path)
    tel  = demo["telemetry"]

    imu_trigger_idx:  int | None = None
    load_trigger_idx: int | None = None

    for i, row in enumerate(tel):
        if imu_trigger_idx is None and int(row["contact_flag"]):
            imu_trigger_idx = i
        if load_trigger_idx is None and float(row["servo_load"][4]) > LOAD_THRESHOLD:
            load_trigger_idx = i

    if imu_trigger_idx is None or load_trigger_idx is None:
        return float("nan")

    delta_samples = load_trigger_idx - imu_trigger_idx
    return delta_samples * (1000.0 / TELEMETRY_HZ)    # convert samples → ms


def batch_latency(demo_dir: str) -> None:
    """Compute and print IMU advantage statistics across all demo HDF5 files."""
    from pathlib import Path

    paths     = sorted(Path(demo_dir).glob("demo_*.h5"))
    latencies = [measure_imu_vs_load_latency(str(p)) for p in paths]
    valid     = [v for v in latencies if not np.isnan(v)]

    print(f"IMU contact detection advantage over load-based ({len(valid)}/{len(paths)} demos):")
    if not valid:
        print("  No demos had both triggers fire — cannot compute latency.")
        return
    print(f"  Mean:   {np.mean(valid):.1f} ms earlier")
    print(f"  Median: {np.median(valid):.1f} ms earlier")
    print(f"  Std:    {np.std(valid):.1f} ms")
    print(f"  Max:    {np.max(valid):.1f} ms earlier")
```

- [ ] **Step 3: Write `ablation.py`**

Create `rpi5_inference/evaluation/ablation.py`:

```python
"""
Ablation study patch functions.

Each function monkey-patches a live component to disable one sensing modality
so that evaluation trials can measure the contribution of that modality.

Apply before calling run_eval(); undo by restarting the pipeline.

Ablation A — no wrist ToF:     ablation_A_no_tof(pose_estimator)
Ablation B — no IMU contact:   ablation_B_no_imu(teensy_bridge)
"""

from __future__ import annotations


def ablation_A_no_tof(pose_estimator) -> None:
    """Disable the wrist ToF Z estimate in PoseEstimator.

    wrist_tof_z() is patched to always raise ValueError (the existing
    compute_pick_pose call-site catches this and falls back to Z_TABLE).
    Measures task-1 success degradation from Z estimation errors.
    """

    def _always_invalid(tof_grid):
        raise ValueError("Ablation A: wrist ToF disabled")

    pose_estimator.wrist_tof_z = _always_invalid
    print("Ablation A active: wrist ToF disabled (Z fallback = Z_TABLE constant).")


def ablation_B_no_imu(teensy_bridge) -> None:
    """Zero out the contact_flag and contact_rms fields in every telemetry packet.

    Skill transitions from GRASP→LIFT must now come from the VLA model alone;
    the IMU contact oracle is silenced.  Measures added latency and grip damage.

    Handles both @property and regular-method implementations of latest_telemetry.
    """
    import numpy as np

    cls  = type(teensy_bridge)
    attr = cls.__dict__.get("latest_telemetry")

    def _zero_contact(t):
        if t is None:
            return None
        t = t.copy()
        t["contact_flag"] = 0
        t["contact_rms"]  = np.float32(0.0)
        return t

    if isinstance(attr, property):
        original_getter = attr.fget

        def _patched_getter(self):
            return _zero_contact(original_getter(self))

        setattr(cls, "latest_telemetry", property(_patched_getter))
    else:
        # Regular method: patch on the instance to avoid affecting other instances
        original_method = teensy_bridge.latest_telemetry

        def _patched_method():
            return _zero_contact(original_method())

        teensy_bridge.latest_telemetry = _patched_method

    print("Ablation B active: IMU contact oracle disabled (contact_flag zeroed).")
```

- [ ] **Step 4: Syntax-check all evaluation utilities**

```bash
python3 -m py_compile rpi5_inference/evaluation/skill_f1.py       && echo "skill_f1 OK"
python3 -m py_compile rpi5_inference/evaluation/contact_latency.py && echo "contact_latency OK"
python3 -m py_compile rpi5_inference/evaluation/ablation.py        && echo "ablation OK"
```

Expected: all three print OK.

- [ ] **Step 5: Full final verification — run all five test commands**

```bash
cd /home/m0mspagetthi/vla_rob
python3 -c "from rpi5_inference.vla.vla_policy import VLARuntime; print('vla_policy OK')"
python3 -c "from rpi5_inference.vla.action_generator import ActionGenerator; print('action_generator OK')"
python3 -c "from rpi5_inference.perception.camera_manager import CameraManager; print('camera_manager OK')"
python3 -c "from rpi5_inference.evaluation.run_eval import setup_pipeline; print('run_eval OK')"
python3 -m py_compile rpi5_inference/main.py && echo "main.py syntax OK"
```

Expected: all five lines print OK.

- [ ] **Step 6: Commit**

```bash
git add rpi5_inference/calibration/ \
        rpi5_inference/evaluation/skill_f1.py \
        rpi5_inference/evaluation/contact_latency.py \
        rpi5_inference/evaluation/ablation.py
git commit -m "feat: add calibration scripts, stub YAMLs, and evaluation utilities"
```

---

## Success Criteria

After all four commits, the following must hold:

```
✓ git log --oneline -4  shows 4 clean commits
✓ ls rpi5_inference/ checkpoints/ dataset/  (no vla-robotic-arm-main/ exists)
✓ All 5 test-command imports print OK
✓ python3 -m rpi5_inference.main --dry-run  prints "ok" with no [FAIL] lines
✓ python3 -m py_compile on all 8 new/modified Python files exits 0
```
