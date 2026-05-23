# VLA Robotic Arm — End-to-End Implementation Design

**Date:** 2026-05-21  
**Author:** Team  
**Approach:** Option B — Incremental hardware verification, then calibration, demos, training, inference

---

## Current State

- Arm assembled (WaveShare RoArm-M2-S style, 3D printed + STS3215 servos)
- Sensors arrived (ISM330DHCX, VL53L5CX) — not yet wired
- ESP32-WROOM-32 firmware: written, compiled, not flashed
- SmartElex Serial Bus Servo Driver Board: servos daisy-chained into it
- 4S LiPo battery (14.7V nominal, 16.8V fully charged)
- Dev machine: WSL2 Ubuntu 22.04 on Windows 11 (edit + git push here)
- Hardware host: Raspberry Pi 5 8GB (git pull + flash + run everything here)
- Friend's AI/ML code: ~60% done in `vla-robotic-arm-main/`

---

## Workflow

Every change follows this path:

```
WSL2 (edit) → git push → RPi5: git pull → pio run --target upload → pio device monitor
```

PlatformIO install on RPi5 (once):
```bash
pip3 install platformio
```

From `firmware/` on RPi5:
```bash
pio run --target upload        # compile + flash
pio device monitor -b 2000000  # serial monitor
```

ESP32 appears as `/dev/ttyUSB0` or `/dev/ttyACM0` on RPi5.

---

## Power Architecture

```
4S LiPo (14.7V nom, 16.8V max)
         │
         ├── Buck #1 (XL4016/LM2596) → 12V @ 8A+ → SmartElex board → 5× STS3215 servos
         │                              1000µF 16V cap across servo terminals
         │
         └── Buck #2 (LM2596) → 5V @ 5A → RPi5 (USB-C)
                                            └── RPi5 USB-A → ESP32
```

- Set both buck outputs with multimeter **before** connecting any load
- Servo rail and compute rail share only common ground at battery negative
- ESP32 powered from RPi5 USB — never connect ESP32 VIN to 12V

---

## Hardware Wiring

### Servo Bus
SmartElex board handles half-duplex switching internally. ESP32 connects via standard UART only.

| ESP32 | SmartElex Board |
|---|---|
| GPIO17 (UART2 TX) | RX |
| GPIO16 (UART2 RX) | TX |
| GND | GND |

12V into SmartElex DC barrel jack from Buck #1.

### ISM330DHCX (SPI — VSPI)

| ESP32 | IMU |
|---|---|
| GPIO23 (MOSI) | SDI |
| GPIO19 (MISO) | SDO |
| GPIO18 (SCK) | SCL |
| GPIO5 (CS) | CS |
| 3V3 | VCC |
| GND | GND |

### VL53L5CX (I2C)

| ESP32 | ToF |
|---|---|
| GPIO21 (SDA) | SDA |
| GPIO22 (SCL) | SCL |
| GPIO27 | LPn (power enable, pull HIGH) |
| GPIO26 | INT (active LOW, data-ready) |
| 3V3 | VDD |
| GND | GND |

### Forbidden pins (never use)
- GPIO6–11: internal SPI flash
- GPIO0, 2, 12, 15: boot-strapping pins

---

## Firmware Fix Required

`firmware/src/servo_bus.cpp` — `begin_tx()` and `begin_rx()` currently toggle GPIO4 for direction control. Since the SmartElex board handles this internally, make both functions no-ops. GPIO4 can be left unconnected.

---

## Step-by-Step Implementation Plan

### Phase 0 — RPi5 Setup
- Install PlatformIO on RPi5
- Confirm `git pull` + `pio run --target upload` works over USB to ESP32

### Phase 1 — Servo Bus Verification
- Wire SmartElex → ESP32 (3 wires: TX, RX, GND)
- Create `firmware/test_sketches/test_servo_ping.cpp`
- Pings IDs 0x01–0x05, prints which respond
- **Pass:** all 5 print "FOUND"

### Phase 2 — IMU Verification
- Wire ISM330DHCX to ESP32 SPI (6 wires)
- Create `firmware/test_sketches/test_imu_whoami.cpp`
- Reads WHO_AM_I register
- **Pass:** prints "IMU OK: 0x6B"

### Phase 3 — ToF Verification
- Wire VL53L5CX to ESP32 I2C (6 wires)
- Create `firmware/test_sketches/test_tof_distance.cpp`
- Starts 8×8 mode at 15Hz, prints center zone average
- **Pass:** prints ~200mm with flat board at 200mm distance

### Phase 4 — Full Firmware Flash
- Apply GPIO4 no-op fix to `servo_bus.cpp`
- Flash full firmware from `firmware/src/`
- Verify on RPi5: 250-byte telemetry packets arriving at 50Hz
- Verify: servo positions readable, IMU data streaming, ToF updating at 15Hz
- Verify: contact oracle RMS visible in serial output

### Phase 5 — Calibration (Team, on RPi5)
Extract scripts from `Team_HARDWARE_EMBEDDED_WORKPLAN.md` into `calibration/`:

| Script | Output file | Pass criterion |
|---|---|---|
| `camera_calibrate.py` | `camera_intrinsics.yaml` | Reprojection error < 0.5px |
| `overhead_height_calib.py` | `overhead_height.yaml` | Verification error < 5mm |
| `wrist_tof_calib.py` | `wrist_tof_offset.yaml` | Depth predictions ±5mm |
| `compute_T_cam_base.py` | `camera_to_base_transform.yaml` | Mean error < 8mm |
| Manual measurement | `joint_limits.yaml` | Arm cannot exceed limits |
| Manual measurement | `servo_zero_offsets.yaml` | All joints read 0° at true zero |

### Phase 6 — Teleoperation + Demo Collection (Team, on RPi5)
- Extract `teleop/teleop_interface.py` from `Team_HARDWARE_EMBEDDED_WORKPLAN.md` (code already written there, just needs saving as a file)
- Collect 30 demonstrations across 3 task types (pick-place, stacking, color sorting)
- Output: `demos/demo_001…demo_030.h5`
- Each demo: ~8s, synchronized 50Hz telemetry + 30fps overhead RGB

### Phase 7 — VLA Training (Friend, on Colab)
Unblocked once `demos/` exists and calibration YAMLs are present. Training code already written in friend's `VLA_Training.ipynb` notebook.
- Run skill segmentation on 30 demos
- Fine-tune SmolVLA-450M + LoRA (rank 16, AdamW, 5 epochs)
- Target: skill accuracy ≥ 75%, action MSE < 5 deg²
- Output: `checkpoints/smolvla_lora_best/`, `skill_head_best.pt`, `action_head_best.pt`

### Phase 8 — Inference Implementation (Friend, on WSL2 → RPi5)
Three empty files need implementing:

| File | What it does |
|---|---|
| `rpi5_inference/vla/vla_policy.py` | Loads TorchScript checkpoint, runs predict() returning skill + delta joints + chunk |
| `rpi5_inference/vla/action_generator.py` | Decodes 8-step action chunk, feeds to TeensySerial |
| `rpi5_inference/evaluation/run_eval.py` | Runs 3 task types × N trials, records success/failure |

### Phase 9 — Integration Test
- RPi5 runs `main.py` with real ESP32 connected
- Confirm 8Hz inference loop: telemetry in → YOLO → pose → VLA → IK → command out
- Confirm arm moves in response to language instruction
- Confirm skill transitions (REACH → GRASP → LIFT → PLACE) fire correctly

### Phase 10 — Evaluation
- Task 1: Single-object pick-place, 60 trials, target ≥ 85% success
- Task 2: Block stacking, 30 trials, target ≥ 75% success
- Task 3: Language-conditioned color sorting, 40 trials, target ≥ 80% success
- Ablation: no wrist ToF, no IMU contact, no skill decomposition

---

## Dependencies Between Steps

```
Phase 0 → Phase 1 → Phase 2 → Phase 3 → Phase 4
Phase 4 → Phase 5 → Phase 6 → Phase 7 → Phase 8 → Phase 9 → Phase 10
Phase 7 and Phase 8 can overlap once Phase 6 is complete
```

---

## What's Already Done (Do Not Redo)

- All 8 firmware modules written and compiled (`firmware/src/`)
- Friend's dataset pipeline, perception, planning, comms, dashboard modules written
- YOLOv8-nano fine-tuned checkpoint exists (`checkpoints/yolov8n_vla/weights/best.pt`)
- Skill state machine written (`rpi5_inference/vla/skill_predictor.py`)
