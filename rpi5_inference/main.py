#!/usr/bin/env python3
"""
VLA Robotic Arm — 8 Hz inference loop.

Modes
─────
Live  (default) : opens Teensy serial, camera, all models, runs until Ctrl-C.
Dry-run         : imports every module, runs lightweight smoke tests,
                  prints ok, exits 0.  No hardware required.

Usage
─────
  python -m rpi5_inference.main --dry-run
  python -m rpi5_inference.main --instruction "pick up the red cube"
  python -m rpi5_inference.main --port /dev/ttyACM0 --instruction "grab the blue cube"
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import warnings

import numpy as np

# ── constants ─────────────────────────────────────────────────────────────────
LOOP_HZ       = 8
LOOP_PERIOD_S = 1.0 / LOOP_HZ          # 125 ms
OVERRUN_MS    = LOOP_PERIOD_S * 1e3     # 125.0 — threshold for overrun warning

DEFAULT_PORT        = "/dev/ttyUSB0"   # ESP32 sensor telemetry
DEFAULT_SERVO_PORT  = "/dev/ttyACM0"  # SmartElex servo bus
DEFAULT_INSTRUCTION = "pick up the red cube"
DEFAULT_CHECKPOINT  = "checkpoints/yolov8n_vla/weights/best.pt"
DEFAULT_VLA_CHECKPOINT = "checkpoints/vla_policy_traced.pt"

# Safe hold position used when no target is visible or IK fails.
_HOLD_JOINTS = np.array([0.0, 22.0, -70.0, 45.0])


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vla-arm",
        description="Vision-Language-Action robotic arm inference loop",
    )
    p.add_argument("--port",        default=DEFAULT_PORT,
                   help=f"ESP32 serial port for sensor telemetry (default: {DEFAULT_PORT})")
    p.add_argument("--servo-port",  default=DEFAULT_SERVO_PORT,
                   help=f"SmartElex servo port (default: {DEFAULT_SERVO_PORT})")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION,
                   help="Natural-language task instruction")
    p.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT,
                   help="YOLOv8 checkpoint path")
    p.add_argument("--dry-run",     action="store_true",
                   help="Import all modules, run smoke tests, exit without hardware")
    return p


# ── dry-run ───────────────────────────────────────────────────────────────────

def _dry_run() -> int:
    """
    Import every module, run lightweight in-process smoke tests, report status.
    No hardware connections are made.  Returns 0 on full success, 1 otherwise.
    """
    all_ok = True

    def report(label: str, fn) -> None:
        nonlocal all_ok
        try:
            fn()
            print(f"  [ok]   {label}")
        except Exception as exc:
            print(f"  [FAIL] {label}: {exc}")
            all_ok = False

    # ── module imports ────────────────────────────────────────────────
    print("── module imports ──────────────────────────────────────────")

    report("rpi5_inference.planning.ik_solver",
           lambda: __import__("rpi5_inference.planning.ik_solver",    fromlist=[""]))
    report("rpi5_inference.planning.safety_filter",
           lambda: __import__("rpi5_inference.planning.safety_filter", fromlist=[""]))
    report("rpi5_inference.vla.skill_predictor",
           lambda: __import__("rpi5_inference.vla.skill_predictor",    fromlist=[""]))
    report("rpi5_inference.language.language_encoder",
           lambda: __import__("rpi5_inference.language.language_encoder", fromlist=[""]))
    report("rpi5_inference.perception.yolo_detector",
           lambda: __import__("rpi5_inference.perception.yolo_detector",  fromlist=[""]))
    report("rpi5_inference.perception.pose_estimation",
           lambda: __import__("rpi5_inference.perception.pose_estimation", fromlist=[""]))
    report("rpi5_inference.comms.teensy_serial",
           lambda: __import__("rpi5_inference.comms.teensy_serial",   fromlist=[""]))

    # ── lightweight smoke tests ───────────────────────────────────────
    print("── smoke tests (no hardware) ───────────────────────────────")

    # IK round-trip within 1 mm
    from rpi5_inference.planning.ik_solver import (
        inverse_kinematics, forward_kinematics,
    )
    def _ik_smoke():
        target = (0.0, 0.220, 0.045)
        j = inverse_kinematics(*target)
        assert j is not None, "IK returned None for a reachable workspace point"
        recon = forward_kinematics(*j)
        err_mm = float(np.linalg.norm(recon - np.array(target))) * 1e3
        assert err_mm < 1.0, f"FK(IK(target)) residual {err_mm:.3f} mm > 1 mm"
    report("IK: (0.0, 0.220, 0.045) round-trips within 1 mm", _ik_smoke)

    # SafetyFilter loads arm_config.yaml and clamps a bad joint
    from rpi5_inference.planning.safety_filter import SafetyFilter
    def _sf_smoke():
        sf = SafetyFilter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = sf.filter(np.array([200.0, 10.0, -50.0, 45.0]))
        assert np.all(out >= sf.j_min - 1e-6), "output violates lower limits"
        assert np.all(out <= sf.j_max + 1e-6), "output violates upper limits"
    report("SafetyFilter: loads config, clamps J0=200° within limits", _sf_smoke)

    # SkillStateMachine full cycle
    from rpi5_inference.vla.skill_predictor import SkillStateMachine, Skill
    def _sm_smoke():
        sm = SkillStateMachine()
        sm.advance(); sm.advance(); sm.advance()
        assert sm.state is Skill.PLACE, f"expected PLACE, got {sm.state}"
        assert sm.done
        sm.reset()
        assert sm.state is Skill.REACH
    report("SkillStateMachine: REACH→GRASP→LIFT→PLACE, reset", _sm_smoke)

    # PoseEstimator with dummy calibration
    from rpi5_inference.perception.pose_estimation import PoseEstimator
    def _pe_smoke():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pe = PoseEstimator()
        tof = np.full((8, 8), 50.0)
        pose = pe.compute_pick_pose((320.0, 310.0), tof)
        assert pose.shape == (3,), f"expected shape (3,), got {pose.shape}"
    report("PoseEstimator: compute_pick_pose returns (3,) array", _pe_smoke)

    # Teensy dtype sizes — hard invariant
    from rpi5_inference.comms.teensy_serial import TELEMETRY_DTYPE, COMMAND_DTYPE
    def _dtype_smoke():
        assert TELEMETRY_DTYPE.itemsize == 250, \
            f"TELEMETRY_DTYPE is {TELEMETRY_DTYPE.itemsize} B, expected 250"
        assert COMMAND_DTYPE.itemsize == 20, \
            f"COMMAND_DTYPE is {COMMAND_DTYPE.itemsize} B, expected 20"
    report("TeensySerial dtypes: 250 B telemetry / 20 B command", _dtype_smoke)

    # YOLODetector import + CLASS_NAMES present (no model load in dry-run)
    from rpi5_inference.perception.yolo_detector import CLASS_NAMES, CONF_THRESHOLD
    def _yolo_smoke():
        assert CLASS_NAMES == ["red_cube", "blue_cube", "green_cube"]
        assert 0.0 < CONF_THRESHOLD < 1.0
    report("YOLODetector: CLASS_NAMES and CONF_THRESHOLD defined", _yolo_smoke)

    # LanguageEncoder import + constant check (no model load)
    from rpi5_inference.language.language_encoder import MODEL_NAME, EMBED_DIM
    def _enc_smoke():
        assert "flan-t5" in MODEL_NAME.lower(), f"unexpected model: {MODEL_NAME}"
        assert EMBED_DIM == 512
    report("LanguageEncoder: model name and embed dim constants", _enc_smoke)

    # Loop-timing sanity
    def _timing_smoke():
        assert abs(LOOP_PERIOD_S - 0.125) < 1e-9, "LOOP_PERIOD_S != 0.125"
        assert abs(OVERRUN_MS - 125.0) < 1e-9,    "OVERRUN_MS != 125.0"
    report("Loop timing: 8 Hz → 125 ms period", _timing_smoke)

    print()
    if all_ok:
        print("ok")
    else:
        print("SOME CHECKS FAILED", file=sys.stderr)
    return 0 if all_ok else 1


# ── helpers ───────────────────────────────────────────────────────────────────

def _gripper_pct(state) -> int:
    """Gripper %open from skill state: open during REACH/PLACE, closed during GRASP/LIFT."""
    from rpi5_inference.vla.skill_predictor import Skill
    return 0 if state in (Skill.GRASP, Skill.LIFT) else 100


def setup_pipeline():
    """Delegate to run_eval.setup_pipeline(). Imported by evaluation scripts."""
    from rpi5_inference.evaluation.run_eval import setup_pipeline as _sp
    return _sp()


# ── live inference loop ───────────────────────────────────────────────────────

def run_loop(args) -> int:
    """
    Full 8 Hz inference loop.  Runs until KeyboardInterrupt.
    Returns exit code 0.
    """
    from rpi5_inference.planning.ik_solver       import inverse_kinematics
    from rpi5_inference.planning.safety_filter   import SafetyFilter
    from rpi5_inference.vla.skill_predictor      import SkillStateMachine
    from rpi5_inference.language.language_encoder import LanguageEncoder
    from rpi5_inference.perception.yolo_detector  import YOLODetector
    from rpi5_inference.perception.pose_estimation import PoseEstimator
    from rpi5_inference.comms.teensy_serial       import TeensySerial
    from rpi5_inference.comms.servo_driver        import ServoDriver
    from rpi5_inference.perception.camera_manager import CameraManager
    from rpi5_inference.vla.vla_policy            import VLARuntime
    from rpi5_inference.vla.action_generator      import ActionGenerator

    log = logging.getLogger("vla.loop")

    log.info("Initialising components…")
    enc = LanguageEncoder()
    det = YOLODetector(args.checkpoint)
    pe  = PoseEstimator()
    sf  = SafetyFilter()
    sm  = SkillStateMachine()
    ts     = TeensySerial(args.port)
    servo  = ServoDriver(args.servo_port)
    camera = None
    camera = CameraManager()
    vla        = VLARuntime(DEFAULT_VLA_CHECKPOINT, enc)
    action_gen = ActionGenerator()

    # Pre-encode once; cache hit on every tick.
    lang_vec = enc.encode(args.instruction)
    log.info("Instruction: '%s'  embedding shape: %s",
             args.instruction, lang_vec.shape)
    log.info("Starting inference loop at %d Hz.  Press Ctrl-C to stop.", LOOP_HZ)

    tick = 0
    overruns = 0

    try:
        while True:
            t0 = time.monotonic()
            tick += 1

            # ── 1. Telemetry ──────────────────────────────────────────
            telem = ts.latest_telemetry
            if telem is not None:
                contact  = bool(telem["contact_flag"][0])
                tof_grid = telem["tof_grid"][0].astype(np.float64).reshape(8, 8)
            else:
                contact  = False
                tof_grid = np.full((8, 8), 80.0)   # safe fallback [mm]

            sm.notify_contact(contact)

            # ── 2. Camera → detection ─────────────────────────────────
            frame = camera.latest_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            detections = det.detect(frame)
            target_det = det.match_instruction(detections, args.instruction)

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

            # ── 3. Pose → IK → safety ────────────────────────────────
            joints_4 = _HOLD_JOINTS.copy()

            if target_det is not None:
                try:
                    pose = pe.compute_pick_pose(target_det.centroid, tof_grid)
                    x, y, z = pose
                    ik_result = inverse_kinematics(x, y, z)
                    if ik_result is not None:
                        joints_4 = np.append(ik_result, 45.0)   # append gripper
                    else:
                        log.warning("Tick %d: IK unreachable (%.3f, %.3f, %.3f)",
                                    tick, x, y, z)
                except Exception as exc:
                    log.warning("Tick %d: pose/IK error: %s", tick, exc)

            # IK-primary: VLA delta is a small additive correction (×0.1 while mock)
            joints_4[:4] = joints_4[:4] + delta_step0 * 0.1

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                safe_joints = sf.filter(joints_4)

            # ── 4. Command — send directly to servos via SmartElex USB ──
            gripper_deg = 0.0 if _gripper_pct(sm.state) == 0 else 95.0
            servo.sync_write(
                ids=[1, 2, 3, 4, 5],
                positions_deg=[
                    float(safe_joints[0]),   # J0 base
                    float(safe_joints[1]),   # J1a shoulder
                    float(safe_joints[1]),   # J1b shoulder (coupled)
                    float(safe_joints[2]),   # J2 elbow
                    gripper_deg,             # J3 gripper
                ],
            )

            # ── 5. Timing ─────────────────────────────────────────────
            elapsed_ms = (time.monotonic() - t0) * 1e3
            if elapsed_ms > OVERRUN_MS:
                overruns += 1
                log.warning("OVERRUN tick %d: %.1f ms  (budget %.0f ms, total overruns: %d)",
                            tick, elapsed_ms, OVERRUN_MS, overruns)

            sleep_s = max(0.0, LOOP_PERIOD_S - elapsed_ms * 1e-3)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        log.info("Interrupted after %d ticks (%d overruns). Shutting down.", tick, overruns)
    finally:
        ts.close()
        servo.close()
        if camera is not None:
            camera.close()

    return 0


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    args = _build_parser().parse_args()
    return _dry_run() if args.dry_run else run_loop(args)


if __name__ == "__main__":
    sys.exit(main())
