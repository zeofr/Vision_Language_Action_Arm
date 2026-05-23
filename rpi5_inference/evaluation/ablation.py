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
