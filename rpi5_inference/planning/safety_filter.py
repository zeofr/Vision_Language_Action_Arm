"""
Safety filter for joint commands.

Three sequential stages applied in order:
  1. Singularity check  – rejects configurations where the Jacobian
                          determinant is near zero (arm fully extended
                          or fully folded).
  2. Workspace clamp    – if FK position is outside the declared
                          Cartesian box, clamps the position to the
                          box boundary and re-solves IK.
  3. Joint limit clamp  – hard-clips every joint to its min/max.

All 4 joints [J0, J1, J2, J3] are accepted; J3 (gripper) passes
through limits-only (no IK involved).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from rpi5_inference.planning.ik_solver import (
    A2, A3, D1,
    forward_kinematics,
    inverse_kinematics,
)

_CONFIG_PATH = Path(__file__).parents[2] / "rpi5_inference" / "config" / "arm_config.yaml"

# |det J| = |A2·A3·sin(J2)| below this → singularity
_SINGULARITY_EPS = 1e-3


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class SafetyFilter:
    """
    Stateless filter: call filter(joints) → filtered joints.

    joints : array-like [J0, J1, J2, J3] in degrees.
    Returns: np.ndarray [J0, J1, J2, J3] in degrees (always).
    """

    def __init__(self, config_path: Path = _CONFIG_PATH) -> None:
        cfg = _load_config(config_path)
        self.j_min = np.array(cfg["joint_limits_min_deg"], dtype=float)
        self.j_max = np.array(cfg["joint_limits_max_deg"], dtype=float)

        ws = cfg["workspace"]
        self.ws_x = tuple(ws["x_m"])
        self.ws_y = tuple(ws["y_m"])
        self.ws_z = tuple(ws["z_m"])

    # ── public API ────────────────────────────────────────────────────

    def filter(self, joints: np.ndarray) -> np.ndarray:
        """
        Run all three safety stages in order.  Always returns a
        4-element array.  Emits a UserWarning for each stage that fires.
        """
        joints = np.asarray(joints, dtype=float).copy()
        joints = self._stage1_singularity(joints)
        joints = self._stage2_workspace(joints)
        joints = self._stage3_joint_limits(joints)
        return joints

    def is_singular(self, joints: np.ndarray) -> bool:
        return self._jacobian_det(joints) < _SINGULARITY_EPS

    # ── internal stages ───────────────────────────────────────────────

    def _jacobian_det(self, joints: np.ndarray) -> float:
        """
        For the 2-R sagittal plane: det(J) = A2·A3·sin(J2).
        Singularities at J2 = 0° (full extension) or ±180°.
        """
        j2 = np.radians(joints[2])
        return abs(A2 * A3 * np.sin(j2))

    def _stage1_singularity(self, joints: np.ndarray) -> np.ndarray:
        det = self._jacobian_det(joints)
        if det < _SINGULARITY_EPS:
            warnings.warn(
                f"[SafetyFilter] Singularity detected (|det J|={det:.6f}). "
                f"Nudging J2 away from 0°.",
                UserWarning, stacklevel=4,
            )
            # Nudge J2 away from 0° by at least 1°
            if joints[2] >= 0:
                joints[2] = -(max(abs(joints[2]), 0.0) + 1.0)
            else:
                joints[2] = max(abs(joints[2]) + 1.0, 1.0)
        return joints

    def _stage2_workspace(self, joints: np.ndarray) -> np.ndarray:
        pos = forward_kinematics(joints[0], joints[1], joints[2])
        x, y, z = pos

        x_lo, x_hi = self.ws_x
        y_lo, y_hi = self.ws_y
        z_lo, z_hi = self.ws_z

        if not (x < x_lo or x > x_hi or
                y < y_lo or y > y_hi or
                z < z_lo or z > z_hi):
            return joints   # already inside — fast path

        x_c = float(np.clip(x, x_lo, x_hi))
        y_c = float(np.clip(y, y_lo, y_hi))
        z_c = float(np.clip(z, z_lo, z_hi))

        warnings.warn(
            f"[SafetyFilter] FK ({x*1e3:.1f},{y*1e3:.1f},{z*1e3:.1f}) mm "
            f"outside workspace. Clamping to "
            f"({x_c*1e3:.1f},{y_c*1e3:.1f},{z_c*1e3:.1f}) mm.",
            UserWarning, stacklevel=4,
        )

        solution = inverse_kinematics(x_c, y_c, z_c)
        if solution is None:
            warnings.warn(
                "[SafetyFilter] IK failed for clamped workspace point; "
                "falling through to joint-limit clamp only.",
                UserWarning, stacklevel=4,
            )
            return joints   # stage 3 will still hard-clip

        joints[0] = solution[0]
        joints[1] = solution[1]
        joints[2] = solution[2]
        return joints

    def _stage3_joint_limits(self, joints: np.ndarray) -> np.ndarray:
        clamped = np.clip(joints, self.j_min, self.j_max)
        violated = np.where(~np.isclose(joints, clamped, atol=1e-6))[0]
        if len(violated):
            warnings.warn(
                f"[SafetyFilter] Joint limit clamp on joints "
                f"{violated.tolist()}: "
                f"{joints[violated].round(2)} → {clamped[violated].round(2)}",
                UserWarning, stacklevel=4,
            )
        return clamped


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import warnings as _warnings

    sf = SafetyFilter()
    all_pass = True

    def run(label: str, raw, check_fn, expect_warn: bool = True):
        """Run filter, print result. check_fn(raw, result) must return True."""
        raw = np.asarray(raw, dtype=float)
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            result = sf.filter(raw)

        ok = check_fn(raw, result)
        status = "PASS" if ok else "FAIL"
        global all_pass
        if not ok:
            all_pass = False

        print(f"  [{status}] {label}")
        print(f"         in  : {np.round(raw, 2)}")
        print(f"         out : {np.round(result, 2)}")
        for w in caught:
            print(f"         warn: {str(w.message)[:110]}")
        print()

    # ── Stage 3: joint limit clamp ────────────────────────────────────
    print("=== Stage 3: joint limit clamp ===")

    # After all stages, each joint must be within its declared limits.
    def within_limits(_, f):
        return np.all(f >= sf.j_min - 1e-6) and np.all(f <= sf.j_max + 1e-6)

    run("J0 = +200° → output within [-150,+150]",
        [200.0, 10.0, -50.0, 45.0], within_limits)

    run("J0 = -200° → output within [-150,+150]",
        [-200.0, 10.0, -50.0, 45.0], within_limits)

    run("J1 = +90° → clamp to 60°",
        [0.0, 90.0, -40.0, 45.0],
        lambda _, f: abs(f[1] - 60.0) < 0.5 and within_limits(_, f))

    run("J1 = -60° → clamp to -30°",
        [0.0, -60.0, -40.0, 45.0],
        lambda _, f: abs(f[1] + 30.0) < 0.5 and within_limits(_, f))

    run("J2 = -150° → clamp to -120°",
        [0.0, 10.0, -150.0, 45.0],
        lambda _, f: abs(f[2] + 120.0) < 0.5 and within_limits(_, f))

    run("J2 = +80° → clamp to +30°",
        [0.0, 10.0, 80.0, 45.0],
        lambda _, f: abs(f[2] - 30.0) < 0.5 and within_limits(_, f))

    run("J3 = +120° → clamp to 90°",
        [0.0, 22.0, -70.0, 120.0],
        lambda _, f: abs(f[3] - 90.0) < 0.01 and within_limits(_, f))

    run("J3 = -20° → clamp to 0°",
        [0.0, 22.0, -70.0, -20.0],
        lambda _, f: abs(f[3] - 0.0) < 0.01 and within_limits(_, f))

    # ── Stage 1: singularity ──────────────────────────────────────────
    print("=== Stage 1: singularity check ===")

    run("J2 = 0.0° exactly → J2 nudged to -1°",
        [0.0, 10.0, 0.0, 45.0],
        lambda _, f: f[2] < -0.5)

    run("J2 = 0.1° → J2 nudged negative",
        [0.0, 10.0, 0.1, 45.0],
        lambda _, f: f[2] < 0.0)

    run("J2 = -0.05° → J2 nudged negative (more negative)",
        [0.0, 10.0, -0.05, 45.0],
        lambda _, f: f[2] < -0.5)

    # ── Stage 2: workspace clamp ──────────────────────────────────────
    print("=== Stage 2: workspace clamp ===")

    # J0=0, J1=10, J2=-50 → FK y=273.5mm > 260mm (workspace y_max).
    # After clamp to y=260mm, IK re-solves; verify FK of output is in workspace.
    def fk_in_workspace(_, f):
        pos = forward_kinematics(f[0], f[1], f[2])
        xf, yf, zf = pos
        x_ok = sf.ws_x[0] - 1e-3 <= xf <= sf.ws_x[1] + 1e-3
        y_ok = sf.ws_y[0] - 1e-3 <= yf <= sf.ws_y[1] + 1e-3
        z_ok = sf.ws_z[0] - 1e-3 <= zf <= sf.ws_z[1] + 1e-3
        return x_ok and y_ok and z_ok

    run("J1=10°,J2=-50° → FK y=273mm > 260mm; FK of output is in workspace",
        [0.0, 10.0, -50.0, 45.0], fk_in_workspace)

    # J0=0, J1=30°, J2=10° → FK z=312mm >> 120mm.
    # Clamping to (x, y, 120mm) leaves a solvable IK → verify output in workspace.
    run("J1=30°,J2=10° → FK z=312mm > 120mm; IK re-solves; FK of output is in workspace",
        [0.0, 30.0, 10.0, 45.0], fk_in_workspace)

    # ── Good joints pass through unchanged ───────────────────────────
    # J0=0, J1=22, J2=-70 → FK: x=0mm, y=247.7mm, z=32.5mm → inside workspace.
    print("=== Good joints pass through unchanged ===")

    good = np.array([0.0, 22.0, -70.0, 45.0])
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        result = sf.filter(good)

    unchanged = np.allclose(result, good, atol=1e-6)
    no_warn = len(caught) == 0
    ok = unchanged and no_warn
    if not ok:
        all_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] No warnings, output identical to input")
    print(f"         in       : {good}")
    print(f"         out      : {result}")
    print(f"         warnings : {len(caught)}")
    if caught:
        for w in caught:
            print(f"                    {str(w.message)[:110]}")

    print()
    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
