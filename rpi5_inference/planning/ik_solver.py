"""
Closed-form 3-DOF inverse kinematics for the VLA robotic arm.

DH parameters (meters):
  d1 = 0.125  (base riser + arm offset)
  a2 = 0.130  (upper arm link)
  a3 = 0.190  (forearm link)

Joint convention (degrees):
  J0 – base yaw,        limits [-150, +150]
  J1 – shoulder pitch,  limits [ -30,  +60]
  J2 – elbow pitch,     limits [-120,  +30]

Coordinate frame:
  +Y forward (depth), +X right, +Z up.
  J0 = atan2(x, y)  → arm points along +Y when J0=0.
"""

import numpy as np
from typing import Optional

# DH parameters (meters)
D1: float = 0.125
A2: float = 0.130
A3: float = 0.190

# IK operates on the first 3 joints only
_J_MIN = np.array([-150.0, -30.0, -120.0])
_J_MAX = np.array([ 150.0,  60.0,   30.0])

# Tolerance for limit checking (degrees) to absorb floating-point noise
_LIMIT_TOL = 0.5


def forward_kinematics(j0_deg: float, j1_deg: float, j2_deg: float) -> np.ndarray:
    """
    Returns end-effector position [x, y, z] in metres.
    Does not enforce joint limits — useful for verification.
    """
    j0 = np.radians(j0_deg)
    j1 = np.radians(j1_deg)
    j2 = np.radians(j2_deg)

    # Radial reach and vertical offset from shoulder
    r  = A2 * np.cos(j1) + A3 * np.cos(j1 + j2)
    dz = A2 * np.sin(j1) + A3 * np.sin(j1 + j2)

    x = r * np.sin(j0)
    y = r * np.cos(j0)
    z = D1 + dz
    return np.array([x, y, z])


def inverse_kinematics(
    x: float, y: float, z: float
) -> Optional[np.ndarray]:
    """
    Closed-form IK.  Returns [J0, J1, J2] in degrees or None if
    the target is geometrically unreachable or violates joint limits.

    Tries elbow-down (J2 < 0) first — the natural pick posture —
    then elbow-up (J2 > 0).  Returns the first solution whose joints
    fall within limits.
    """
    # ── base yaw ────────────────────────────────────────────────────
    j0 = np.arctan2(x, y)

    # ── 2-D problem in the sagittal plane ────────────────────────────
    r  = np.sqrt(x**2 + y**2)   # horizontal distance from base axis
    dz = z - D1                  # height relative to shoulder joint

    D_sq = r**2 + dz**2
    D    = np.sqrt(D_sq)

    # Geometric reachability (add tiny epsilon for floating-point)
    if D > A2 + A3 + 1e-6:
        return None   # too far
    if D < abs(A2 - A3) - 1e-6:
        return None   # too close (arm fully folded can't reach)

    cos_j2 = (D_sq - A2**2 - A3**2) / (2.0 * A2 * A3)
    cos_j2 = np.clip(cos_j2, -1.0, 1.0)   # guard against rounding

    best: Optional[np.ndarray] = None

    for sign in (-1, +1):          # elbow-down first, then elbow-up
        j2 = sign * np.arccos(cos_j2)

        k1 = A2 + A3 * np.cos(j2)
        k2 = A3 * np.sin(j2)
        j1 = np.arctan2(dz, r) - np.arctan2(k2, k1)

        joints_deg = np.array([np.degrees(j0),
                                np.degrees(j1),
                                np.degrees(j2)])

        if (np.all(joints_deg >= _J_MIN - _LIMIT_TOL) and
                np.all(joints_deg <= _J_MAX + _LIMIT_TOL)):
            best = np.clip(joints_deg, _J_MIN, _J_MAX)
            break   # accept first valid solution

    return best


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    PASS_THRESHOLD_M = 0.001   # 1 mm

    # Generate test targets via FK with known-good joint angles so we are
    # guaranteed to be inside the reachable workspace.
    test_joints = [
        ( 0.0,  5.0, -40.0),
        (20.0,  5.0, -40.0),
        (-20.0,  5.0, -40.0),
        ( 0.0, 10.0, -50.0),
        (30.0, 10.0, -50.0),
        (-30.0, 10.0, -50.0),
        ( 0.0, 20.0, -80.0),
        (15.0, 20.0, -80.0),
        (-15.0, 20.0, -80.0),
        ( 0.0, 30.0, -100.0),
    ]

    print(f"{'#':>2}  {'Target (mm)':>30}  {'Error (mm)':>10}  {'J0':>7} {'J1':>7} {'J2':>7}  Status")
    print("-" * 90)

    all_pass = True
    for i, (j0, j1, j2) in enumerate(test_joints):
        target = forward_kinematics(j0, j1, j2)
        x, y, z = target

        result = inverse_kinematics(x, y, z)

        if result is None:
            print(f"{i+1:>2}  ({x*1e3:+7.1f},{y*1e3:+7.1f},{z*1e3:+7.1f}) mm  "
                  f"{'N/A':>10}  FAIL (no solution)")
            all_pass = False
            continue

        recon = forward_kinematics(*result)
        err_m = np.linalg.norm(recon - target)
        err_mm = err_m * 1e3
        status = "PASS" if err_m < PASS_THRESHOLD_M else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"{i+1:>2}  ({x*1e3:+7.1f},{y*1e3:+7.1f},{z*1e3:+7.1f}) mm  "
              f"{err_mm:>10.4f}  "
              f"{result[0]:+7.2f} {result[1]:+7.2f} {result[2]:+7.2f}  {status}")

    print("-" * 90)
    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
