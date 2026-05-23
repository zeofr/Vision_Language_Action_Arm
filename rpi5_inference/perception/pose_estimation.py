"""
Pose estimation for pick targets.

Two complementary sub-systems:

  overhead_xy(centroid_px)
    Pinhole back-projection from the overhead/tilted camera image plane
    to the known table-height plane (Z_table). Uses camera intrinsics
    (fx, fy, cx, cy) and extrinsics (R, t) from calibration yamls.

  wrist_tof_z(tof_grid)
    Z estimate from the wrist VL53L5CX TOF sensor grid.
    Takes the median of the 2×2 centre cells of an NxN grid [mm → m].

  apply_homography(centroid_px)
    Direct pixel → robot-frame XY via a 3×3 homography computed from the
    four ABCD reference dots printed on the workspace paper.

    World coordinates (robot frame, metres):
      A: x=-0.180, y=0.100  (left,  near)
      B: x=+0.180, y=0.100  (right, near)
      C: x=+0.180, y=0.220  (right, far )
      D: x=-0.180, y=0.220  (left,  far )

  compute_pick_pose(centroid_px, tof_grid)
    Combines apply_homography (XY) + wrist_tof_z (Z).
    Returns (x, y, z) in metres in the robot frame.

Calibration yamls (rpi5_inference/calibration/):
  camera_intrinsics.yaml  – fx, fy, cx, cy, dist_coeffs
  camera_extrinsics.yaml  – rvec, tvec  (Rodrigues, metres)
  homography_dots.yaml    – pixel coords of A, B, C, D

If any yaml is missing, hardcoded dummy values are used so that the
module is fully functional before Pi Camera calibration is performed.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

# ── calibration file paths ────────────────────────────────────────────────────
_CALIB_DIR = Path(__file__).parents[2] / "rpi5_inference" / "calibration"
_INTRINSICS_PATH  = _CALIB_DIR / "camera_intrinsics.yaml"
_EXTRINSICS_PATH  = _CALIB_DIR / "camera_extrinsics.yaml"
_HOMOGRAPHY_PATH  = _CALIB_DIR / "homography_dots.yaml"

# ── fixed world positions of the four reference dots (metres) ─────────────────
# A=near-left, B=near-right, C=far-right, D=far-left
_DOT_WORLD: np.ndarray = np.array([
    [-0.180, 0.100],   # A
    [ 0.180, 0.100],   # B
    [ 0.180, 0.220],   # C
    [-0.180, 0.220],   # D
], dtype=np.float64)

# Known table height in the robot world frame (metres).
Z_TABLE: float = 0.005   # cubes sit ~5 mm above workspace paper

# ── dummy calibration values (phone camera, 65 cm above, 45° tilt) ───────────
_DUMMY_INTRINSICS = {
    "fx": 1050.0, "fy": 1050.0, "cx": 320.0, "cy": 240.0,
    "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
}
# Camera rotated ~45° around X (tilt down), translated 65 cm up, 20 cm left.
_DUMMY_RVEC = np.array([np.pi / 4, 0.0, 0.0], dtype=np.float64)
_DUMMY_TVEC = np.array([-0.20, 0.0, 0.65], dtype=np.float64)

# Dummy pixel positions of ABCD for a 640×480 frame under the 45° tilt.
# These form a realistic trapezoid (near row lower/wider, far row higher/narrower).
_DUMMY_DOT_PX: np.ndarray = np.array([
    [100.0, 400.0],   # A – near-left
    [540.0, 400.0],   # B – near-right
    [480.0, 220.0],   # C – far-right
    [160.0, 220.0],   # D – far-left
], dtype=np.float64)


def _load_yaml(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def _build_camera_matrix(intr: dict) -> np.ndarray:
    return np.array([
        [intr["fx"],      0.0, intr["cx"]],
        [0.0,       intr["fy"], intr["cy"]],
        [0.0,             0.0,        1.0],
    ], dtype=np.float64)


class PoseEstimator:
    """
    Estimates pick-pose (x, y, z) in metres from camera observations.

    Instantiation loads calibration yamls; falls back to dummy values
    if files are absent.  Call compute_pick_pose() for the combined result.
    """

    def __init__(self, calib_dir: Path = _CALIB_DIR) -> None:
        self._load_intrinsics(calib_dir / "camera_intrinsics.yaml")
        self._load_extrinsics(calib_dir / "camera_extrinsics.yaml")
        self._load_homography(calib_dir / "homography_dots.yaml")

    # ── loading ───────────────────────────────────────────────────────

    def _load_intrinsics(self, path: Path) -> None:
        d = _load_yaml(path)
        if d is None:
            warnings.warn(
                f"[PoseEstimator] {path.name} not found — using dummy intrinsics.",
                UserWarning, stacklevel=3,
            )
            d = _DUMMY_INTRINSICS
        self.K    = _build_camera_matrix(d)
        self.dist = np.array(d["dist_coeffs"], dtype=np.float64)

    def _load_extrinsics(self, path: Path) -> None:
        d = _load_yaml(path)
        if d is None:
            warnings.warn(
                f"[PoseEstimator] {path.name} not found — using dummy extrinsics.",
                UserWarning, stacklevel=3,
            )
            rvec = _DUMMY_RVEC
            tvec = _DUMMY_TVEC
        else:
            rvec = np.array(d["rvec"], dtype=np.float64)
            tvec = np.array(d["tvec"], dtype=np.float64)

        R, _ = cv2.Rodrigues(rvec)
        self.R    = R           # 3×3 rotation: camera←world
        self.t    = tvec        # translation: camera origin in world frame
        self.R_cw = R           # camera-from-world rotation
        self.t_cw = tvec        # camera origin in world coords

    def _load_homography(self, path: Path) -> None:
        d = _load_yaml(path)
        if d is None:
            warnings.warn(
                f"[PoseEstimator] {path.name} not found — using dummy dot pixels.",
                UserWarning, stacklevel=3,
            )
            dot_px = _DUMMY_DOT_PX
        else:
            dot_px = np.array(
                [d["A_px"], d["B_px"], d["C_px"], d["D_px"]],
                dtype=np.float64,
            )

        # H maps image pixels → robot-frame XY (in metres).
        # cv2.findHomography expects (N,1,2) or (N,2).
        H, status = cv2.findHomography(dot_px, _DOT_WORLD)
        if H is None or np.count_nonzero(status) < 4:
            raise RuntimeError(
                "Homography computation failed — check dot pixel coords."
            )
        self.H: np.ndarray = H   # (3,3) float64

    # ── public API ────────────────────────────────────────────────────

    def apply_homography(self, centroid_px: tuple[float, float]) -> np.ndarray:
        """
        Map a pixel centroid to robot-frame XY via the precomputed
        homography.  Returns np.ndarray([x, y]) in metres.
        """
        u, v  = centroid_px
        src   = np.array([[[u, v]]], dtype=np.float64)         # (1,1,2)
        dst   = cv2.perspectiveTransform(src, self.H)           # (1,1,2)
        return dst[0, 0].astype(np.float64)                    # [x, y]

    def overhead_xy(
        self,
        centroid_px: tuple[float, float],
        z_table: float = Z_TABLE,
    ) -> np.ndarray:
        """
        Back-project pixel centroid to robot-frame XY using pinhole
        model + known table height.

        Steps:
          1. Undistort pixel to normalised image coordinates.
          2. Form ray in camera frame.
          3. Transform ray to world frame via R^T.
          4. Solve for scale λ such that world-point Z == z_table.
          5. Return world XY.
        """
        u, v = centroid_px
        # Undistort
        uv_dist = np.array([[[u, v]]], dtype=np.float64)
        uv_und  = cv2.undistortPoints(uv_dist, self.K, self.dist)  # normalised (no P)
        xn, yn  = float(uv_und[0, 0, 0]), float(uv_und[0, 0, 1])

        # Ray in camera frame (unit direction doesn't matter — scale later)
        ray_cam = np.array([xn, yn, 1.0])

        # Camera pose: origin in world = R^T @ (-t) if t is world-expressed;
        # but our convention stores t_cw = camera-origin in world coords.
        cam_origin_world = self.t_cw                     # (3,)
        ray_world        = self.R_cw.T @ ray_cam         # (3,)

        # Solve: cam_origin_world[2] + λ * ray_world[2] == z_table
        if abs(ray_world[2]) < 1e-9:
            raise ValueError(
                "Camera ray is parallel to the table plane — cannot intersect."
            )
        lam = (z_table - cam_origin_world[2]) / ray_world[2]
        world_pt = cam_origin_world + lam * ray_world   # (3,)
        return world_pt[:2].astype(np.float64)          # [x, y]

    def wrist_tof_z(self, tof_grid: np.ndarray) -> float:
        """
        Estimate table-relative Z from a wrist TOF sensor grid.

        tof_grid : (N, N) array of distances in mm (N ≥ 2).
                   Values ≤ 0 are treated as invalid and excluded.

        Returns Z in metres (median of the centre 2×2 cells).
        Raises ValueError if all centre cells are invalid.
        """
        grid = np.asarray(tof_grid, dtype=np.float64)
        N = grid.shape[0]
        if grid.ndim != 2 or grid.shape[1] != N or N < 2:
            raise ValueError(f"tof_grid must be square NxN with N≥2, got {grid.shape}")

        mid = N // 2
        centre = grid[mid - 1 : mid + 1, mid - 1 : mid + 1].ravel()  # 4 values
        valid  = centre[centre > 0]
        if valid.size == 0:
            raise ValueError("All TOF centre cells are invalid (≤ 0).")

        return float(np.median(valid)) * 1e-3   # mm → m

    def compute_pick_pose(
        self,
        centroid_px: tuple[float, float],
        tof_grid:    np.ndarray,
    ) -> np.ndarray:
        """
        Combine apply_homography (XY) + wrist_tof_z (Z).
        Returns np.ndarray([x, y, z]) in metres, robot frame.
        """
        xy = self.apply_homography(centroid_px)
        z  = self.wrist_tof_z(tof_grid)
        return np.array([xy[0], xy[1], z], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import warnings as _warnings

    all_pass = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        global all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    # ── instantiation with dummy calibration ──────────────────────────
    print("=== Instantiation (no calibration yamls — dummy values) ===")
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        pe = PoseEstimator()

    check("PoseEstimator created without exception", True)
    check("Three dummy-value warnings emitted", len(caught) == 3)
    check("Homography matrix is 3×3", pe.H.shape == (3, 3))
    print()

    # ── apply_homography: ABCD dots must round-trip ───────────────────
    print("=== apply_homography: ABCD reference dots round-trip ===")
    dot_labels   = ["A", "B", "C", "D"]
    dot_px_list  = _DUMMY_DOT_PX
    dot_world    = _DOT_WORLD
    HOMY_TOL_M   = 0.001   # 1 mm tolerance

    for lbl, px, world_xy in zip(dot_labels, dot_px_list, dot_world):
        got = pe.apply_homography(tuple(px))
        err = np.linalg.norm(got - world_xy)
        check(
            f"Dot {lbl}: ({world_xy[0]:+.3f},{world_xy[1]:+.3f}) m  "
            f"→ err={err*1e3:.3f} mm",
            err < HOMY_TOL_M,
        )

    # Monotonicity: pixel midpoint of AB (bottom edge) should land between
    # A-world and B-world, i.e. X ≈ 0 and Y < 0.120 (near side).
    mid_ab_px  = ((_DUMMY_DOT_PX[0] + _DUMMY_DOT_PX[1]) / 2).tolist()
    got_mid_ab = pe.apply_homography(tuple(mid_ab_px))
    check(
        f"Midpoint of AB pixels → X ≈ 0 (|x|<{HOMY_TOL_M*1e3:.0f}mm), "
        f"Y on near edge: got ({got_mid_ab[0]:+.4f},{got_mid_ab[1]:+.4f}) m",
        abs(got_mid_ab[0]) < HOMY_TOL_M and
        dot_world[0, 1] - HOMY_TOL_M <= got_mid_ab[1] <= dot_world[2, 1] + HOMY_TOL_M,
    )

    # Symmetry: dot A and dot B should have X values that are equal-and-opposite.
    got_A = pe.apply_homography(tuple(_DUMMY_DOT_PX[0]))
    got_B = pe.apply_homography(tuple(_DUMMY_DOT_PX[1]))
    check(
        "Dot A and B X-coords are symmetric around 0",
        abs(got_A[0] + got_B[0]) < HOMY_TOL_M,
    )
    print()

    # ── wrist_tof_z ───────────────────────────────────────────────────
    print("=== wrist_tof_z ===")

    # 8×8 grid, all 200 mm → z = 0.200 m
    grid_8 = np.full((8, 8), 200.0)
    z = pe.wrist_tof_z(grid_8)
    check("Uniform 200 mm grid → 0.200 m", abs(z - 0.200) < 1e-6, f"z={z:.4f}")

    # 4×4 grid with centre cells = [50, 60, 70, 80] → median = 65 mm = 0.065 m
    grid_4 = np.zeros((4, 4))
    grid_4[1, 1] = 50.0
    grid_4[1, 2] = 60.0
    grid_4[2, 1] = 70.0
    grid_4[2, 2] = 80.0
    z4 = pe.wrist_tof_z(grid_4)
    check("4×4 grid centre [50,60,70,80] → 0.065 m", abs(z4 - 0.065) < 1e-6, f"z={z4:.4f}")

    # Invalid cells (0) are excluded
    grid_invalid = np.full((8, 8), 150.0)
    grid_invalid[3, 3] = 0.0
    grid_invalid[3, 4] = 0.0
    z_inv = pe.wrist_tof_z(grid_invalid)
    check("Grid with 2 invalid centre cells → still returns valid z",
          abs(z_inv - 0.150) < 1e-6, f"z={z_inv:.4f}")

    # All-invalid centre raises ValueError
    grid_bad = np.zeros((8, 8))
    try:
        pe.wrist_tof_z(grid_bad)
        check("All-invalid centre should raise ValueError", False)
    except ValueError:
        check("All-invalid centre raises ValueError", True)
    print()

    # ── compute_pick_pose ─────────────────────────────────────────────
    print("=== compute_pick_pose (homography XY + TOF Z) ===")

    tof_grid = np.full((8, 8), 45.0)   # 45 mm → z = 0.045 m

    # Use dot A pixel → world A = (-0.18, 0.10)
    pose = pe.compute_pick_pose(tuple(_DUMMY_DOT_PX[0]), tof_grid)
    check("Returns 3-element array",  pose.shape == (3,))
    check("X ≈ -0.180 m",  abs(pose[0] - (-0.180)) < HOMY_TOL_M,
          f"x={pose[0]:.4f}")
    check("Y ≈  0.100 m",  abs(pose[1] -  0.100)   < HOMY_TOL_M,
          f"y={pose[1]:.4f}")
    check("Z ≈  0.045 m",  abs(pose[2] -  0.045)   < 1e-6,
          f"z={pose[2]:.4f}")

    # Use dot C pixel → world C = (+0.18, 0.22)
    pose2 = pe.compute_pick_pose(tuple(_DUMMY_DOT_PX[2]), tof_grid)
    check("Dot C: X ≈ +0.180 m", abs(pose2[0] -  0.180) < HOMY_TOL_M,
          f"x={pose2[0]:.4f}")
    check("Dot C: Y ≈  0.220 m", abs(pose2[1] -  0.220) < HOMY_TOL_M,
          f"y={pose2[1]:.4f}")

    # ── overhead_xy sanity ────────────────────────────────────────────
    print("\n=== overhead_xy (pinhole + known Z) ===")

    # Project a world point through the dummy camera model, then back-project.
    # World point: (0, 0.17, Z_TABLE)
    target_world = np.array([0.0, 0.17, Z_TABLE])
    # Forward project with dummy R, t, K
    R_cw  = pe.R_cw
    t_cw  = pe.t_cw
    # Point in camera frame: R_cw @ (P - t_cw)  [t_cw is cam origin in world]
    P_cam = R_cw @ (target_world - t_cw)
    # Project to pixel
    u_ideal = pe.K[0, 0] * (P_cam[0] / P_cam[2]) + pe.K[0, 2]
    v_ideal = pe.K[1, 1] * (P_cam[1] / P_cam[2]) + pe.K[1, 2]

    xy_back = pe.overhead_xy((u_ideal, v_ideal), z_table=Z_TABLE)
    err_oh  = np.linalg.norm(xy_back - target_world[:2])
    check(
        f"overhead_xy round-trip: ({xy_back[0]:+.4f},{xy_back[1]:+.4f}) m  "
        f"err={err_oh*1e3:.3f} mm",
        err_oh < 0.001,
    )

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
