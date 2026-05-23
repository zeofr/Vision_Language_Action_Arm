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
