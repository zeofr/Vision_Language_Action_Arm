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
