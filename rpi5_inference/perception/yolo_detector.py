"""
YOLOv8-nano detector wrapper for the VLA robotic arm.

CLASS_NAMES = ['red_cube', 'blue_cube', 'green_cube']   (class ids 0, 1, 2)

Detection dataclass fields:
  class_id    int
  class_name  str
  confidence  float
  bbox_xyxy   np.ndarray shape (4,)  [x1, y1, x2, y2] pixels
  centroid    property → (cx, cy) pixel tuple  (computed from bbox)

match_instruction(detections, instruction) → Detection | None
  Returns the highest-confidence detection whose class_name appears
  anywhere in the lowercased instruction string, or None if no match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

CLASS_NAMES: list[str] = ["red_cube", "blue_cube", "green_cube"]

CONF_THRESHOLD: float = 0.5

# Fine-tuned checkpoint (present on RPi after training).
_DEFAULT_CKPT = Path(__file__).parents[2] / "checkpoints" / "yolov8n_vla" / "weights" / "best.pt"


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox_xyxy:  np.ndarray  # [x1, y1, x2, y2] float32

    @property
    def centroid(self) -> tuple[float, float]:
        """(cx, cy) in pixels."""
        x1, y1, x2, y2 = self.bbox_xyxy
        return float((x1 + x2) / 2), float((y1 + y2) / 2)

    def __repr__(self) -> str:
        cx, cy = self.centroid
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"cx={cx:.1f}, cy={cy:.1f})")


class YOLODetector:
    """
    Thin wrapper around an Ultralytics YOLO model.

    If the fine-tuned checkpoint does not exist, falls back to the
    base yolov8n weights so the code path remains testable on the
    development machine before the RPi checkpoint is copied over.
    """

    def __init__(
        self,
        checkpoint: Path | str = _DEFAULT_CKPT,
        conf_threshold: float = CONF_THRESHOLD,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        ckpt = Path(checkpoint)
        if not ckpt.exists():
            import warnings
            warnings.warn(
                f"[YOLODetector] Checkpoint not found: {ckpt}. "
                "Falling back to base yolov8n weights.",
                UserWarning, stacklevel=2,
            )
            ckpt = "yolov8n.pt"   # ultralytics auto-downloads if absent

        self.model = YOLO(str(ckpt))
        self.conf_threshold = conf_threshold
        self.device = device

    # ── public API ────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run inference on a BGR or RGB uint8 HxWx3 numpy array.
        Returns a list of Detection objects with confidence ≥ CONF_THRESHOLD,
        sorted by confidence descending.

        The model's own class names are remapped to CLASS_NAMES for the
        fine-tuned checkpoint; for the base model the list may differ —
        detections with class_id ≥ len(CLASS_NAMES) are dropped so the
        interface stays consistent.
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for box in boxes:
                cls_id    = int(box.cls[0].item())
                conf      = float(box.conf[0].item())
                xyxy      = box.xyxy[0].cpu().numpy().astype(np.float32)

                if cls_id >= len(CLASS_NAMES):
                    continue   # class outside our label set

                detections.append(Detection(
                    class_id=cls_id,
                    class_name=CLASS_NAMES[cls_id],
                    confidence=conf,
                    bbox_xyxy=xyxy,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── helper ────────────────────────────────────────────────────────

    @staticmethod
    def match_instruction(
        detections: list[Detection],
        instruction: str,
    ) -> Optional[Detection]:
        """
        Return the highest-confidence detection whose class_name is a
        substring of the lowercased instruction, or None.

        Example: "pick up the red cube" matches class_name "red_cube"
        because "red_cube" and "red" both appear — we check the
        colour token (first word of class_name split by '_') to be
        robust to phrasing variation.
        """
        instruction_lower = instruction.lower()
        for det in detections:   # already sorted by confidence desc
            colour = det.class_name.split("_")[0]   # "red", "blue", "green"
            if colour in instruction_lower or det.class_name in instruction_lower:
                return det
        return None


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import warnings

    all_pass = True

    def check(label: str, condition: bool) -> None:
        global all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        print(f"  [{status}] {label}")

    # ── instantiation ─────────────────────────────────────────────────
    print("=== Instantiation ===")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector = YOLODetector()

    check("Detector created without exception", True)
    if caught:
        print(f"  (fallback warning: {str(caught[0].message)[:80]})")

    # ── dummy black frame returns empty list ──────────────────────────
    print("\n=== Dummy black image (480×640) ===")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)

    result = detector.detect(dummy)
    check("detect() returns a list",                  isinstance(result, list))
    check("No detections on all-black image",         len(result) == 0)

    # ── dummy white frame ─────────────────────────────────────────────
    print("\n=== Dummy white image (480×640) ===")
    white = np.full((480, 640, 3), 255, dtype=np.uint8)
    result_w = detector.detect(white)
    check("detect() returns a list on white image",   isinstance(result_w, list))
    print(f"  Detections on white frame: {len(result_w)} "
          f"(0 expected for base model — coloured cubes not in COCO)")

    # ── Detection dataclass API ───────────────────────────────────────
    print("\n=== Detection dataclass API ===")
    dummy_det = Detection(
        class_id=0,
        class_name="red_cube",
        confidence=0.92,
        bbox_xyxy=np.array([100.0, 200.0, 160.0, 260.0], dtype=np.float32),
    )
    cx, cy = dummy_det.centroid
    check("centroid cx == 130.0",  abs(cx - 130.0) < 1e-4)
    check("centroid cy == 230.0",  abs(cy - 230.0) < 1e-4)
    check("repr contains class name", "red_cube" in repr(dummy_det))

    # ── match_instruction ─────────────────────────────────────────────
    print("\n=== match_instruction ===")
    det_red   = Detection(0, "red_cube",   0.91,
                          np.array([10., 10., 50., 50.], dtype=np.float32))
    det_blue  = Detection(1, "blue_cube",  0.85,
                          np.array([60., 60., 100., 100.], dtype=np.float32))
    det_green = Detection(2, "green_cube", 0.78,
                          np.array([110., 110., 150., 150.], dtype=np.float32))

    all_dets = [det_blue, det_red, det_green]   # NOT sorted by conf (on purpose)
    # sort descending by conf to mimic detect() output
    all_dets.sort(key=lambda d: d.confidence, reverse=True)

    m = YOLODetector.match_instruction(all_dets, "pick up the red cube")
    check("'red cube' instruction matches red_cube",    m is not None and m.class_name == "red_cube")

    m = YOLODetector.match_instruction(all_dets, "grab the blue cube")
    check("'blue cube' instruction matches blue_cube",  m is not None and m.class_name == "blue_cube")

    m = YOLODetector.match_instruction(all_dets, "move the green cube")
    check("'green cube' instruction matches green_cube",m is not None and m.class_name == "green_cube")

    m = YOLODetector.match_instruction(all_dets, "place something somewhere")
    check("No colour keyword → returns None",           m is None)

    m = YOLODetector.match_instruction([], "pick up the red cube")
    check("Empty list → returns None",                  m is None)

    # Highest-confidence match is returned when multiple colours present
    multi = "grab the red or blue cube"
    m = YOLODetector.match_instruction(all_dets, multi)
    check("Multiple matches → highest confidence returned (red, 0.91)",
          m is not None and m.class_name == "red_cube")

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
