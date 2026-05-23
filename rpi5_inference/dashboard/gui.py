"""
VLA Robotic Arm — Live Dashboard

5-panel PyQt6 dashboard running at 10 Hz on synthetic data.
Launch: python -m rpi5_inference.dashboard.gui
"""

from __future__ import annotations

import sys
import time
import threading
import random
import math
from collections import deque
from typing import Deque

import numpy as np
import serial

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QFrame,
    QLineEdit, QTextEdit,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

# suppress dataclass/field imports — SharedState is a plain class


# ── Skill enum (mirrors skill_predictor.py) ───────────────────────────────────

SKILLS = ["REACH", "GRASP", "LIFT", "PLACE"]
SKILL_COLORS = {
    "REACH": QColor(70, 130, 180),    # steel blue
    "GRASP": QColor(255, 165, 0),     # orange
    "LIFT":  QColor(60, 179, 113),    # medium sea green
    "PLACE": QColor(205, 92, 92),     # indian red
}
SKILL_COLORS_MPL = {
    "REACH": "#4682B4",
    "GRASP": "#FFA500",
    "LIFT":  "#3CB371",
    "PLACE": "#CD5C5C",
}


# ── Servo commander ───────────────────────────────────────────────────────────

SERVO_PORT   = '/dev/ttyACM0'
SERVO_BAUD   = 1_000_000
SERVO_IDS    = [0x01, 0x02, 0x03, 0x04, 0x05]   # J0 J1A J1B J2 J3
GOAL_POS_REG = 0x2A                               # Goal Position (STS/SMS series)

# ── Fill in all poses below (servo steps, 0–4095 = 0°–360°) ──────────────────
# Use read_servos.py to read live positions while you pose the arm manually.

GRIPPER_OPEN   = 1540
GRIPPER_CLOSED = 1937

HOME_POSE = [3064, 2055, 1928, 1279, GRIPPER_OPEN]

# Per-color pick sequences: [J0, J1A, J1B, J2, J3] in servo steps
# "approach" = position above/at the cube, gripper open
# "pick"     = same as approach (adjust if you want a separate lower step)
# "lift"     = where the arm goes after grasping
COLOR_POSES = {
    "red": {
        "approach": [3063, 2607, 1393, 1806, GRIPPER_OPEN],
        "pick":     [3063, 2607, 1393, 1806, GRIPPER_OPEN],
        "lift":     [3064, 1969, 2014, 1805, 1862],
        "place":    [3994, 2517, 1465, 1519, GRIPPER_CLOSED],
    },
    "blue": {
        "approach": [3063, 2607, 1393, 1806, GRIPPER_OPEN],  # ← fill in
        "pick":     [3063, 2607, 1393, 1806, GRIPPER_OPEN],  # ← fill in
        "lift":     [3064, 1969, 2014, 1805, 1862],          # ← fill in
    },
    "green": {
        "approach": [3063, 2607, 1393, 1806, GRIPPER_OPEN],  # ← fill in
        "pick":     [3063, 2607, 1393, 1806, GRIPPER_OPEN],  # ← fill in
        "lift":     [3064, 1969, 2014, 1805, 1862],          # ← fill in
    },
}


class ServoCommander:
    def __init__(self):
        self._ser = None
        self._lock = threading.Lock()
        self._last_pose: list | None = None
        try:
            self._ser = serial.Serial(SERVO_PORT, SERVO_BAUD, timeout=0.1)
            time.sleep(0.2)
            self._ser.reset_input_buffer()
        except Exception:
            pass

    def connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def send_pose(self, positions: list) -> bool:
        if not self.connected():
            return False
        with self._lock:
            for sid, pos in zip(SERVO_IDS, positions):
                pos_l = pos & 0xFF
                pos_h = (pos >> 8) & 0xFF
                length = 5
                instr  = 0x03
                chk = (~(sid + length + instr + GOAL_POS_REG + pos_l + pos_h)) & 0xFF
                pkt = bytes([0xFF, 0xFF, sid, length, instr, GOAL_POS_REG, pos_l, pos_h, chk])
                self._ser.write(pkt)
                time.sleep(0.005)
        self._last_pose = list(positions)
        return True

    def move_smooth(self, target: list, duration_s: float = 1.5, steps: int = 30) -> bool:
        """Linearly interpolate from last known pose to target over duration_s seconds."""
        if not self.connected():
            return False
        start = list(self._last_pose) if self._last_pose else list(target)
        # Each send_pose call takes ~25 ms (5 servos × 5 ms); subtract that from sleep
        step_delay = max(0.0, duration_s / steps - 0.025)
        for i in range(1, steps + 1):
            t = i / steps
            pose = [int(round(s + (e - s) * t)) for s, e in zip(start, target)]
            if not self.send_pose(pose):
                return False
            time.sleep(step_delay)
        return True

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()


# ── Shared state ──────────────────────────────────────────────────────────────

class SharedState:
    """Thread-safe container for telemetry pushed from inference loop (or synth)."""

    def __init__(self):
        self._lock = threading.Lock()
        # Camera
        self.frame: np.ndarray = np.ones((480, 640, 3), dtype=np.uint8) * 245
        self.bboxes: list = []  # [(x1,y1,x2,y2,label,conf), ...]
        # Skill
        self.skill_history: Deque[str] = deque(maxlen=100)
        self.current_skill: str = "REACH"
        # ToF
        self.tof_grid: np.ndarray = np.full((8, 8), 300.0, dtype=np.float32)
        # Contact signals (200-sample rolling)
        self.imu_rms: Deque[float] = deque([0.5] * 200, maxlen=200)
        self.gripper_load: Deque[float] = deque([0.1] * 200, maxlen=200)
        # Status
        self.inference_latency_ms: float = 95.0
        self.safety_clamped: bool = False
        self.wrist_z_mm: float = 300.0
        self.loop_hz: float = 10.0
        self.estop: bool = False

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frame": self.frame.copy(),
                "bboxes": list(self.bboxes),
                "skill_history": list(self.skill_history),
                "current_skill": self.current_skill,
                "tof_grid": self.tof_grid.copy(),
                "imu_rms": list(self.imu_rms),
                "gripper_load": list(self.gripper_load),
                "inference_latency_ms": self.inference_latency_ms,
                "safety_clamped": self.safety_clamped,
                "wrist_z_mm": self.wrist_z_mm,
                "loop_hz": self.loop_hz,
                "estop": self.estop,
            }


# ── Synthetic data generator ───────────────────────────────────────────────────

class SyntheticDataGenerator:
    """Produces realistic fake telemetry at 10 Hz without any hardware."""

    CYCLE_S = 3.0  # seconds per skill phase
    SKILLS = ["REACH", "GRASP", "LIFT", "PLACE"]

    # Simulated cube positions in image space
    _CUBES = [
        {"label": "red_cube",   "color": (200, 60, 60),   "x": 200, "y": 180, "w": 80, "h": 80},
        {"label": "blue_cube",  "color": (60, 100, 200),  "x": 380, "y": 260, "w": 70, "h": 70},
        {"label": "green_cube", "color": (60, 180, 80),   "x": 480, "y": 150, "w": 75, "h": 75},
    ]

    def __init__(self, state: SharedState):
        self._state = state
        self._t0 = time.monotonic()
        self._tick = 0
        self._last_hz_time = time.monotonic()
        self._hz_ticks = 0

    def tick(self):
        t = time.monotonic() - self._t0
        self._tick += 1

        # ── Skill cycling ─────────────────────────────────────────────
        phase_idx = int(t / self.CYCLE_S) % len(self.SKILLS)
        skill = self.SKILLS[phase_idx]
        phase_frac = (t % self.CYCLE_S) / self.CYCLE_S  # 0.0→1.0 within phase

        # ── IMU RMS ──────────────────────────────────────────────────
        if skill == "GRASP":
            # Spike to 4.5 near middle of GRASP phase
            base = 0.5 + 4.0 * math.exp(-((phase_frac - 0.5) ** 2) / 0.05)
        else:
            base = 0.5
        imu_val = base + random.gauss(0, 0.08)

        # ── Gripper load ─────────────────────────────────────────────
        if skill == "REACH":
            load_base = 0.1
        elif skill in ("GRASP", "LIFT"):
            load_base = 0.1 + 0.6 * min(1.0, phase_frac * 2)
        else:
            load_base = max(0.1, 0.7 - phase_frac * 0.6)
        gripper_val = load_base + random.gauss(0, 0.02)

        # ── ToF grid ─────────────────────────────────────────────────
        tof = np.random.uniform(80, 580, (8, 8)).astype(np.float32)
        if skill == "GRASP":
            # Center zone drops to ~30 mm during grasp
            for r in range(3, 6):
                for c in range(3, 6):
                    tof[r, c] = 30.0 + random.gauss(0, 5)
        wrist_z = float(np.mean(tof[3:5, 3:5]))

        # ── Camera frame ─────────────────────────────────────────────
        frame = self._make_frame(t, skill)

        # ── Bounding boxes ───────────────────────────────────────────
        bboxes = self._make_bboxes(t, skill)

        # ── Status values ─────────────────────────────────────────────
        latency = 80 + 40 * abs(math.sin(t * 0.7)) + random.gauss(0, 3)
        safety_clamped = (skill == "GRASP" and phase_frac > 0.8)

        # ── Loop Hz counter ──────────────────────────────────────────
        self._hz_ticks += 1
        now = time.monotonic()
        if now - self._last_hz_time >= 1.0:
            hz = self._hz_ticks / (now - self._last_hz_time)
            self._last_hz_time = now
            self._hz_ticks = 0
        else:
            hz = 10.0

        # ── Push to shared state ─────────────────────────────────────
        with self._state._lock:
            self._state.frame = frame
            self._state.bboxes = bboxes
            self._state.skill_history.append(skill)
            self._state.current_skill = skill
            self._state.tof_grid = tof
            self._state.imu_rms.append(float(imu_val))
            self._state.gripper_load.append(float(gripper_val))
            self._state.inference_latency_ms = float(latency)
            self._state.safety_clamped = safety_clamped
            self._state.wrist_z_mm = float(wrist_z)
            self._state.loop_hz = hz

    def _make_frame(self, t: float, skill: str) -> np.ndarray:
        frame = np.full((480, 640, 3), 245, dtype=np.uint8)

        # Draw table surface hint
        frame[320:, :] = np.array([210, 200, 190], dtype=np.uint8)

        # Animate cubes slightly
        for i, cube in enumerate(self._CUBES):
            dx = int(8 * math.sin(t * 0.3 + i * 1.2))
            dy = int(5 * math.cos(t * 0.4 + i * 0.9))
            x1 = cube["x"] + dx
            y1 = cube["y"] + dy
            x2 = x1 + cube["w"]
            y2 = y1 + cube["h"]
            r, g, b = cube["color"]
            frame[y1:y2, x1:x2] = [r, g, b]
            # Highlight top face
            frame[y1:y1+8, x1:x2] = [min(255, r+50), min(255, g+50), min(255, b+50)]
            frame[y1:y2, x1:x1+8] = [min(255, r+30), min(255, g+30), min(255, b+30)]

        return frame

    def _make_bboxes(self, t: float, skill: str) -> list:
        bboxes = []
        for i, cube in enumerate(self._CUBES):
            dx = int(8 * math.sin(t * 0.3 + i * 1.2))
            dy = int(5 * math.cos(t * 0.4 + i * 0.9))
            x1 = cube["x"] + dx - 4
            y1 = cube["y"] + dy - 4
            x2 = x1 + cube["w"] + 8
            y2 = y1 + cube["h"] + 8
            conf = 0.85 + random.gauss(0, 0.04)
            conf = max(0.5, min(0.99, conf))
            bboxes.append((x1, y1, x2, y2, cube["label"], conf))
        return bboxes


# ── Helper: render matplotlib figure → QPixmap ───────────────────────────────

def _fig_to_pixmap(fig) -> QPixmap:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    buf.seek(0)
    img = QImage.fromData(buf.read())
    return QPixmap.fromImage(img)


# ── Panel 1: Camera feed ──────────────────────────────────────────────────────

class CameraPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("background: #1a1a2e; border: 1px solid #444;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Overhead Camera Feed")
        title.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_label.setMinimumSize(320, 240)
        layout.addWidget(self._img_label)

    def update_frame(self, frame: np.ndarray, bboxes: list):
        h, w, c = frame.shape
        # Draw bboxes onto a copy
        canvas = frame.copy()
        for (x1, y1, x2, y2, label, conf) in bboxes:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Clamp to frame
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
            # Box outline (green, 2px)
            canvas[y1:y1+2, x1:x2] = [0, 220, 0]
            canvas[y2-2:y2, x1:x2] = [0, 220, 0]
            canvas[y1:y2, x1:x1+2] = [0, 220, 0]
            canvas[y1:y2, x2-2:x2] = [0, 220, 0]

        qimg = QImage(canvas.data, w, h, w * c, QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        label_w = self._img_label.width()
        label_h = self._img_label.height()
        pix = pix.scaled(label_w, label_h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)

        # Paint labels on top with QPainter
        painter = QPainter(pix)
        painter.setFont(QFont("Courier New", 8))
        for (x1, y1, x2, y2, label, conf) in bboxes:
            sx = pix.width() / w
            sy = pix.height() / h
            px = int(x1 * sx)
            py = int(y1 * sy)
            painter.setPen(QPen(QColor(0, 255, 0)))
            painter.drawText(px + 2, max(10, py - 2), f"{label} {conf:.2f}")
        painter.end()

        self._img_label.setPixmap(pix)


# ── Panel 2: Skill timeline ───────────────────────────────────────────────────

class SkillTimelinePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("background: #1a1a2e; border: 1px solid #444;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QHBoxLayout()
        title = QLabel("Skill Timeline")
        title.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        header.addWidget(title)
        self._skill_label = QLabel("REACH")
        self._skill_label.setStyleSheet(
            "color: #4682B4; font-size: 22px; font-weight: bold; padding: 2px 8px;"
        )
        self._skill_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._skill_label)
        layout.addLayout(header)

        self._chart_label = QLabel()
        self._chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chart_label.setMinimumSize(320, 180)
        layout.addWidget(self._chart_label)

        self._fig, self._ax = plt.subplots(figsize=(4, 2.2))
        self._fig.patch.set_facecolor("#1a1a2e")
        self._ax.set_facecolor("#12122a")

    def update_data(self, history: list, current: str):
        color = SKILL_COLORS.get(current, QColor(170, 170, 170))
        hex_color = "#{:02x}{:02x}{:02x}".format(color.red(), color.green(), color.blue())
        self._skill_label.setText(current)
        self._skill_label.setStyleSheet(
            f"color: {hex_color}; font-size: 22px; font-weight: bold; padding: 2px 8px;"
        )

        ax = self._ax
        ax.cla()
        ax.set_facecolor("#12122a")

        if history:
            xs = list(range(len(history)))
            colors = [SKILL_COLORS_MPL.get(s, "#aaa") for s in history]
            ax.bar(xs, [1] * len(history), color=colors, width=1.0, align="edge")

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlabel("last 100 ticks", color="#888", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        legend_patches = [
            mpatches.Patch(color=SKILL_COLORS_MPL[s], label=s) for s in SKILLS
        ]
        ax.legend(handles=legend_patches, loc="upper left", fontsize=6,
                  facecolor="#1a1a2e", edgecolor="#444", labelcolor="#ccc",
                  ncol=4, handlelength=0.8)

        self._fig.tight_layout(pad=0.3)
        self._chart_label.setPixmap(_fig_to_pixmap(self._fig))


# ── Panel 3: ToF heatmap ──────────────────────────────────────────────────────

class ToFHeatmapPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("background: #1a1a2e; border: 1px solid #444;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Wrist ToF Heatmap (8×8)")
        title.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        self._chart_label = QLabel()
        self._chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chart_label.setMinimumSize(260, 220)
        layout.addWidget(self._chart_label)

        self._fig, self._ax = plt.subplots(figsize=(3.2, 2.8))
        self._fig.patch.set_facecolor("#1a1a2e")
        self._img_obj = None

    def update_data(self, tof_grid: np.ndarray):
        ax = self._ax
        ax.cla()
        ax.set_facecolor("#12122a")

        im = ax.imshow(tof_grid, vmin=0, vmax=600, cmap="RdYlBu",
                       interpolation="nearest", aspect="equal")

        ax.set_xticks(range(8))
        ax.set_yticks(range(8))
        ax.tick_params(colors="#888", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        # Annotate cells with value
        for r in range(8):
            for c in range(8):
                val = tof_grid[r, c]
                text_color = "white" if val < 300 else "black"
                ax.text(c, r, f"{int(val)}", ha="center", va="center",
                        color=text_color, fontsize=5)

        if not hasattr(self, "_cbar") or self._cbar is None:
            self._cbar = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._cbar.ax.tick_params(colors="#888", labelsize=6)
            self._cbar.set_label("mm", color="#888", fontsize=7)
        else:
            self._cbar.update_normal(im)

        self._fig.tight_layout(pad=0.3)
        self._chart_label.setPixmap(_fig_to_pixmap(self._fig))

    _cbar = None


# ── Panel 4: Contact oracle signals ──────────────────────────────────────────

class ContactOraclePanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("background: #1a1a2e; border: 1px solid #444;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Contact Oracle Signals")
        title.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        self._chart_label = QLabel()
        self._chart_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chart_label.setMinimumSize(320, 220)
        layout.addWidget(self._chart_label)

        self._fig, self._ax = plt.subplots(figsize=(4, 2.8))
        self._fig.patch.set_facecolor("#1a1a2e")
        self._ax.set_facecolor("#12122a")

    def update_data(self, imu_rms: list, gripper_load: list):
        ax = self._ax
        ax.cla()
        ax.set_facecolor("#12122a")

        xs = list(range(len(imu_rms)))
        ax.plot(xs, imu_rms, color="#9370DB", linewidth=1.2, label="IMU RMS (deg/s)")
        ax.plot(xs, gripper_load, color="#FFA500", linewidth=1.2, label="Gripper load")
        ax.axhline(y=3.5, color="#FF4444", linewidth=1.0, linestyle="--",
                   label="threshold 3.5")

        ax.set_xlim(0, 200)
        ax.set_ylim(-0.2, 6.0)
        ax.tick_params(colors="#888", labelsize=7)
        ax.set_xlabel("samples", color="#888", fontsize=7)
        ax.set_ylabel("value", color="#888", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.legend(loc="upper left", fontsize=6, facecolor="#1a1a2e",
                  edgecolor="#444", labelcolor="#ccc")

        self._fig.tight_layout(pad=0.3)
        self._chart_label.setPixmap(_fig_to_pixmap(self._fig))


# ── Panel 5: Status bar ───────────────────────────────────────────────────────

class StatusBarPanel(QFrame):
    estop_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("background: #0d0d1a; border: 1px solid #444;")
        self.setFixedHeight(72)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(20)

        def _stat_widget(key: str, val: str) -> tuple[QLabel, QLabel]:
            col = QVBoxLayout()
            k_lbl = QLabel(key)
            k_lbl.setStyleSheet("color: #666; font-size: 9px;")
            v_lbl = QLabel(val)
            v_lbl.setStyleSheet("color: #eee; font-size: 13px; font-weight: bold;")
            col.addWidget(k_lbl)
            col.addWidget(v_lbl)
            layout.addLayout(col)
            return k_lbl, v_lbl

        _, self._latency_lbl = _stat_widget("Inference latency", "95 ms")
        _, self._skill_lbl   = _stat_widget("Skill", "REACH")
        _, self._clamp_lbl   = _stat_widget("Safety", "OK")
        _, self._z_lbl       = _stat_widget("Wrist Z", "300 mm")
        _, self._hz_lbl      = _stat_widget("Loop Hz", "10.0")

        layout.addStretch()

        self._estop_btn = QPushButton("EMERGENCY STOP")
        self._estop_btn.setFixedSize(160, 50)
        self._estop_btn.setStyleSheet("""
            QPushButton {
                background-color: #cc0000;
                color: white;
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #ff4444;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #ff2222;
            }
            QPushButton:pressed {
                background-color: #880000;
            }
        """)
        self._estop_btn.clicked.connect(self._on_estop)
        layout.addWidget(self._estop_btn)

    def _on_estop(self):
        print("EMERGENCY STOP", flush=True)
        self.estop_clicked.emit()

    def update_status(self, snap: dict):
        self._latency_lbl.setText(f"{snap['inference_latency_ms']:.1f} ms")

        skill = snap["current_skill"]
        color = SKILL_COLORS.get(skill, QColor(200, 200, 200))
        hex_c = "#{:02x}{:02x}{:02x}".format(color.red(), color.green(), color.blue())
        self._skill_lbl.setText(skill)
        self._skill_lbl.setStyleSheet(f"color: {hex_c}; font-size: 13px; font-weight: bold;")

        clamped = snap["safety_clamped"]
        self._clamp_lbl.setText("CLAMPED" if clamped else "OK")
        self._clamp_lbl.setStyleSheet(
            f"color: {'#ff6644' if clamped else '#44cc44'}; font-size: 13px; font-weight: bold;"
        )

        self._z_lbl.setText(f"{snap['wrist_z_mm']:.1f} mm")
        self._hz_lbl.setText(f"{snap['loop_hz']:.1f}")


# ── Panel 6: Chat / command console ──────────────────────────────────────────

class ChatPanel(QFrame):
    command_issued = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.Box)
        self.setStyleSheet("background: #0d0d1a; border: 1px solid #444;")
        self.setFixedHeight(160)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        title = QLabel("Command Console")
        title.setStyleSheet("color: #aaa; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "background: #12122a; color: #ccc; font-family: 'Courier New'; "
            "font-size: 11px; border: 1px solid #333;"
        )
        layout.addWidget(self._log)

        row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText("Type a command (e.g. 'pick', 'home') and press Enter...")
        self._input.setStyleSheet(
            "background: #1a1a2e; color: #eee; font-family: 'Courier New'; "
            "font-size: 11px; border: 1px solid #555; padding: 3px 6px;"
        )
        self._input.returnPressed.connect(self._on_send)
        row.addWidget(self._input)

        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(64)
        send_btn.setStyleSheet(
            "QPushButton { background: #1e3a5f; color: #eee; border: 1px solid #4682B4; "
            "font-size: 11px; padding: 3px; }"
            "QPushButton:hover { background: #2a5080; }"
        )
        send_btn.clicked.connect(self._on_send)
        row.addWidget(send_btn)
        layout.addLayout(row)

    def append(self, text: str, color: str = "#ccc"):
        self._log.append(f'<span style="color:{color};">{text}</span>')

    def _on_send(self):
        text = self._input.text().strip()
        if not text:
            return
        self._input.clear()
        self.append(f"&gt; {text}", "#888")
        self.command_issued.emit(text.lower())


# ── Main window ───────────────────────────────────────────────────────────────

class DashboardWindow(QMainWindow):
    _chat_append = pyqtSignal(str, str)   # text, color — safe cross-thread chat update

    def __init__(self, state: SharedState, generator: SyntheticDataGenerator):
        super().__init__()
        self._state = state
        self._generator = generator
        self._commander = ServoCommander()

        self.setWindowTitle("VLA Robotic Arm — Live Dashboard")
        self.setMinimumSize(1100, 720)
        self.setStyleSheet("QMainWindow { background: #0d0d1a; }")

        central = QWidget()
        self.setCentralWidget(central)
        grid = QGridLayout(central)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)

        # Panel 1 — camera (row 0, col 0)
        self._camera = CameraPanel()
        grid.addWidget(self._camera, 0, 0)

        # Panel 2 — skill timeline (row 0, col 1)
        self._skills = SkillTimelinePanel()
        grid.addWidget(self._skills, 0, 1)

        # Panel 3 — ToF heatmap (row 1, col 0)
        self._tof = ToFHeatmapPanel()
        grid.addWidget(self._tof, 1, 0)

        # Panel 4 — contact oracle (row 1, col 1)
        self._contact = ContactOraclePanel()
        grid.addWidget(self._contact, 1, 1)

        # Panel 5 — status bar (row 2, full width)
        self._status = StatusBarPanel()
        self._status.estop_clicked.connect(self._handle_estop)
        grid.addWidget(self._status, 2, 0, 1, 2)

        # Panel 6 — chat / command console (row 3, full width)
        self._chat = ChatPanel()
        self._chat.command_issued.connect(self._handle_command)
        self._chat_append.connect(self._chat.append)
        grid.addWidget(self._chat, 3, 0, 1, 2)
        status_msg = (
            f"Servo board connected on {SERVO_PORT}"
            if self._commander.connected()
            else f"WARNING: servo board not found on {SERVO_PORT} — commands will be ignored"
        )
        self._chat.append(status_msg, "#44cc44" if self._commander.connected() else "#ff6644")
        self._chat.append("Commands: pick &lt;red|blue|green&gt; | home", "#666")

        # Row stretch: panels equal, status bar + chat fixed
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)
        grid.setRowStretch(2, 0)
        grid.setRowStretch(3, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        # 10 Hz update timer
        self._timer = QTimer(self)
        self._timer.setInterval(100)  # 100 ms → 10 Hz
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self):
        self._generator.tick()
        snap = self._state.snapshot()

        self._camera.update_frame(snap["frame"], snap["bboxes"])
        self._skills.update_data(snap["skill_history"], snap["current_skill"])
        self._tof.update_data(snap["tof_grid"])
        self._contact.update_data(snap["imu_rms"], snap["gripper_load"])
        self._status.update_status(snap)

    def _handle_command(self, cmd: str):
        parts = cmd.split()
        if parts[0] == "home":
            self._chat.append("Moving to home...", "#4682B4")
            ok = self._commander.send_pose(HOME_POSE)
            self._chat.append("Home done." if ok else "ERROR: servo board not connected.",
                              "#44cc44" if ok else "#ff6644")
        elif parts[0] == "pick" and len(parts) == 2:
            color = parts[1]
            if color not in COLOR_POSES:
                self._chat.append(
                    f"Unknown color '{color}'. Use: red, blue, green", "#ff6644")
                return
            self._chat.append(f"Starting pick sequence → {color.upper()} cube", "#4682B4")
            threading.Thread(target=self._run_pick, args=(color,), daemon=True).start()
        else:
            self._chat.append(
                f"Unknown command: '{cmd}'. Try: pick red | pick blue | pick green | home",
                "#ff6644")

    def _run_pick(self, color: str):
        def log(text, col="#aaa"):
            self._chat_append.emit(text, col)

        if not self._commander.connected():
            log("ERROR: servo board not connected.", "#ff6644")
            return

        poses = COLOR_POSES[color]

        # Step 1 — home
        log("1/7  Moving to home...")
        self._commander.move_smooth(HOME_POSE, duration_s=2.0)

        # Step 2 — approach
        log("2/7  Approaching cube...")
        self._commander.move_smooth(poses["approach"], duration_s=2.5)

        # Step 3 — lower to pick
        log("3/7  Lowering to pick position...")
        self._commander.move_smooth(poses["pick"], duration_s=1.2)

        # Step 4 — close gripper
        log("4/7  Closing gripper...", "#FFA500")
        grasp = list(poses["pick"])
        grasp[4] = GRIPPER_CLOSED
        self._commander.move_smooth(grasp, duration_s=0.6, steps=15)

        # Step 5 — lift
        log("5/7  Lifting...")
        self._commander.move_smooth(poses["lift"], duration_s=2.0)

        # Step 6 — move to place position
        log("6/7  Moving to place position...")
        self._commander.move_smooth(poses["place"], duration_s=2.5)

        # Step 7 — open gripper to release
        log("7/7  Releasing...", "#FFA500")
        release = list(poses["place"])
        release[4] = GRIPPER_OPEN
        self._commander.move_smooth(release, duration_s=0.6, steps=15)

        log(f"{color.upper()} placed!", "#44cc44")

    def _handle_estop(self):
        with self._state._lock:
            self._state.estop = True
        self._status._estop_btn.setEnabled(False)
        self._status._estop_btn.setText("STOPPED")
        self._status._estop_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: #aaa;
                font-size: 13px;
                font-weight: bold;
                border: 2px solid #666;
                border-radius: 6px;
            }
        """)

    def closeEvent(self, event):
        self._commander.close()
        super().closeEvent(event)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette
    from PyQt6.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window,          QColor(13, 13, 26))
    palette.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Base,            QColor(20, 20, 40))
    palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(30, 30, 50))
    palette.setColor(QPalette.ColorRole.ToolTipBase,     QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.Button,          QColor(40, 40, 60))
    palette.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
    palette.setColor(QPalette.ColorRole.BrightText,      QColor(255, 100, 100))
    palette.setColor(QPalette.ColorRole.Highlight,       QColor(70, 130, 180))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)

    state = SharedState()
    generator = SyntheticDataGenerator(state)
    window = DashboardWindow(state, generator)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
