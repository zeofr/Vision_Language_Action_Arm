"""
Evaluation harness for the VLA robotic arm pipeline.

Public API
----------
setup_pipeline()          -- create all inference components, return dict
EvaluationRecorder        -- record trial outcomes, save CSV, print report
TASKS                     -- task configs (name → trial params)
run_eval(name, n, pipe)   -- interactive trial loop
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any


# ── task configs ──────────────────────────────────────────────────────────────

TASKS: dict[str, dict[str, Any]] = {
    "task1_pick_place": {
        "instruction": "pick the {color} block and place it in the tray",
        "colors": ["red", "blue", "yellow"],
        "trials_per_color": 20,
        "total_trials": 60,
        "success_criterion": "placement_within_3cm",
    },
    "task2_stacking": {
        "instruction": "stack the {color1} block on top of the {color2} block",
        "pairs": [("red", "blue"), ("blue", "yellow"), ("yellow", "red")],
        "trials_per_pair": 10,
        "total_trials": 30,
        "success_criterion": "stable_3s",
    },
    "task3_sorting": {
        "instruction": "pick the {color} block",
        "colors": ["red", "blue"],
        "trials_per_color": 20,
        "total_trials": 40,
        "success_criterion": "correct_object_placed",
    },
}


# ── pipeline factory ──────────────────────────────────────────────────────────

def setup_pipeline(
    port: str = "/dev/ttyACM0",
    yolo_checkpoint: str = "checkpoints/yolov8n_vla/weights/best.pt",
    vla_checkpoint:  str = "checkpoints/vla_policy_traced.pt",
) -> dict:
    """Instantiate all inference components and return them as a dict.

    All imports are lazy so that importing this module does not trigger
    hardware initialisation (required for the dry-run import test).

    Keys in the returned dict:
        camera, teensy, det, pose, lang, vla, action_gen, sm, sf
    """
    from rpi5_inference.comms.teensy_serial        import TeensySerial
    from rpi5_inference.perception.camera_manager  import CameraManager
    from rpi5_inference.perception.yolo_detector   import YOLODetector
    from rpi5_inference.perception.pose_estimation import PoseEstimator
    from rpi5_inference.language.language_encoder  import LanguageEncoder
    from rpi5_inference.vla.vla_policy             import VLARuntime
    from rpi5_inference.vla.action_generator       import ActionGenerator
    from rpi5_inference.vla.skill_predictor        import SkillStateMachine
    from rpi5_inference.planning.safety_filter     import SafetyFilter

    lang       = LanguageEncoder()
    camera     = CameraManager()
    det        = YOLODetector(yolo_checkpoint)
    pe         = PoseEstimator()
    vla        = VLARuntime(vla_checkpoint, lang)
    action_gen = ActionGenerator()
    sm         = SkillStateMachine()
    sf         = SafetyFilter()
    ts         = TeensySerial(port)

    return {
        "camera":     camera,
        "teensy":     ts,
        "det":        det,
        "pose":       pe,
        "lang":       lang,
        "vla":        vla,
        "action_gen": action_gen,
        "sm":         sm,
        "sf":         sf,
    }


# ── evaluation recorder ───────────────────────────────────────────────────────

class EvaluationRecorder:
    """Accumulate trial results in memory, save to CSV, print summary report."""

    _FIELDS = (
        "task", "trial", "instruction", "success",
        "skill_f1", "contact_latency_ms", "inference_ms", "timestamp",
    )

    def __init__(self, output_path: str) -> None:
        self.path = output_path
        self.rows: list[dict] = []

    def record(
        self,
        task: str,
        trial: int,
        instruction: str,
        success: bool,
        skill_f1: float,
        contact_latency_ms: float,
        inference_ms: float,
    ) -> None:
        self.rows.append({
            "task":               task,
            "trial":              trial,
            "instruction":        instruction,
            "success":            int(success),
            "skill_f1":           skill_f1,
            "contact_latency_ms": contact_latency_ms,
            "inference_ms":       inference_ms,
            "timestamp":          time.strftime("%Y%m%d_%H%M%S"),
        })

    def save(self) -> None:
        if not self.rows:
            raise ValueError("No rows to save — record at least one trial first.")
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Results saved to {self.path}")

    def report(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            print("pandas not available — raw row dump:")
            for row in self.rows:
                print(row)
            return

        df = pd.DataFrame(self.rows)
        for task in df["task"].unique():
            sub = df[df["task"] == task]
            print(
                f"\n{task}: success={sub['success'].mean():.1%} "
                f"({int(sub['success'].sum())}/{len(sub)}) "
                f"| mean_latency={sub['inference_ms'].mean():.1f}ms "
                f"| mean_skill_f1={sub['skill_f1'].mean():.3f}"
            )


# ── trial loop ────────────────────────────────────────────────────────────────

def _build_instruction(task_cfg: dict, trial: int) -> str:
    if "pairs" in task_cfg:
        pair = task_cfg["pairs"][trial % len(task_cfg["pairs"])]
        return task_cfg["instruction"].format(color1=pair[0], color2=pair[1])
    if "colors" in task_cfg:
        color = task_cfg["colors"][trial % len(task_cfg["colors"])]
        return task_cfg["instruction"].format(color=color)
    return task_cfg["instruction"]


def run_eval(
    task_name: str,
    n_trials: int,
    pipeline: dict,
    output_path: str = "results/eval_results.csv",
) -> EvaluationRecorder:
    """Interactive evaluation loop.

    The inference pipeline (pipeline dict from setup_pipeline()) should already
    be running in a background thread or called externally. This function
    collects trial outcomes via manual user input.
    """
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Valid: {list(TASKS)}")

    task_cfg = TASKS[task_name]
    recorder = EvaluationRecorder(output_path)

    print(f"\n{'='*60}")
    print(f"Task: {task_name}  ({n_trials} trials)")
    print(f"Criterion: {task_cfg['success_criterion']}")
    print(f"{'='*60}")

    for trial in range(n_trials):
        instruction = _build_instruction(task_cfg, trial)
        print(f"\nTrial {trial + 1}/{n_trials}: '{instruction}'")
        input("  Set up the scene, then press Enter to start…")

        t0 = time.monotonic()
        input("  Press Enter when the trial is complete…")
        inference_ms = (time.monotonic() - t0) * 1000.0

        s = input("  Success? [y/N]: ").strip().lower()
        success = s == "y"

        recorder.record(
            task=task_name,
            trial=trial,
            instruction=instruction,
            success=success,
            skill_f1=0.0,           # filled post-hoc via skill_f1.py
            contact_latency_ms=0.0,  # filled post-hoc via contact_latency.py
            inference_ms=inference_ms,
        )
        print(f"  Recorded: {'✓ success' if success else '✗ failure'}")

    recorder.save()
    recorder.report()
    return recorder
