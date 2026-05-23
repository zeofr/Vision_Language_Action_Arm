"""
Skill F1 measurement utilities.

compute_skill_f1  -- macro F1 between auto-segmented and human-annotated labels
annotate_demo_manually -- interactive CLI to label every 10th timestep of a demo
"""

from __future__ import annotations

SKILL_INT = {"REACH": 0, "GRASP": 1, "LIFT": 2, "PLACE": 3}
SKILL_STR = {v: k for k, v in SKILL_INT.items()}


def compute_skill_f1(
    predicted_labels: list,
    ground_truth_labels: list,
) -> float:
    """Return macro-averaged F1 between predicted and ground-truth skill labels.

    Both lists may contain int (0–3) or str ('REACH'/'GRASP'/'LIFT'/'PLACE').
    """
    from sklearn.metrics import f1_score

    def to_int(labels: list) -> list[int]:
        return [SKILL_INT[s] if isinstance(s, str) else int(s) for s in labels]

    pred = to_int(predicted_labels)
    gt   = to_int(ground_truth_labels)
    return float(f1_score(gt, pred, average="macro", zero_division=0))


def annotate_demo_manually(demo_path: str) -> list[str]:
    """Interactive CLI for human annotation of a demo HDF5 file.

    Prints sensor summary for every 10th timestep and prompts the user to
    label it REACH / GRASP / LIFT / PLACE. Returns a full-length list
    (labels for every timestep, populated by forward-filling from annotated
    keyframes).

    Usage::

        labels = annotate_demo_manually("demos/demo_001_pick_red_block.h5")
        # labels is a list of str of length == len(telemetry)
    """
    import sys
    import numpy as np

    sys.path.insert(0, __file__.split("rpi5_inference")[0])
    from dataset.hdf5_reader import load_demo

    _MAP = {"R": "REACH", "G": "GRASP", "L": "LIFT", "P": "PLACE"}

    demo = load_demo(demo_path)
    tel  = demo["telemetry"]
    n    = len(tel)

    keyframe_labels: dict[int, str] = {}
    print(f"\nAnnotating {demo_path} ({n} timesteps, labelling every 10th)")
    print("Input: R=REACH  G=GRASP  L=LIFT  P=PLACE\n")

    for i in range(0, n, 10):
        row       = tel[i]
        j_pos     = [float(row["servo_pos"][k]) for k in range(5)]
        load_grip = float(row["servo_load"][4])
        contact   = int(row["contact_flag"])
        tof_c     = int(row["tof_grid"][3 * 8 + 3])

        print(
            f"t={i / 50:.2f}s | pos={[round(v,1) for v in j_pos]} "
            f"| grip_load={load_grip:.2f} | contact={contact} | tof_c={tof_c}mm"
        )
        while True:
            raw = input("  [R/G/L/P]: ").strip().upper()
            if raw in _MAP:
                keyframe_labels[i] = _MAP[raw]
                break
            print("  Enter R, G, L, or P.")

    # Forward-fill from keyframes to produce a label per timestep
    labels: list[str] = ["REACH"] * n
    last = "REACH"
    for i in range(n):
        if i in keyframe_labels:
            last = keyframe_labels[i]
        labels[i] = last

    return labels
