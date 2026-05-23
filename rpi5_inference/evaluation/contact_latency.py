"""
IMU contact detection latency measurement.

Quantifies how many milliseconds earlier the IMU-based contact oracle
detects a grasp event compared to a load-threshold approach on the same demo.

measure_imu_vs_load_latency  -- single demo, returns ms advantage (positive = IMU faster)
batch_latency                -- all demos in a directory, prints summary stats
"""

from __future__ import annotations

import numpy as np

LOAD_THRESHOLD = 0.35   # normalised gripper load that indicates contact
TELEMETRY_HZ   = 50     # samples per second


def measure_imu_vs_load_latency(h5_path: str) -> float:
    """Return how many ms earlier the IMU contact flag fires vs. load threshold.

    Positive value = IMU fires first (expected).
    NaN = one or both triggers never fired in this demo.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(h5_path).parents[1]))
    from dataset.hdf5_reader import load_demo

    demo = load_demo(h5_path)
    tel  = demo["telemetry"]

    imu_trigger_idx:  int | None = None
    load_trigger_idx: int | None = None

    for i, row in enumerate(tel):
        if imu_trigger_idx is None and int(row["contact_flag"]):
            imu_trigger_idx = i
        if load_trigger_idx is None and float(row["servo_load"][4]) > LOAD_THRESHOLD:
            load_trigger_idx = i

    if imu_trigger_idx is None or load_trigger_idx is None:
        return float("nan")

    delta_samples = load_trigger_idx - imu_trigger_idx
    return delta_samples * (1000.0 / TELEMETRY_HZ)    # convert samples → ms


def batch_latency(demo_dir: str) -> None:
    """Compute and print IMU advantage statistics across all demo HDF5 files."""
    from pathlib import Path

    paths     = sorted(Path(demo_dir).glob("demo_*.h5"))
    latencies = [measure_imu_vs_load_latency(str(p)) for p in paths]
    valid     = [v for v in latencies if not np.isnan(v)]

    print(f"IMU contact detection advantage over load-based ({len(valid)}/{len(paths)} demos):")
    if not valid:
        print("  No demos had both triggers fire — cannot compute latency.")
        return
    print(f"  Mean:   {np.mean(valid):.1f} ms earlier")
    print(f"  Median: {np.median(valid):.1f} ms earlier")
    print(f"  Std:    {np.std(valid):.1f} ms")
    print(f"  Max:    {np.max(valid):.1f} ms earlier")
