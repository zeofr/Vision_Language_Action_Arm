#!/usr/bin/env python3
"""
Wrist VL53L5CX ToF sensor offset calibration.

Holds the arm wrist at known distances from a flat surface, reads live ToF
centre-zone averages via Teensy telemetry, and computes the systematic offset.

Output
------
  rpi5_inference/calibration/wrist_tof_offset.yaml  (key: wrist_to_sensor_offset_mm)

Requirements
------------
  Teensy must be connected and transmitting 50 Hz telemetry.
"""

import sys
import time
from pathlib import Path

import numpy as np
import yaml

OUTPUT_PATH = Path(__file__).parent / "wrist_tof_offset.yaml"
KNOWN_DISTANCES_MM = [100, 200, 300]   # hold wrist at these heights above a flat surface
N_SAMPLES = 20                          # samples to average at each distance


def _read_tof_centre(ts) -> float | None:
    telem = ts.latest_telemetry
    if telem is None:
        return None
    tof_raw = np.asarray(telem["tof_grid"]).flatten()
    centre  = [
        float(tof_raw[3 * 8 + 3]), float(tof_raw[3 * 8 + 4]),
        float(tof_raw[4 * 8 + 3]), float(tof_raw[4 * 8 + 4]),
    ]
    valid = [z for z in centre if 20 < z < 1000]
    return float(np.mean(valid)) if valid else None


def main() -> None:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    port = input("Teensy serial port [/dev/ttyACM0]: ").strip() or "/dev/ttyACM0"

    from rpi5_inference.comms.teensy_serial import TeensySerial

    print(f"Connecting to {port}…")
    ts = TeensySerial(port)
    time.sleep(2.0)   # allow rx thread to buffer at least one packet

    offsets: list[float] = []

    for known_mm in KNOWN_DISTANCES_MM:
        input(f"\nHold wrist {known_mm} mm above a flat surface. Press Enter when stable…")

        readings: list[float] = []
        for _ in range(N_SAMPLES):
            r = _read_tof_centre(ts)
            if r is not None:
                readings.append(r)
            time.sleep(0.02)

        if not readings:
            print(f"  No valid ToF readings at {known_mm} mm — skipping.")
            continue

        mean_mm = float(np.mean(readings))
        offset  = mean_mm - known_mm
        offsets.append(offset)
        print(f"  ToF reads {mean_mm:.1f} mm at {known_mm} mm true → offset = {offset:+.1f} mm")

    ts.close()

    if not offsets:
        print("No usable measurements collected. Aborted.")
        sys.exit(1)

    mean_offset = float(np.mean(offsets))
    print(f"\nMean offset: {mean_offset:+.2f} mm")

    confirm = input("Save? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    data = {
        "wrist_to_sensor_offset_mm": mean_offset,
        "n_measurements":            len(offsets),
        "raw_offsets_mm":            offsets,
        "known_distances_mm":        KNOWN_DISTANCES_MM,
    }
    OUTPUT_PATH.write_text(yaml.dump(data, default_flow_style=False))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
