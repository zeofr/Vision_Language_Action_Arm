#!/usr/bin/env python3
"""
Overhead camera height calibration.

Measures the perpendicular distance from the camera lens to the workspace surface.
Takes three manual tape-measure readings and saves the average.

Output
------
  rpi5_inference/calibration/overhead_height.yaml  (key: Z_table_m)
"""

import yaml
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "overhead_height.yaml"


def main() -> None:
    print("Overhead camera height calibration")
    print("Measure from the camera lens down to the workspace mat surface.")
    print("Take 3 tape-measure readings for averaging.\n")

    readings: list[float] = []
    for i in range(3):
        while True:
            try:
                val = float(input(f"  Measurement {i + 1}/3 (mm): ").strip())
                if val <= 0:
                    print("  Value must be positive.")
                    continue
                readings.append(val)
                break
            except ValueError:
                print("  Enter a numeric value.")

    avg_mm = sum(readings) / len(readings)
    avg_m  = avg_mm / 1000.0
    std_mm = (sum((x - avg_mm) ** 2 for x in readings) / len(readings)) ** 0.5

    print(f"\n  Mean: {avg_mm:.1f} mm = {avg_m:.4f} m")
    print(f"  Std:  {std_mm:.2f} mm")

    confirm = input("\nSave this value? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted — no file written.")
        return

    data = {
        "Z_table_m":              avg_m,
        "Z_table_mm":             avg_mm,
        "n_measurements":         len(readings),
        "raw_measurements_mm":    readings,
        "std_mm":                 std_mm,
    }
    OUTPUT_PATH.write_text(yaml.dump(data, default_flow_style=False))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
