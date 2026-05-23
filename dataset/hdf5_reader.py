"""
dataset/hdf5_reader.py – Load and inspect demonstration HDF5 files.

HDF5 layout (written by the data-collection script on RPi5):
  /telemetry   uint8 dataset, shape (T, 250) — raw bytes of TELEMETRY_DTYPE
  /rgb_frames  uint8 dataset, shape (F, H, W, 3)
  /frame_ts    uint64 dataset, shape (F,)  microseconds
  attrs:  instruction (str), task_type (str)
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent   # dataset/
_ROOT = _HERE.parent                      # vla-robotic-arm/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rpi5_inference.comms.teensy_serial import TELEMETRY_DTYPE  # noqa: E402

# ── servo conversion ──────────────────────────────────────────────────────────
_CENTER_TICK  = 2047          # ticks corresponding to 0°
_TICKS_TO_DEG = 300.0 / 4095  # degrees per tick


# ── public API ────────────────────────────────────────────────────────────────

def load_demo(h5_path: str | Path) -> dict[str, Any]:
    """
    Load a single demonstration HDF5 file.

    Returns
    -------
    dict with keys:
        telemetry   np.ndarray  shape (T,)        dtype=TELEMETRY_DTYPE
        rgb_frames  np.ndarray  shape (F, H, W, 3) dtype=uint8
        frame_ts    np.ndarray  shape (F,)          dtype=uint64  (µs)
        instruction str
        task_type   str
        path        str  absolute path of the source file
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, 'r') as f:
        raw_telem  = f['telemetry'][:]    # (T, 250) uint8
        rgb_frames = f['rgb_frames'][:]
        frame_ts   = f['frame_ts'][:]

        instruction = f.attrs.get('instruction', '')
        task_type   = f.attrs.get('task_type',   '')

    # h5py may return bytes on some versions
    if isinstance(instruction, (bytes, np.bytes_)):
        instruction = instruction.decode()
    if isinstance(task_type, (bytes, np.bytes_)):
        task_type = task_type.decode()

    # Reconstruct structured array from raw bytes
    telemetry = np.frombuffer(raw_telem.tobytes(), dtype=TELEMETRY_DTYPE).copy()

    return {
        'telemetry':   telemetry,
        'rgb_frames':  rgb_frames,
        'frame_ts':    frame_ts,
        'instruction': str(instruction),
        'task_type':   str(task_type),
        'path':        str(h5_path.resolve()),
    }


def load_all_demos(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Load every *.h5 file under *dataset_dir* recursively."""
    return [load_demo(p) for p in sorted(Path(dataset_dir).rglob('*.h5'))]


def get_logical_joints(telemetry_row: np.ndarray) -> np.ndarray:
    """
    Convert one telemetry row to a 4-element joint-angle array [J0, J1, J2, J3]
    in degrees.

    J1 is the average of servo channels 1 and 2 (parallel-link mechanism).
    Conversion: degrees = (tick – 2047) × (300 / 4095).

    Parameters
    ----------
    telemetry_row : structured array row — either a numpy void (telem[i])
                    or a 1-element slice (telem[i:i+1]).
    """
    raw = telemetry_row['servo_pos_raw']  # (5,) or (1,5)
    if raw.ndim > 1:
        raw = raw[0]

    def _deg(tick) -> float:
        return float((int(tick) - _CENTER_TICK) * _TICKS_TO_DEG)

    return np.array([
        _deg(raw[0]),
        (_deg(raw[1]) + _deg(raw[2])) / 2.0,
        _deg(raw[3]),
        _deg(raw[4]),
    ], dtype=np.float32)


def get_nearest_frame(
    frame_ts:   np.ndarray,
    query_us:   int,
    rgb_frames: np.ndarray,
) -> np.ndarray:
    """
    Return the RGB frame whose timestamp is closest to *query_us* (microseconds).

    Parameters
    ----------
    frame_ts   : shape (F,) uint64 timestamps in µs
    query_us   : target timestamp in µs
    rgb_frames : shape (F, H, W, 3)
    """
    idx = int(np.argmin(np.abs(frame_ts.astype(np.int64) - int(query_us))))
    return rgb_frames[idx]


def generate_synthetic_demo(
    n_timesteps: int = 200,
    n_frames:    int = 40,
    img_h:       int = 120,
    img_w:       int = 160,
) -> str:
    """
    Write a synthetic HDF5 demo to a temp file and return its path.

    The demo covers a complete REACH → GRASP → LIFT → PLACE sequence:
      t 0–49   REACH   approaching object, tof decreasing toward 50 mm
      t 50–99  GRASP   contact detected, load spikes, gripper closes
      t 100–149 LIFT   J1 sweeps 0→60°, end-effector rises
      t 150–199 PLACE  arm descends to place position
    """
    rng = np.random.default_rng(42)

    telem = np.zeros(n_timesteps, dtype=TELEMETRY_DTYPE)
    telem['magic']         = 0xABCD
    telem['seq']           = np.arange(n_timesteps, dtype=np.uint32)
    telem['timestamp_ms']  = np.arange(n_timesteps, dtype=np.uint32) * 20  # 50 Hz
    telem['servo_pos_raw'] = _CENTER_TICK   # all joints at 0°
    telem['adc_supply_mv'] = 12_000
    telem['temp_c']        = 2500           # 25.00 °C

    for i in range(n_timesteps):
        phase = i // 50   # 0=REACH  1=GRASP  2=LIFT  3=PLACE

        # ── J1 angle: channels 1 & 2 sweep from 0° to 60° during LIFT ──
        if phase == 2:
            j1_deg = (i - 100) / 50.0 * 60.0        # 0 → 60°
        elif phase == 3:
            j1_deg = 60.0 - (i - 150) / 50.0 * 30.0  # 60 → 30°
        else:
            j1_deg = 0.0

        j1_tick = int(np.clip(_CENTER_TICK + j1_deg / _TICKS_TO_DEG, 0, 4095))
        telem['servo_pos_raw'][i, 1] = j1_tick
        telem['servo_pos_raw'][i, 2] = j1_tick

        # ── servo velocities (deg10/s units match command encoding) ─────
        if phase in (0, 2):
            telem['servo_vel'][i] = int(rng.integers(30, 80))    # moving
        else:
            telem['servo_vel'][i] = int(rng.integers(0, 6))      # nearly stopped

        # ── servo load (raw Dynamixel units, max ~1023) ─────────────────
        if phase == 0:
            telem['servo_load'][i] = int(rng.integers(50, 100))   # no contact
        elif phase in (1, 2):
            telem['servo_load'][i] = int(rng.integers(350, 500))  # gripping
        else:
            telem['servo_load'][i] = int(rng.integers(200, 350))  # partial load

        # ── contact_flag: 1 once gripper makes contact ──────────────────
        telem['contact_flag'][i] = 1 if phase >= 1 else 0

        # ── imu_gyro_rms: spikes at GRASP impact ────────────────────────
        telem['imu_gyro_rms'][i] = float(
            rng.uniform(8.0, 15.0) if phase == 1 else rng.uniform(0.5, 3.0)
        )

        # ── tof_grid: linear approach 200 mm → ~33 mm during REACH ──────
        if phase == 0:
            dist_mm = int(200 - i * (170.0 / 50))   # 200 → 30 mm
        else:
            dist_mm = 200
        telem['tof_grid'][i] = dist_mm

        # ── gripper_pos: open → closed during GRASP ─────────────────────
        if phase == 0:
            telem['gripper_pos'][i] = 30.0
        elif phase == 1:
            t_frac = (i - 50) / 50.0
            telem['gripper_pos'][i] = float(30.0 - t_frac * 28.0)  # 30→2 mm
        else:
            telem['gripper_pos'][i] = 2.0

        telem['imu_accel'][i] = [
            float(rng.normal(0.0, 0.05)),
            float(rng.normal(0.0, 0.05)),
            float(rng.normal(-9.81, 0.10)),
        ]

    # ── pack as raw bytes for h5py storage (avoids compound-type edge cases) ──
    telem_raw = np.frombuffer(telem.tobytes(), dtype=np.uint8).reshape(n_timesteps, 250)

    # ── random RGB frames ──────────────────────────────────────────────────────
    rgb_frames = rng.integers(0, 256, size=(n_frames, img_h, img_w, 3), dtype=np.uint8)

    # ── frame timestamps: evenly spaced over demo duration ────────────────────
    demo_us = n_timesteps * 20_000   # 20 ms per telemetry step
    frame_ts = np.linspace(0, demo_us, n_frames, endpoint=False, dtype=np.uint64)

    # ── write ──────────────────────────────────────────────────────────────────
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp.close()

    with h5py.File(tmp.name, 'w') as f:
        f.create_dataset('telemetry',  data=telem_raw)
        f.create_dataset('rgb_frames', data=rgb_frames,
                         compression='gzip', compression_opts=1)
        f.create_dataset('frame_ts',   data=frame_ts)
        f.attrs['instruction'] = 'pick up the red block and place it on the shelf'
        f.attrs['task_type']   = 'pick_and_place'

    return tmp.name


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    all_pass = True

    def check(label: str, cond: bool, detail: str = '') -> None:
        global all_pass
        status = 'PASS' if cond else 'FAIL'
        if not cond:
            all_pass = False
        suffix = f'  ({detail})' if detail else ''
        print(f'  [{status}] {label}{suffix}')

    # ── generate ──────────────────────────────────────────────────────────────
    print('=== generate_synthetic_demo ===')
    h5_path = generate_synthetic_demo()
    check('file created in temp dir',
          h5_path.startswith(tempfile.gettempdir()))
    check('file exists on disk', Path(h5_path).exists())

    # ── load_demo ─────────────────────────────────────────────────────────────
    print('\n=== load_demo: all keys present ===')
    demo = load_demo(h5_path)

    for key in ('telemetry', 'rgb_frames', 'frame_ts', 'instruction',
                'task_type', 'path'):
        check(f'key "{key}" present', key in demo)

    # ── telemetry ─────────────────────────────────────────────────────────────
    print('\n=== load_demo: telemetry ===')
    telem = demo['telemetry']
    check('telemetry dtype == TELEMETRY_DTYPE',
          telem.dtype == TELEMETRY_DTYPE,
          f'got {telem.dtype}')
    check('telemetry shape == (200,)',
          telem.shape == (200,), f'got {telem.shape}')
    check('TELEMETRY_DTYPE.itemsize == 250',
          TELEMETRY_DTYPE.itemsize == 250, f'got {TELEMETRY_DTYPE.itemsize}')
    check('magic field == 0xABCD for every row',
          bool(np.all(telem['magic'] == 0xABCD)))
    check('seq is sequential',
          bool(np.all(telem['seq'] == np.arange(200, dtype=np.uint32))))
    check('contact_flag 0 for t<50, 1 for t>=50',
          bool(np.all(telem['contact_flag'][:50] == 0)) and
          bool(np.all(telem['contact_flag'][50:] == 1)))

    # ── rgb_frames ────────────────────────────────────────────────────────────
    print('\n=== load_demo: rgb_frames ===')
    rgb = demo['rgb_frames']
    check('rgb_frames shape == (40, 120, 160, 3)',
          rgb.shape == (40, 120, 160, 3), f'got {rgb.shape}')
    check('rgb_frames dtype == uint8', rgb.dtype == np.uint8)

    # ── frame_ts ──────────────────────────────────────────────────────────────
    print('\n=== load_demo: frame_ts ===')
    ft = demo['frame_ts']
    check('frame_ts shape == (40,)', ft.shape == (40,), f'got {ft.shape}')
    check('frame_ts monotonically increasing',
          bool(np.all(np.diff(ft.astype(np.int64)) > 0)))

    # ── metadata ──────────────────────────────────────────────────────────────
    print('\n=== load_demo: metadata ===')
    check('instruction is non-empty string',
          isinstance(demo['instruction'], str) and len(demo['instruction']) > 0)
    check('task_type == "pick_and_place"',
          demo['task_type'] == 'pick_and_place',
          f'got "{demo["task_type"]}"')
    check('path is absolute', Path(demo['path']).is_absolute())

    # ── get_logical_joints ────────────────────────────────────────────────────
    print('\n=== get_logical_joints ===')

    # Using numpy void (scalar row)
    joints_void = get_logical_joints(telem[0])
    check('output shape (4,) from void row',
          joints_void.shape == (4,), f'got {joints_void.shape}')
    check('output dtype float32', joints_void.dtype == np.float32)
    check('J0 ≈ 0° at t=0 (center tick)',
          abs(joints_void[0]) < 0.5, f'got {joints_void[0]:.3f}')
    check('J1 ≈ 0° at t=0 (center tick)',
          abs(joints_void[1]) < 0.5, f'got {joints_void[1]:.3f}')

    # Using 1-element slice
    joints_slice = get_logical_joints(telem[0:1])
    check('output shape (4,) from 1-element slice',
          joints_slice.shape == (4,), f'got {joints_slice.shape}')
    check('void and slice results match',
          np.allclose(joints_void, joints_slice))

    # LIFT midpoint t=125: J1 should be ~30°
    j_lift = get_logical_joints(telem[125])
    check('J1 ≈ 30° at LIFT midpoint (t=125)',
          20.0 < j_lift[1] < 40.0, f'got {j_lift[1]:.2f}°')

    # LIFT end t=149: J1 should be ~60°
    j_lift_end = get_logical_joints(telem[149])
    check('J1 ≈ 60° at LIFT end (t=149)',
          50.0 < j_lift_end[1] < 70.0, f'got {j_lift_end[1]:.2f}°')

    # ── get_nearest_frame ─────────────────────────────────────────────────────
    print('\n=== get_nearest_frame ===')
    frame_ts_arr = demo['frame_ts']
    rgb_frames   = demo['rgb_frames']

    # Exact timestamp
    frame_exact = get_nearest_frame(frame_ts_arr, int(frame_ts_arr[5]), rgb_frames)
    check('shape (120, 160, 3)',
          frame_exact.shape == (120, 160, 3), f'got {frame_exact.shape}')
    check('exact timestamp returns frame[5]',
          np.array_equal(frame_exact, rgb_frames[5]))

    # Slightly offset timestamp should still return the closest frame
    offset_us    = int(frame_ts_arr[5]) + 100
    frame_offset = get_nearest_frame(frame_ts_arr, offset_us, rgb_frames)
    check('small offset still returns closest frame',
          np.array_equal(frame_offset, rgb_frames[5]))

    # ── load_all_demos ────────────────────────────────────────────────────────
    print('\n=== load_all_demos ===')
    tmpdir  = tempfile.mkdtemp()
    h5_b    = generate_synthetic_demo()
    shutil.copy(h5_path, os.path.join(tmpdir, 'demo_000.h5'))
    shutil.copy(h5_b,    os.path.join(tmpdir, 'demo_001.h5'))

    all_demos = load_all_demos(tmpdir)
    check('returns 2 demos', len(all_demos) == 2, f'got {len(all_demos)}')
    check('all demos have telemetry key',
          all('telemetry' in d for d in all_demos))
    check('all demos have correct telemetry shape',
          all(d['telemetry'].shape == (200,) for d in all_demos))

    os.unlink(h5_path)
    os.unlink(h5_b)
    shutil.rmtree(tmpdir)

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
