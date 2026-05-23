"""
dataset/skill_segmenter.py – Rule-based skill segmentation for demonstrations.

Labels each telemetry timestep with one of: REACH | GRASP | LIFT | PLACE.

Algorithm
---------
1. Per-timestep features are derived from TELEMETRY_DTYPE fields.
2. A forward-pass state machine (label_timestep + prev_label) resolves
   the LIFT/PLACE ambiguity that instantaneous features alone cannot.
3. A median filter (window=5) smoothes transient mis-labels at boundaries.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import median_filter

_HERE = Path(__file__).resolve().parent   # dataset/
_ROOT = _HERE.parent                      # vla-robotic-arm/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset.hdf5_reader import (   # noqa: E402
    generate_synthetic_demo,
    get_logical_joints,
    get_nearest_frame,
    load_demo,
)

# ── thresholds ────────────────────────────────────────────────────────────────
THRESHOLDS: dict[str, float] = {
    'load_contact':  0.30,   # normalized load (0–1) above which grip is detected
    'vel_stop':      5.0,    # servo_vel raw units; below this arm is stationary
    'lift_j1_angle': 45.0,   # degrees; J1 above threshold confirms arm is elevated
    'lift_height':   0.08,   # metres; FK end-effector height above shoulder pivot
    'tof_approach':  0.05,   # metres; ToF < 50 mm flags close approach to object
}

_MAX_LOAD     = 1023.0   # Dynamixel max load ticks (denominator for normalization)
_TOF_SCALE    = 1e-3     # mm → m

# 3-link planar arm: shoulder pivot → elbow → wrist → tool tip
_L1 = 0.12   # upper arm   m
_L2 = 0.10   # forearm     m
_L3 = 0.06   # tool        m

_SKILL_LABELS = ['REACH', 'GRASP', 'LIFT', 'PLACE']
_LABEL_TO_INT = {lbl: i for i, lbl in enumerate(_SKILL_LABELS)}
_INT_TO_LABEL = {i: lbl for lbl, i in _LABEL_TO_INT.items()}
_MEDIAN_WINDOW = 5


# ── feature helpers ───────────────────────────────────────────────────────────

def forward_kinematics_z(
    j1_deg: float,
    j2_deg: float = 0.0,
    j3_deg: float = 0.0,
) -> float:
    """
    End-effector height above the shoulder pivot (metres) for a 3-link planar arm.

    z = L1·sin(J1) + L2·sin(J1+J2) + L3·sin(J1+J2+J3)

    Base rotation J0 does not affect height and is ignored.
    """
    j1 = math.radians(j1_deg)
    j2 = math.radians(j2_deg)
    j3 = math.radians(j3_deg)
    return (
        _L1 * math.sin(j1)
        + _L2 * math.sin(j1 + j2)
        + _L3 * math.sin(j1 + j2 + j3)
    )


def get_logical_speeds(telemetry_row: np.ndarray) -> np.ndarray:
    """
    4-element absolute-speed array [J0, J1, J2, J3] in servo_vel raw units.
    J1 is the average of servo channels 1 and 2.

    Accepts both a numpy void (telem[i]) and a 1-element slice (telem[i:i+1]).
    """
    vel = telemetry_row['servo_vel']
    if vel.ndim > 1:
        vel = vel[0]
    return np.array([
        abs(int(vel[0])),
        abs((int(vel[1]) + int(vel[2])) / 2.0),
        abs(int(vel[3])),
        abs(int(vel[4])),
    ], dtype=np.float32)


def get_logical_loads(telemetry_row: np.ndarray) -> np.ndarray:
    """
    4-element normalized load array [J0, J1, J2, J3] in range [0, 1].
    J1 is the average of servo channels 1 and 2.

    Accepts both a numpy void (telem[i]) and a 1-element slice (telem[i:i+1]).
    """
    load = telemetry_row['servo_load']
    if load.ndim > 1:
        load = load[0]
    return np.array([
        abs(int(load[0])),
        abs((int(load[1]) + int(load[2])) / 2.0),
        abs(int(load[3])),
        abs(int(load[4])),
    ], dtype=np.float32) / _MAX_LOAD


# ── per-timestep labeler ──────────────────────────────────────────────────────

def label_timestep(
    contact:       bool,
    avg_load_norm: float,
    max_speed:     float,
    j1_deg:        float,
    fk_z_m:        float,
    min_tof_m:     float,
    prev_label:    str = 'REACH',
) -> str:
    """
    Assign a skill label for one timestep from per-step features plus the
    previous label (state-machine context).

    Threshold usage
    ---------------
    load_contact  — detect grip via servo load when contact_flag may lag
    vel_stop      — distinguish stationary phases (GRASP/PLACE) from LIFT
    lift_j1_angle — shoulder angle > 45° confirms the arm is elevated
    lift_height   — FK height > 0.08 m confirms the end-effector is lifted
    tof_approach  — ToF < 50 mm is a backup GRASP trigger before contact_flag fires

    State machine
    -------------
    REACH → GRASP : first timestep where is_contact is True
    GRASP → LIFT  : is_high_j1 OR fk_z_m > lift_height
    LIFT  → PLACE : arm becomes stationary (max_speed < vel_stop) while loaded
    PLACE         : terminal within a single demo
    """
    is_contact     = contact or avg_load_norm > THRESHOLDS['load_contact']
    is_slow        = max_speed  < THRESHOLDS['vel_stop']
    is_high_j1     = j1_deg     > THRESHOLDS['lift_j1_angle']
    is_lifted      = is_high_j1 or fk_z_m > THRESHOLDS['lift_height']
    is_approaching = min_tof_m  < THRESHOLDS['tof_approach']

    if not is_contact:
        # tof_approach: pre-contact grip trigger when contact_flag is delayed
        if is_approaching and is_slow:
            return 'GRASP'
        return 'REACH'

    if prev_label == 'REACH':
        return 'GRASP'
    if prev_label == 'GRASP':
        return 'LIFT' if is_lifted else 'GRASP'
    if prev_label == 'LIFT':
        return 'PLACE' if is_slow else 'LIFT'
    return 'PLACE'   # terminal


# ── full demo segmentation ────────────────────────────────────────────────────

def segment_demo(demo_dict: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Segment a loaded demo into a list of per-timestep dicts with skill labels.

    Each dict contains:
        timestep_idx   int
        timestamp_ms   int
        joints_deg     np.ndarray (4,) float32   [J0, J1, J2, J3] in degrees
        speeds         np.ndarray (4,) float32   raw servo_vel units
        loads          np.ndarray (4,) float32   normalized 0–1
        fk_z_m         float                     end-effector height (m)
        contact_flag   bool
        gripper_pos_mm float
        min_tof_m      float                     minimum ToF reading (m)
        rgb_frame      np.ndarray (H, W, 3) uint8
        skill_label    str                        REACH | GRASP | LIFT | PLACE

    Skill boundaries are smoothed with a median filter (window=5, mode=nearest).
    """
    telem       = demo_dict['telemetry']
    frame_ts    = demo_dict['frame_ts']
    rgb_frames  = demo_dict['rgb_frames']
    instruction = demo_dict.get('instruction', '')

    raw_labels: list[str]      = []
    cache_joints: list[np.ndarray] = []
    cache_speeds: list[np.ndarray] = []
    cache_loads:  list[np.ndarray] = []
    cache_fk_z:   list[float]      = []
    cache_tof:    list[float]      = []

    prev = 'REACH'

    for i in range(len(telem)):
        row    = telem[i]
        joints = get_logical_joints(row)
        speeds = get_logical_speeds(row)
        loads  = get_logical_loads(row)

        j1_deg = float(joints[1])
        j2_deg = float(joints[2])
        j3_deg = float(joints[3])
        fk_z   = forward_kinematics_z(j1_deg, j2_deg, j3_deg)

        tof_grid  = row['tof_grid']   # shape (8, 8) mm
        min_tof_m = float(np.min(tof_grid)) * _TOF_SCALE

        contact   = bool(row['contact_flag'])
        avg_load  = float(np.mean(loads))
        max_speed = float(np.max(speeds))

        label = label_timestep(
            contact, avg_load, max_speed, j1_deg, fk_z, min_tof_m, prev
        )
        raw_labels.append(label)
        prev = label

        cache_joints.append(joints)
        cache_speeds.append(speeds)
        cache_loads.append(loads)
        cache_fk_z.append(fk_z)
        cache_tof.append(min_tof_m)

    # ── smooth label boundaries ────────────────────────────────────────────────
    # Encode as integers (monotone sequence REACH=0 < GRASP=1 < LIFT=2 < PLACE=3)
    # so the median filter preserves ordering at transitions.
    label_ints  = np.array([_LABEL_TO_INT[l] for l in raw_labels], dtype=np.float32)
    smoothed    = median_filter(label_ints, size=_MEDIAN_WINDOW, mode='nearest')
    smooth_lbls = [_INT_TO_LABEL[int(round(v))] for v in smoothed]

    # ── build output ───────────────────────────────────────────────────────────
    timesteps: list[dict[str, Any]] = []
    for i, row in enumerate(telem):
        ts_us = int(row['timestamp_ms']) * 1000   # ms → µs to match frame_ts
        frame = get_nearest_frame(frame_ts, ts_us, rgb_frames)

        timesteps.append({
            'timestep_idx':   i,
            'timestamp_ms':   int(row['timestamp_ms']),
            'joints_deg':     cache_joints[i],
            'speeds':         cache_speeds[i],
            'loads':          cache_loads[i],
            'fk_z_m':         cache_fk_z[i],
            'contact_flag':   bool(row['contact_flag']),
            'imu_gyro_rms':   float(row['imu_gyro_rms']),
            'gripper_pos_mm': float(row['gripper_pos']),
            'min_tof_m':      cache_tof[i],
            'tof_grid':       np.array(row['tof_grid'], dtype=np.uint16),
            'rgb_frame':      frame,
            'skill_label':    smooth_lbls[i],
            'instruction':    instruction,
        })

    return timesteps


# ── self-test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import os

    all_pass = True

    def check(label: str, cond: bool, detail: str = '') -> None:
        global all_pass
        status = 'PASS' if cond else 'FAIL'
        if not cond:
            all_pass = False
        suffix = f'  ({detail})' if detail else ''
        print(f'  [{status}] {label}{suffix}')

    # ── forward_kinematics_z ──────────────────────────────────────────────────
    print('=== forward_kinematics_z ===')
    z_90 = forward_kinematics_z(90.0)
    check('J1=90°, J2=J3=0 → z == L1+L2+L3 == 0.28 m',
          abs(z_90 - 0.28) < 1e-9, f'got {z_90:.4f}')
    check('J1=0° → z == 0',
          abs(forward_kinematics_z(0.0)) < 1e-9)
    check('J1=16.8° → z > 0.08 m (lift_height threshold)',
          forward_kinematics_z(16.8) > THRESHOLDS['lift_height'],
          f'got {forward_kinematics_z(16.8):.4f}')
    check('J1=15.0° → z < 0.08 m',
          forward_kinematics_z(15.0) < THRESHOLDS['lift_height'],
          f'got {forward_kinematics_z(15.0):.4f}')
    # non-zero J2 increases height
    z_j2 = forward_kinematics_z(45.0, j2_deg=30.0)
    check('J2=30° increases z compared to J2=0°',
          z_j2 > forward_kinematics_z(45.0),
          f'{z_j2:.4f} vs {forward_kinematics_z(45.0):.4f}')

    # ── get_logical_speeds ────────────────────────────────────────────────────
    print('\n=== get_logical_speeds ===')
    h5 = generate_synthetic_demo()
    demo = load_demo(h5)
    telem = demo['telemetry']

    # LIFT phase t=120 — servo_vel should be 30–80 raw units
    spd_lift = get_logical_speeds(telem[120])
    check('shape (4,)',   spd_lift.shape == (4,), f'got {spd_lift.shape}')
    check('dtype float32', spd_lift.dtype == np.float32)
    check('LIFT speeds > vel_stop (5.0)',
          float(np.max(spd_lift)) > THRESHOLDS['vel_stop'],
          f'max={float(np.max(spd_lift)):.1f}')

    # GRASP phase t=70 — servo_vel should be 0–5
    spd_grasp = get_logical_speeds(telem[70])
    check('GRASP speeds < vel_stop (5.0)',
          float(np.max(spd_grasp)) < THRESHOLDS['vel_stop'],
          f'max={float(np.max(spd_grasp)):.1f}')

    # accepts 1-element slice
    spd_slice = get_logical_speeds(telem[120:121])
    check('void and slice results match', np.allclose(spd_lift, spd_slice))

    # ── get_logical_loads ─────────────────────────────────────────────────────
    print('\n=== get_logical_loads ===')
    ld_grasp = get_logical_loads(telem[70])
    check('shape (4,)',     ld_grasp.shape == (4,))
    check('dtype float32',  ld_grasp.dtype == np.float32)
    check('values in [0,1]',
          bool(np.all(ld_grasp >= 0) and np.all(ld_grasp <= 1)))
    check('GRASP loads > load_contact (0.30)',
          float(np.mean(ld_grasp)) > THRESHOLDS['load_contact'],
          f'mean={float(np.mean(ld_grasp)):.3f}')

    ld_reach = get_logical_loads(telem[10])
    check('REACH loads < load_contact (0.30)',
          float(np.mean(ld_reach)) < THRESHOLDS['load_contact'],
          f'mean={float(np.mean(ld_reach)):.3f}')

    # ── label_timestep: unit tests ────────────────────────────────────────────
    print('\n=== label_timestep ===')
    # No contact → REACH
    check('no contact → REACH',
          label_timestep(False, 0.1, 50.0, 0.0, 0.0, 0.2, 'REACH') == 'REACH')
    # tof backup: no contact, approaching, slow → GRASP
    check('no contact + tof<50mm + slow → GRASP',
          label_timestep(False, 0.1, 2.0, 0.0, 0.0, 0.03, 'REACH') == 'GRASP')
    # no contact, tof close but NOT slow → REACH (moving, not grasping yet)
    check('no contact + tof<50mm + NOT slow → REACH',
          label_timestep(False, 0.1, 50.0, 0.0, 0.0, 0.03, 'REACH') == 'REACH')
    # contact, prev=REACH → GRASP
    check('contact + prev=REACH → GRASP',
          label_timestep(True, 0.4, 3.0, 0.0, 0.0, 0.2, 'REACH') == 'GRASP')
    # contact, prev=GRASP, not lifted → GRASP
    check('contact + prev=GRASP + not lifted → GRASP',
          label_timestep(True, 0.4, 3.0, 10.0, 0.03, 0.2, 'GRASP') == 'GRASP')
    # contact, prev=GRASP, high J1 → LIFT
    check('contact + prev=GRASP + J1>45° → LIFT',
          label_timestep(True, 0.4, 50.0, 50.0, 0.2, 0.2, 'GRASP') == 'LIFT')
    # contact, prev=GRASP, high fk_z → LIFT
    check('contact + prev=GRASP + fk_z>0.08 → LIFT',
          label_timestep(True, 0.4, 50.0, 20.0, 0.09, 0.2, 'GRASP') == 'LIFT')
    # contact, prev=LIFT, NOT slow → LIFT
    check('contact + prev=LIFT + moving → LIFT',
          label_timestep(True, 0.4, 50.0, 50.0, 0.2, 0.2, 'LIFT') == 'LIFT')
    # contact, prev=LIFT, slow → PLACE
    check('contact + prev=LIFT + slow → PLACE',
          label_timestep(True, 0.4, 2.0, 50.0, 0.2, 0.2, 'LIFT') == 'PLACE')
    # PLACE is terminal
    check('PLACE is terminal (stays PLACE)',
          label_timestep(True, 0.4, 2.0, 30.0, 0.14, 0.2, 'PLACE') == 'PLACE')
    # load alone triggers contact (no contact_flag)
    check('high load alone triggers contact (load_contact threshold)',
          label_timestep(False, 0.4, 3.0, 0.0, 0.0, 0.2, 'REACH') == 'GRASP')

    # ── segment_demo ──────────────────────────────────────────────────────────
    print('\n=== segment_demo: structure ===')
    timesteps = segment_demo(demo)

    check('returns 200 timestep dicts',
          len(timesteps) == 200, f'got {len(timesteps)}')

    expected_keys = {
        'timestep_idx', 'timestamp_ms', 'joints_deg', 'speeds', 'loads',
        'fk_z_m', 'contact_flag', 'gripper_pos_mm', 'min_tof_m',
        'rgb_frame', 'skill_label',
    }
    check('all expected keys present in each dict',
          all(expected_keys <= set(ts.keys()) for ts in timesteps))
    check('skill_label is valid string in all dicts',
          all(ts['skill_label'] in _SKILL_LABELS for ts in timesteps))
    check('joints_deg shape (4,) float32 in all dicts',
          all(ts['joints_deg'].shape == (4,) and
              ts['joints_deg'].dtype == np.float32 for ts in timesteps))
    check('rgb_frame shape (120,160,3) uint8 in all dicts',
          all(ts['rgb_frame'].shape == (120, 160, 3) and
              ts['rgb_frame'].dtype == np.uint8 for ts in timesteps))

    # ── segment_demo: all 4 labels appear ─────────────────────────────────────
    print('\n=== segment_demo: label coverage ===')
    labels = [ts['skill_label'] for ts in timesteps]
    label_set = set(labels)

    check('all 4 skill labels present',
          label_set == set(_SKILL_LABELS), f'got {label_set}')

    counts = {l: labels.count(l) for l in _SKILL_LABELS}
    check('each label appears > 5 times',
          all(v > 5 for v in counts.values()),
          '  '.join(f'{l}={n}' for l, n in counts.items()))

    # ── segment_demo: ordering (physical sense) ───────────────────────────────
    print('\n=== segment_demo: boundary ordering ===')
    first = {l: labels.index(l) for l in _SKILL_LABELS}

    check('REACH  < GRASP  (ordering)',
          first['REACH']  < first['GRASP'],
          f"REACH@{first['REACH']} GRASP@{first['GRASP']}")
    check('GRASP  < LIFT   (ordering)',
          first['GRASP']  < first['LIFT'],
          f"GRASP@{first['GRASP']} LIFT@{first['LIFT']}")
    check('LIFT   < PLACE  (ordering)',
          first['LIFT']   < first['PLACE'],
          f"LIFT@{first['LIFT']} PLACE@{first['PLACE']}")

    check('no REACH after first GRASP',
          'REACH' not in labels[first['GRASP']:])
    check('no GRASP after first LIFT',
          'GRASP' not in labels[first['LIFT']:])
    check('no LIFT after first PLACE',
          'LIFT'  not in labels[first['PLACE']:])

    # ── segment_demo: boundaries at physically reasonable timesteps ────────────
    print('\n=== segment_demo: boundary positions ===')
    check('REACH starts at t=0',
          first['REACH'] == 0,
          f"got t={first['REACH']}")
    check('GRASP starts at t=50 ± 3 (contact onset)',
          first['GRASP'] <= 53,
          f"got t={first['GRASP']}")
    check('LIFT starts in t=[100, 130] (height threshold ~t=114)',
          100 <= first['LIFT'] <= 130,
          f"got t={first['LIFT']}")
    check('PLACE starts in t=[145, 160] (velocity drops at t=150)',
          145 <= first['PLACE'] <= 160,
          f"got t={first['PLACE']}")

    # ── segment_demo: feature values at key timesteps ─────────────────────────
    print('\n=== segment_demo: physical feature checks ===')
    # All LIFT timesteps must have J1 > 0° (arm has risen)
    lift_ts = [ts for ts in timesteps if ts['skill_label'] == 'LIFT']
    check('all LIFT timesteps have J1 > 0°',
          all(ts['joints_deg'][1] > 0.0 for ts in lift_ts),
          f'min J1={min(ts["joints_deg"][1] for ts in lift_ts):.2f}°')

    # All PLACE timesteps have fk_z > 0 (arm is not at floor)
    place_ts = [ts for ts in timesteps if ts['skill_label'] == 'PLACE']
    check('all PLACE timesteps have fk_z > 0.0',
          all(ts['fk_z_m'] > 0.0 for ts in place_ts),
          f'min fk_z={min(ts["fk_z_m"] for ts in place_ts):.4f} m')

    # REACH timesteps all have contact_flag=False
    reach_ts = [ts for ts in timesteps if ts['skill_label'] == 'REACH']
    check('all REACH timesteps have contact_flag=False',
          all(not ts['contact_flag'] for ts in reach_ts))

    # GRASP timesteps all have contact_flag=True
    grasp_ts = [ts for ts in timesteps if ts['skill_label'] == 'GRASP']
    check('all GRASP timesteps have contact_flag=True',
          all(ts['contact_flag'] for ts in grasp_ts))

    # median filter: check that no single-timestep islands exist
    # (any label that appears only once means filter didn't work)
    runs = [(labels[0], 1)]
    for l in labels[1:]:
        if l == runs[-1][0]:
            runs[-1] = (l, runs[-1][1] + 1)
        else:
            runs.append((l, 1))
    min_run = min(r[1] for r in runs)
    check('median filter: no single-frame label islands',
          min_run >= 2,
          f'shortest run={min_run}')

    os.unlink(h5)

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
