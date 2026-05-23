"""
Dataset augmentation for the VLA robotic-arm training pipeline.

augment_sample(sample, aug_id)  → augmented copy of a timestep dict
build_training_set(all_timesteps, augmentation_factor=4)  → flat list

Augmentation IDs
----------------
0  original           no change
1  joint_noise        joints_deg += N(0, 0.05) clipped to ±0.15 deg
2  load_joint_noise   loads += N(0, 0.03) clipped to [0, 1]; + joint_noise
3  flip               rgb_frame flipped L↔R; joints_deg[0] sign-flipped (J0)

build_training_set
------------------
1. Subsample every 6th timestep (indices 0, 6, 12, …).
2. Apply all augmentation_factor augmentations (aug_ids 0 .. factor-1).
3. Return a flat list — length == ceil(len(all_timesteps) / 6) × factor.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

# ── noise parameters ──────────────────────────────────────────────────────────
_JOINT_NOISE_STD  = 0.05   # deg  → 3σ = 0.15 deg
_JOINT_NOISE_CLIP = 0.15   # deg
_LOAD_NOISE_STD   = 0.03   # normalised load (0–1)
_LOAD_NOISE_CLIP  = 0.10   # normalised load


# ── public API ────────────────────────────────────────────────────────────────

def augment_sample(sample: dict[str, Any], aug_id: int) -> dict[str, Any]:
    """
    Return an augmented deep-copy of *sample*.

    sample keys expected (from skill_segmenter.segment_demo):
        joints_deg   np.ndarray (4,) float32
        loads        np.ndarray (4,) float32  normalised 0–1
        rgb_frame    np.ndarray (H, W, 3) uint8
        … all other keys are passed through unchanged …

    aug_id:
        0  original
        1  joint_noise
        2  load_noise + joint_noise
        3  horizontal_flip + J0_sign_flip
    """
    if aug_id not in (0, 1, 2, 3):
        raise ValueError(f"aug_id must be 0-3, got {aug_id}")

    out = copy.deepcopy(sample)

    if aug_id == 0:
        return out

    if aug_id == 1:
        _apply_joint_noise(out)

    elif aug_id == 2:
        _apply_load_noise(out)
        _apply_joint_noise(out)

    elif aug_id == 3:
        _apply_flip(out)

    return out


def build_training_set(
    all_timesteps: list[dict[str, Any]],
    augmentation_factor: int = 4,
) -> list[dict[str, Any]]:
    """
    Subsample every 6th timestep then expand with *augmentation_factor* augs.

    Returns a flat list of augmented sample dicts.  Each output sample has
    an additional key ``aug_id`` (int) recording which augmentation was used.
    """
    subsampled = all_timesteps[::6]
    result: list[dict[str, Any]] = []
    for sample in subsampled:
        for aug_id in range(augmentation_factor):
            aug = augment_sample(sample, aug_id)
            aug['aug_id'] = aug_id
            result.append(aug)
    return result


# ── private helpers ───────────────────────────────────────────────────────────

def _apply_joint_noise(sample: dict[str, Any]) -> None:
    noise = np.random.normal(0.0, _JOINT_NOISE_STD, size=4).astype(np.float32)
    noise = np.clip(noise, -_JOINT_NOISE_CLIP, _JOINT_NOISE_CLIP)
    sample['joints_deg'] = sample['joints_deg'] + noise


def _apply_load_noise(sample: dict[str, Any]) -> None:
    noise = np.random.normal(0.0, _LOAD_NOISE_STD, size=4).astype(np.float32)
    noise = np.clip(noise, -_LOAD_NOISE_CLIP, _LOAD_NOISE_CLIP)
    sample['loads'] = np.clip(sample['loads'] + noise, 0.0, 1.0).astype(np.float32)


def _apply_flip(sample: dict[str, Any]) -> None:
    sample['rgb_frame']   = np.ascontiguousarray(sample['rgb_frame'][:, ::-1, :])
    sample['joints_deg']  = sample['joints_deg'].copy()
    sample['joints_deg'][0] = -sample['joints_deg'][0]   # J0 sign flip


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    all_pass = True

    def check(label: str, condition: bool, detail: str = "") -> None:
        global all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{status}] {label}{suffix}")

    # ── build 10 synthetic samples ────────────────────────────────────────────
    rng = np.random.default_rng(0)

    def _make_sample(i: int) -> dict[str, Any]:
        return {
            'timestep_idx':  i,
            'timestamp_ms':  i * 20,
            'joints_deg':    rng.uniform(-30.0, 60.0, 4).astype(np.float32),
            'speeds':        rng.uniform(0.0,  50.0, 4).astype(np.float32),
            'loads':         rng.uniform(0.0,   1.0, 4).astype(np.float32),
            'fk_z_m':        float(rng.uniform(0.0, 0.3)),
            'contact_flag':  int(rng.integers(0, 2)),
            'gripper_pos_mm':float(rng.uniform(0.0, 50.0)),
            'min_tof_m':     float(rng.uniform(0.01, 0.5)),
            'rgb_frame':     rng.integers(0, 256, (120, 160, 3), dtype=np.uint8),
            'skill_label':   rng.choice(['REACH', 'GRASP', 'LIFT', 'PLACE']),
        }

    samples = [_make_sample(i) for i in range(10)]

    # ── aug_id=0: identical to input ─────────────────────────────────────────
    print("=== aug_id 0: identity ===")
    for i, s in enumerate(samples):
        out = augment_sample(s, aug_id=0)
        check(
            f"sample {i}: joints_deg unchanged",
            np.array_equal(out['joints_deg'], s['joints_deg']),
        )
        check(
            f"sample {i}: rgb_frame unchanged",
            np.array_equal(out['rgb_frame'], s['rgb_frame']),
        )
        check(
            f"sample {i}: loads unchanged",
            np.array_equal(out['loads'], s['loads']),
        )

    # ── aug_id=1: joint noise ─────────────────────────────────────────────────
    print("\n=== aug_id 1: joint_noise ===")
    np.random.seed(7)
    all_deltas: list[np.ndarray] = []
    for i, s in enumerate(samples):
        out = augment_sample(s, aug_id=1)
        check(
            f"sample {i}: joints_deg shape preserved",
            out['joints_deg'].shape == (4,),
        )
        check(
            f"sample {i}: rgb_frame unchanged",
            np.array_equal(out['rgb_frame'], s['rgb_frame']),
        )
        delta = np.abs(out['joints_deg'] - s['joints_deg'])
        all_deltas.append(delta)
        check(
            f"sample {i}: joint noise within ±0.15 deg",
            float(delta.max()) <= _JOINT_NOISE_CLIP + 1e-6,
            f"max_delta={float(delta.max()):.4f}",
        )
    # across all 10 samples noise should be non-zero at least once
    total_noise = np.concatenate(all_deltas).max()
    check(
        "joint noise is non-zero (augmentation is active)",
        total_noise > 0.0,
        f"max_delta={total_noise:.4f}",
    )

    # ── aug_id=2: load + joint noise ──────────────────────────────────────────
    print("\n=== aug_id 2: load_joint_noise ===")
    np.random.seed(13)
    for i, s in enumerate(samples):
        out = augment_sample(s, aug_id=2)
        check(
            f"sample {i}: joints_deg shape preserved",
            out['joints_deg'].shape == (4,),
        )
        check(
            f"sample {i}: loads shape preserved",
            out['loads'].shape == (4,),
        )
        check(
            f"sample {i}: loads in [0, 1]",
            float(out['loads'].min()) >= 0.0 and float(out['loads'].max()) <= 1.0,
            f"range=[{float(out['loads'].min()):.3f}, {float(out['loads'].max()):.3f}]",
        )
        check(
            f"sample {i}: rgb_frame unchanged",
            np.array_equal(out['rgb_frame'], s['rgb_frame']),
        )

    # ── aug_id=3: horizontal flip + J0 sign ───────────────────────────────────
    print("\n=== aug_id 3: flip ===")
    for i, s in enumerate(samples):
        out = augment_sample(s, aug_id=3)

        # image must be flipped left-right
        expected_frame = s['rgb_frame'][:, ::-1, :]
        check(
            f"sample {i}: rgb_frame flipped L↔R",
            np.array_equal(out['rgb_frame'], expected_frame),
        )

        # J0 must be negated
        check(
            f"sample {i}: J0 sign-flipped",
            abs(out['joints_deg'][0] + s['joints_deg'][0]) < 1e-5,
            f"orig={s['joints_deg'][0]:.3f}, aug={out['joints_deg'][0]:.3f}",
        )

        # J1, J2, J3 unchanged
        check(
            f"sample {i}: J1-J3 unchanged",
            np.allclose(out['joints_deg'][1:], s['joints_deg'][1:]),
        )

        # loads unchanged
        check(
            f"sample {i}: loads unchanged",
            np.array_equal(out['loads'], s['loads']),
        )

    # ── invalid aug_id raises ValueError ─────────────────────────────────────
    print("\n=== invalid aug_id ===")
    try:
        augment_sample(samples[0], aug_id=99)
        check("aug_id=99 raises ValueError", False)
    except ValueError:
        check("aug_id=99 raises ValueError", True)

    # ── deep-copy isolation ───────────────────────────────────────────────────
    print("\n=== deep-copy isolation ===")
    s = _make_sample(99)
    orig_joints = s['joints_deg'].copy()
    out = augment_sample(s, aug_id=1)
    out['joints_deg'] += 999.0   # mutate the output
    check(
        "mutating output does not affect original",
        np.array_equal(s['joints_deg'], orig_joints),
    )

    # ── build_training_set ────────────────────────────────────────────────────
    print("\n=== build_training_set ===")
    # 60 samples → subsample every 6th → 10 → ×4 augmentations → 40
    big = [_make_sample(i) for i in range(60)]
    ts = build_training_set(big, augmentation_factor=4)
    expected_len = (60 // 6) * 4   # 10 × 4 = 40
    check(f"output length == {expected_len}", len(ts) == expected_len,
          f"got {len(ts)}")

    aug_ids_present = {s['aug_id'] for s in ts}
    check("aug_ids 0-3 all present", aug_ids_present == {0, 1, 2, 3})

    check("all outputs have 'joints_deg'",
          all('joints_deg' in s for s in ts))
    check("all outputs have 'rgb_frame'",
          all('rgb_frame'  in s for s in ts))
    check("all outputs have 'skill_label'",
          all('skill_label' in s for s in ts))

    # subsample alignment: first output group should map to index 0
    check("first group timestep_idx == 0",
          ts[0]['timestep_idx'] == 0 and ts[1]['timestep_idx'] == 0)
    # second group should map to index 6
    check("second group timestep_idx == 6",
          ts[4]['timestep_idx'] == 6 and ts[5]['timestep_idx'] == 6)

    # non-multiple-of-6 length: 65 samples → 11 subsampled → 44
    big65 = [_make_sample(i) for i in range(65)]
    ts65 = build_training_set(big65, augmentation_factor=4)
    check("65-sample set: output length == 44", len(ts65) == 44,
          f"got {len(ts65)}")

    # augmentation_factor=1 (identity only)
    ts_id = build_training_set(big, augmentation_factor=1)
    check("factor=1: output length == 10", len(ts_id) == 10)
    check("factor=1: all aug_ids == 0",
          all(s['aug_id'] == 0 for s in ts_id))

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
