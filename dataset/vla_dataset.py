"""
dataset/vla_dataset.py — PyTorch Dataset for VLA robotic-arm training.

VLADataset(timesteps, language_encoder, normalize_joints=True)
  timesteps        : list[dict]  output of skill_segmenter.segment_demo()
                                 or augmentation.build_training_set()
  language_encoder : object with .encode(str) → np.ndarray (512,) float32
  normalize_joints : bool  map joints_deg to [0, 1] using per-joint bounds

CHUNK_SIZE = 8
Joint normalisation bounds (degrees):
  min : [-150, -90, -120,  0]
  max : [ 150,  90,  120, 90]

__getitem__(idx) → dict
  rgb          torch.float32 (3, 256, 256)  resized, normalised 0-1, CHW
  joint_state  torch.float32 (4,)           normalised to [0, 1]
  skill_onehot torch.float32 (4,)           one-hot of current skill
  contact_rms  torch.float32 (1,)           imu_gyro_rms (dps)
  tof_scalar   torch.float32 (1,)           centre-zone mean ToF in metres
  lang_emb     torch.float32 (512,)         from language_encoder.encode()
  delta_joints torch.float32 (8, 4)         future - current joint positions
  skill_label  torch.int64   scalar         REACH=0 GRASP=1 LIFT=2 PLACE=3

__len__ → len(timesteps) - CHUNK_SIZE

tof_scalar
  Centre zones of the 8×8 ToF grid are flat indices [27, 28, 35, 36].
  Valid readings: 20 < z < 600 mm.  Mean converted to metres.
  Fallback: 0.3 m when no valid readings exist.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── constants ─────────────────────────────────────────────────────────────────

CHUNK_SIZE = 8
IMG_SIZE   = 256

JOINT_MIN   = np.array([-150.0, -90.0, -120.0,  0.0], dtype=np.float32)
JOINT_MAX   = np.array([ 150.0,  90.0,  120.0, 90.0], dtype=np.float32)
JOINT_RANGE = JOINT_MAX - JOINT_MIN          # [300, 180, 240, 90]

_SKILL_TO_INT  = {'REACH': 0, 'GRASP': 1, 'LIFT': 2, 'PLACE': 3}
_TOF_CTR_IDX   = np.array([27, 28, 35, 36])  # centre 2×2 of 8×8 grid
_TOF_VALID_MIN = 20.0    # mm
_TOF_VALID_MAX = 600.0   # mm
_TOF_FALLBACK  = 0.3     # metres


# ── dataset ───────────────────────────────────────────────────────────────────

class VLADataset(Dataset):
    """PyTorch Dataset wrapping the skill-segmented + (optionally) augmented
    timestep list produced by the dataset pipeline."""

    CHUNK_SIZE = CHUNK_SIZE

    def __init__(
        self,
        timesteps:        list[dict[str, Any]],
        language_encoder,
        normalize_joints: bool = True,
    ) -> None:
        self._ts        = timesteps
        self._enc       = language_encoder
        self._norm      = normalize_joints
        self._lang_cache: dict[str, np.ndarray] = {}

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._ts) - self.CHUNK_SIZE

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cur = self._ts[idx]

        return {
            'rgb':          self._rgb(cur),
            'joint_state':  self._joints(cur),
            'skill_onehot': self._onehot(cur),
            'contact_rms':  self._contact(cur),
            'tof_scalar':   self._tof(cur),
            'lang_emb':     self._lang(cur),
            'delta_joints': self._delta(idx, cur),
            'skill_label':  torch.tensor(
                                _SKILL_TO_INT.get(cur['skill_label'], 0),
                                dtype=torch.int64),
        }

    # ── private builders ──────────────────────────────────────────────────────

    def _rgb(self, ts: dict) -> torch.Tensor:
        frame = ts['rgb_frame']                        # (H, W, 3) uint8
        resized = _resize_frame(frame, IMG_SIZE)       # (256, 256, 3) float32
        return torch.from_numpy(resized).permute(2, 0, 1)   # (3, 256, 256)

    def _joints(self, ts: dict) -> torch.Tensor:
        j = ts['joints_deg'].astype(np.float32)
        if self._norm:
            j = np.clip((j - JOINT_MIN) / JOINT_RANGE, 0.0, 1.0)
        return torch.from_numpy(j)

    def _onehot(self, ts: dict) -> torch.Tensor:
        idx = _SKILL_TO_INT.get(ts['skill_label'], 0)
        v   = np.zeros(4, dtype=np.float32)
        v[idx] = 1.0
        return torch.from_numpy(v)

    def _contact(self, ts: dict) -> torch.Tensor:
        val = float(ts.get('imu_gyro_rms', float(ts.get('contact_flag', 0))))
        return torch.tensor([val], dtype=torch.float32)

    def _tof(self, ts: dict) -> torch.Tensor:
        return torch.tensor([_tof_scalar(ts)], dtype=torch.float32)

    def _lang(self, ts: dict) -> torch.Tensor:
        instr = str(ts.get('instruction', ''))
        if instr not in self._lang_cache:
            self._lang_cache[instr] = np.asarray(
                self._enc.encode(instr), dtype=np.float32)
        return torch.from_numpy(self._lang_cache[instr])

    def _delta(self, idx: int, cur: dict) -> torch.Tensor:
        cur_j  = cur['joints_deg'].astype(np.float32)
        deltas = np.empty((self.CHUNK_SIZE, 4), dtype=np.float32)
        for k in range(self.CHUNK_SIZE):
            fut_j    = self._ts[idx + 1 + k]['joints_deg'].astype(np.float32)
            deltas[k] = fut_j - cur_j
        return torch.from_numpy(deltas)


# ── helpers ───────────────────────────────────────────────────────────────────

def _resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """Return (size, size, 3) float32 image normalised to [0, 1]."""
    try:
        import cv2
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        resized = np.asarray(
            Image.fromarray(frame).resize((size, size), Image.BILINEAR))
    return resized.astype(np.float32) / 255.0


def _tof_scalar(ts: dict) -> float:
    """Centre-zone mean of the 8×8 ToF grid in metres.  Fallback: 0.3 m."""
    tof = ts.get('tof_grid')
    if tof is None:
        return _TOF_FALLBACK
    flat  = np.asarray(tof, dtype=np.float32).reshape(-1)   # (64,)
    zones = flat[_TOF_CTR_IDX]
    valid = zones[(zones > _TOF_VALID_MIN) & (zones < _TOF_VALID_MAX)]
    if len(valid) == 0:
        return _TOF_FALLBACK
    return float(np.mean(valid)) / 1000.0   # mm → m


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys as _sys
    import os
    import tempfile

    # Add repo root to path for imports.
    _ROOT_STR = str(Path(__file__).resolve().parent.parent)
    if _ROOT_STR not in _sys.path:
        _sys.path.insert(0, _ROOT_STR)

    from dataset.hdf5_reader    import generate_synthetic_demo, load_demo
    from dataset.skill_segmenter import segment_demo

    all_pass = True

    def check(label: str, cond: bool, detail: str = '') -> None:
        global all_pass
        status = 'PASS' if cond else 'FAIL'
        if not cond:
            all_pass = False
        suffix = f'  ({detail})' if detail else ''
        print(f'  [{status}] {label}{suffix}')

    # ── mock language encoder (avoids loading T5 weights) ─────────────────────
    class _MockEncoder:
        def encode(self, instruction: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(instruction)) % (2**31))
            return rng.standard_normal(512).astype(np.float32)

    # ── build dataset from synthetic pipeline ──────────────────────────────────
    print('=== building dataset from synthetic demo ===')
    h5_path   = generate_synthetic_demo()
    demo      = load_demo(h5_path)
    timesteps = segment_demo(demo)
    os.unlink(h5_path)

    enc     = _MockEncoder()
    dataset = VLADataset(timesteps, enc, normalize_joints=True)

    # ── __len__ ────────────────────────────────────────────────────────────────
    print('\n=== __len__ ===')
    expected_len = len(timesteps) - CHUNK_SIZE
    check(f'len(dataset) == {expected_len}',
          len(dataset) == expected_len,
          f'got {len(dataset)}')

    # ── __getitem__(0) — shapes and dtypes ────────────────────────────────────
    print('\n=== __getitem__(0) shape / dtype ===')
    item = dataset[0]

    EXPECTED = {
        'rgb':          (torch.float32, (3, 256, 256)),
        'joint_state':  (torch.float32, (4,)),
        'skill_onehot': (torch.float32, (4,)),
        'contact_rms':  (torch.float32, (1,)),
        'tof_scalar':   (torch.float32, (1,)),
        'lang_emb':     (torch.float32, (512,)),
        'delta_joints': (torch.float32, (8, 4)),
        'skill_label':  (torch.int64,   ()),
    }

    check('item has exactly 8 keys', set(item.keys()) == set(EXPECTED.keys()),
          f'keys={sorted(item.keys())}')

    for key, (exp_dtype, exp_shape) in EXPECTED.items():
        t = item[key]
        check(f'{key}: dtype == {exp_dtype}',
              t.dtype == exp_dtype,
              f'got {t.dtype}')
        check(f'{key}: shape == {exp_shape}',
              tuple(t.shape) == exp_shape,
              f'got {tuple(t.shape)}')

    # ── rgb values in [0, 1] ──────────────────────────────────────────────────
    print('\n=== value range checks ===')
    rgb = item['rgb']
    check('rgb min >= 0.0', float(rgb.min()) >= 0.0,
          f'min={float(rgb.min()):.4f}')
    check('rgb max <= 1.0', float(rgb.max()) <= 1.0,
          f'max={float(rgb.max()):.4f}')

    # ── joint_state in [0, 1] ─────────────────────────────────────────────────
    js = item['joint_state']
    check('joint_state min >= 0.0', float(js.min()) >= 0.0,
          f'min={float(js.min()):.4f}')
    check('joint_state max <= 1.0', float(js.max()) <= 1.0,
          f'max={float(js.max()):.4f}')

    # ── skill_onehot sums to 1 ────────────────────────────────────────────────
    check('skill_onehot sums to 1.0',
          abs(float(item['skill_onehot'].sum()) - 1.0) < 1e-5)

    # ── delta_joints shape explicit ───────────────────────────────────────────
    check('delta_joints shape == (8, 4)',
          tuple(item['delta_joints'].shape) == (8, 4))

    # ── lang_emb is float32 (512,) ────────────────────────────────────────────
    check('lang_emb shape == (512,)',
          tuple(item['lang_emb'].shape) == (512,))

    # ── tof_scalar in reasonable range ───────────────────────────────────────
    tof_val = float(item['tof_scalar'][0])
    check('tof_scalar > 0',      tof_val > 0.0,      f'got {tof_val:.4f}')
    check('tof_scalar <= 1.0',   tof_val <= 1.0,     f'got {tof_val:.4f}')

    # ── last valid index ──────────────────────────────────────────────────────
    print('\n=== boundary index ===')
    last_item = dataset[len(dataset) - 1]
    check('last index accessible without IndexError', True)
    check('last item delta_joints shape == (8, 4)',
          tuple(last_item['delta_joints'].shape) == (8, 4))

    # ── language embedding is cached ──────────────────────────────────────────
    print('\n=== language cache ===')
    _ = dataset[0]
    _ = dataset[1]
    check('lang cache populated after 2 getitems',
          len(dataset._lang_cache) >= 1)
    instr = list(dataset._lang_cache.keys())[0]
    emb1  = dataset._lang_cache[instr]
    _     = dataset[0]
    emb2  = dataset._lang_cache[instr]
    check('lang embedding is the same object on second call', emb1 is emb2)

    print(f'\nResult: {"ALL PASS" if all_pass else "SOME FAILED"}')
    _sys.exit(0 if all_pass else 1)
