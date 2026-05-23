from __future__ import annotations

import logging
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)

CHUNK_SIZE = 8

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _MockModel(nn.Module):
    """Stub used when the TorchScript checkpoint does not exist yet.

    Returns random-but-correctly-shaped outputs. Deltas are scaled ×0.01
    so the IK-primary steering loop stays stable during development.
    """

    def forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        skill_logits = torch.randn(1, 4)
        delta_joints = torch.randn(1, CHUNK_SIZE, 4) * 0.01
        return skill_logits, delta_joints


class VLARuntime:
    """Thin wrapper around a TorchScript VLA policy checkpoint.

    Falls back to _MockModel when the checkpoint file does not exist,
    allowing end-to-end pipeline testing before training is complete.
    """

    CHUNK_SIZE = CHUNK_SIZE

    def __init__(self, model_path: str, lang_encoder) -> None:
        if os.path.exists(model_path):
            self.model: nn.Module = torch.jit.load(model_path)
            log.info("VLARuntime: loaded checkpoint %s", model_path)
        else:
            warnings.warn(
                f"VLARuntime: checkpoint '{model_path}' not found — using _MockModel.",
                UserWarning,
                stacklevel=2,
            )
            self.model = _MockModel()
        self.model.eval()
        self.encoder = lang_encoder
        self._chunk_buffer: np.ndarray | None = None

    def predict(
        self,
        rgb_frame: np.ndarray,
        joint_state_4d: np.ndarray,
        skill_onehot: np.ndarray,
        instruction: str,
        contact_rms: float,
        tof_z_m: float,
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Run one inference step.

        Returns:
            skill_pred  : int — argmax of skill logits (0=REACH … 3=PLACE)
            delta_step0 : ndarray[4] — first step of the action chunk
            chunk_buffer: ndarray[8, 4] — full 8-step action chunk
        """
        # 1. Preprocess image → (1, 3, 224, 224) float32, ImageNet-normalised
        img = cv2.resize(rgb_frame, (224, 224)).astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        rgb_t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)  # (1,3,224,224)

        # 2. Language embedding (LanguageEncoder caches repeated instructions)
        lang_emb = self.encoder.encode(instruction)  # ndarray (512,)

        # 3. Build batch dict
        batch = {
            "rgb":          rgb_t,
            "joint_state":  torch.from_numpy(
                                np.asarray(joint_state_4d, dtype=np.float32)
                            ).unsqueeze(0),
            "skill_onehot": torch.from_numpy(
                                np.asarray(skill_onehot, dtype=np.float32)
                            ).unsqueeze(0),
            "lang_emb":     torch.from_numpy(
                                np.asarray(lang_emb, dtype=np.float32)
                            ).unsqueeze(0),
            "contact_rms":  torch.tensor([[float(contact_rms)]], dtype=torch.float32),
            "tof_scalar":   torch.tensor([[float(tof_z_m)]],     dtype=torch.float32),
        }

        # 4. Forward pass (no gradient needed)
        with torch.no_grad():
            skill_logits, delta_joints = self.model(batch)

        # 5. Decode
        skill_pred  = int(skill_logits.argmax(dim=-1).item())
        chunk_buf   = delta_joints[0].numpy()          # (8, 4)
        delta_step0 = chunk_buf[0].copy()              # (4,)

        self._chunk_buffer = chunk_buf
        return skill_pred, delta_step0, chunk_buf
