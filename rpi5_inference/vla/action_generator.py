from __future__ import annotations

import numpy as np

CHUNK_SIZE = 8


class ActionGenerator:
    """Steps through an 8-step action chunk one tick at a time.

    Decouples VLA inference cadence from command output cadence.
    Call set_chunk() when a new prediction arrives; call step() every tick.
    """

    CHUNK_SIZE = CHUNK_SIZE

    def __init__(self) -> None:
        self._chunk: np.ndarray | None = None
        self._step: int = 0

    def set_chunk(self, chunk: np.ndarray) -> None:
        """Store a new (CHUNK_SIZE, 4) delta array and reset the step counter."""
        self._chunk = np.asarray(chunk, dtype=np.float32)
        self._step = 0

    def step(self, joint_state_4d: np.ndarray) -> np.ndarray:
        """Return joint_state + chunk[i] and advance the internal counter.

        Wraps at CHUNK_SIZE. Returns joint_state unchanged if no chunk is set.
        """
        if self._chunk is None:
            return np.asarray(joint_state_4d, dtype=np.float32).copy()
        delta = self._chunk[self._step % CHUNK_SIZE]
        self._step += 1
        return np.asarray(joint_state_4d, dtype=np.float32) + delta

    def reset(self) -> None:
        """Clear the chunk and reset the counter."""
        self._chunk = None
        self._step = 0
