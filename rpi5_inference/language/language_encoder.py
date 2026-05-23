"""
Language instruction encoder using google/flan-t5-small.

encode(instruction) → float32 numpy array of shape (512,)
  — the mean-pooled encoder hidden states, L2-normalised.

Embeddings are cached in a plain dict keyed by the exact instruction
string, so repeated calls with identical text are free after the first.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import T5EncoderModel, AutoTokenizer

MODEL_NAME = "google/flan-t5-small"

# Hidden size for flan-t5-small encoder is 512.
EMBED_DIM = 512


class LanguageEncoder:
    """
    Wraps flan-t5-small encoder with per-string embedding cache.

    Usage:
        enc = LanguageEncoder()
        vec = enc.encode("pick up the red cube")
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load encoder-only: saves memory vs full seq2seq model.
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self._cache: dict[str, np.ndarray] = {}

    # ── public API ────────────────────────────────────────────────────

    def encode(self, instruction: str) -> np.ndarray:
        """
        Return a float32 numpy array of shape (EMBED_DIM,).
        Identical instruction strings return the cached array (same object).
        """
        if instruction in self._cache:
            return self._cache[instruction]

        vec = self._embed(instruction)
        self._cache[instruction] = vec
        return vec

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── internal ──────────────────────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tokens)

        # out.last_hidden_state: (1, seq_len, hidden)
        hidden = out.last_hidden_state[0]           # (seq_len, hidden)
        mask   = tokens["attention_mask"][0].bool() # (seq_len,)
        pooled = hidden[mask].mean(dim=0)           # (hidden,)

        # L2-normalise so cosine similarity == dot product
        norm = pooled.norm()
        if norm > 0:
            pooled = pooled / norm

        return pooled.cpu().float().numpy()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    all_pass = True

    def check(label: str, condition: bool) -> None:
        global all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        print(f"  [{status}] {label}")

    print("Loading LanguageEncoder (downloads model on first run) …")
    enc = LanguageEncoder()
    print(f"  Device : {enc.device}")
    print(f"  Model  : {MODEL_NAME}\n")

    instructions = [
        "pick up the red cube",
        "place the blue cube on the right",
        "grab the green cube and move it left",
    ]

    print("=== Shape and dtype ===")
    vecs = []
    for inst in instructions:
        v = enc.encode(inst)
        check(f'encode("{inst[:30]}…") shape == ({EMBED_DIM},)', v.shape == (EMBED_DIM,))
        check(f'  dtype is float32', v.dtype == np.float32)
        check(f'  L2 norm ≈ 1.0',   abs(np.linalg.norm(v) - 1.0) < 1e-5)
        vecs.append(v)
    print()

    print("=== Cache behaviour ===")
    check("Cache empty before second call is wrong — should have 3 entries",
          enc.cache_size == 3)

    # Re-encode same strings — must hit cache (same object returned)
    for inst, original_vec in zip(instructions, vecs):
        cached = enc.encode(inst)
        check(f'Same object returned for "{inst[:30]}"',
              cached is original_vec)

    check("Cache size unchanged after repeated encodes", enc.cache_size == 3)
    print()

    print("=== Identical instruction → identical embedding ===")
    fresh_enc = LanguageEncoder()   # new encoder, empty cache
    for inst, original_vec in zip(instructions, vecs):
        v2 = fresh_enc.encode(inst)
        check(f'Identical embedding for "{inst[:30]}"',
              np.allclose(v2, original_vec, atol=1e-6))
    print()

    print("=== Embeddings are distinct across instructions ===")
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sim = float(np.dot(vecs[i], vecs[j]))
            check(
                f'instructions {i} and {j} not identical '
                f'(cosine sim={sim:.3f} < 0.999)',
                sim < 0.999,
            )
    print()

    print(f"Result: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)
