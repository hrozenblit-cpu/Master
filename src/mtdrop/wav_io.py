from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(slots=True)
class WavAudio:
    path: Path
    samples: np.ndarray  # float32, shape (frames,) or (frames, channels)
    sample_rate: int
    channels: int
    bit_depth: int | None
    subtype: str


def read_wav(path: Path | str) -> WavAudio:
    """Read PCM WAV (mono/stereo). Rejects non-PCM and >192 kHz."""
    path = Path(path)
    info = sf.info(str(path))
    if info.samplerate > 192_000:
        raise ValueError(f"{path}: sample rate {info.samplerate} Hz exceeds 192 kHz limit")
    if info.channels not in (1, 2):
        raise ValueError(f"{path}: expected mono or stereo, got {info.channels} channels")

    subtype = info.subtype or ""
    if not subtype.upper().startswith("PCM"):
        raise ValueError(f"{path}: expected PCM WAV, got subtype {subtype!r}")

    data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]
        if channels not in (1, 2):
            raise ValueError(f"{path}: expected mono or stereo, got {channels} channels")

    bit_depth = _subtype_bit_depth(subtype)
    if bit_depth is not None and bit_depth > 24:
        # Allow 32-bit float/PCM float reads only if subtype was PCM; we already gated PCM.
        # 32-bit PCM integer is uncommon for tape; warn via rejection of >24 for archive policy.
        if "PCM_" in subtype.upper() and bit_depth > 24:
            raise ValueError(f"{path}: bit depth {bit_depth} exceeds 24-bit v1 support")

    return WavAudio(
        path=path,
        samples=np.asarray(data, dtype=np.float32),
        sample_rate=int(sr),
        channels=channels,
        bit_depth=bit_depth,
        subtype=subtype,
    )


def _subtype_bit_depth(subtype: str) -> int | None:
    mapping = {
        "PCM_16": 16,
        "PCM_24": 24,
        "PCM_32": 32,
        "PCM_U8": 8,
        "FLOAT": 32,
        "DOUBLE": 64,
    }
    return mapping.get(subtype.upper())


def channel_matrix(samples: np.ndarray) -> np.ndarray:
    """Return float32 array shaped (frames, channels)."""
    if samples.ndim == 1:
        return samples.reshape(-1, 1)
    return samples
