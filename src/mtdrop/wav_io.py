from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

# Hard rule (Helio): derived WAVs must match input PCM subtype / sr / channels.
_ALLOWED_PCM = {"PCM_16", "PCM_24", "PCM_32", "PCM_U8"}


@dataclass(slots=True)
class WavAudio:
    path: Path
    samples: np.ndarray  # float32, shape (frames,) or (frames, channels)
    sample_rate: int
    channels: int
    bit_depth: int | None
    subtype: str


@dataclass(slots=True)
class WavFormat:
    sample_rate: int
    channels: int
    subtype: str
    bit_depth: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "subtype": self.subtype,
            "bit_depth": self.bit_depth,
        }


def read_wav(path: Path | str) -> WavAudio:
    """Read PCM WAV (mono/stereo). Rejects non-PCM and >192 kHz."""
    path = Path(path)
    info = sf.info(str(path))
    if info.samplerate > 192_000:
        raise ValueError(f"{path}: sample rate {info.samplerate} Hz exceeds 192 kHz limit")
    if info.channels not in (1, 2):
        raise ValueError(f"{path}: expected mono or stereo, got {info.channels} channels")

    subtype = (info.subtype or "").upper()
    if not subtype.startswith("PCM"):
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
        if subtype.startswith("PCM_") and bit_depth > 24:
            raise ValueError(f"{path}: bit depth {bit_depth} exceeds 24-bit v1 support")

    return WavAudio(
        path=path,
        samples=np.asarray(data, dtype=np.float32),
        sample_rate=int(sr),
        channels=channels,
        bit_depth=bit_depth,
        subtype=subtype,
    )


def format_of(wav: WavAudio) -> WavFormat:
    return WavFormat(
        sample_rate=wav.sample_rate,
        channels=wav.channels,
        subtype=wav.subtype.upper(),
        bit_depth=wav.bit_depth,
    )


def probe_format(path: Path | str) -> WavFormat:
    info = sf.info(str(path))
    subtype = (info.subtype or "").upper()
    return WavFormat(
        sample_rate=int(info.samplerate),
        channels=int(info.channels),
        subtype=subtype,
        bit_depth=_subtype_bit_depth(subtype),
    )


def formats_match(a: WavFormat, b: WavFormat) -> bool:
    return (
        a.sample_rate == b.sample_rate
        and a.channels == b.channels
        and a.subtype.upper() == b.subtype.upper()
    )


def write_wav_matching(
    path: Path | str,
    samples: np.ndarray,
    *,
    like: WavAudio | WavFormat,
) -> WavFormat:
    """Write a derived WAV matching ``like`` sample rate, channel count, and PCM subtype.

    Hard rule: never silently change sr / bit depth / channels. Raises if the
    written file does not match the requested format.
    """
    path = Path(path)
    fmt = like if isinstance(like, WavFormat) else format_of(like)
    subtype = fmt.subtype.upper()
    if subtype not in _ALLOWED_PCM:
        raise ValueError(
            f"refusing to write non-PCM or unsupported subtype {subtype!r}; "
            f"allowed={sorted(_ALLOWED_PCM)}"
        )

    x = np.asarray(samples)
    if x.ndim == 1:
        if fmt.channels != 1:
            raise ValueError(f"channel mismatch: data is mono but format requires {fmt.channels}")
    elif x.ndim == 2:
        if x.shape[1] != fmt.channels:
            raise ValueError(f"channel mismatch: data has {x.shape[1]} ch, format requires {fmt.channels}")
    else:
        raise ValueError(f"samples must be 1-D or 2-D, got shape {x.shape}")

    # Clip to legal PCM range before quantize
    x = np.clip(x.astype(np.float64), -1.0, 1.0)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x.astype(np.float32), fmt.sample_rate, subtype=subtype)

    written = probe_format(path)
    if not formats_match(fmt, written):
        raise RuntimeError(
            f"format passthrough failed for {path}: wanted {fmt.to_dict()}, got {written.to_dict()}"
        )
    return written


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
