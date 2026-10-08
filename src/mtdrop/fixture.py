from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def synthesize_dropout_wav(
    path: Path,
    *,
    sample_rate: int = 48000,
    duration_s: float = 2.0,
    stereo: bool = True,
    subtype: str = "PCM_24",
    azimuth_lag_samples: float = 3.0,
    level_offset_db_r: float = -2.5,
) -> dict:
    """Generate a short tone with injected dropouts, optional R lag and R level offset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(sample_rate * duration_s)
    t = np.arange(n, dtype=np.float64) / sample_rate
    bed = 0.22 * np.sin(2 * np.pi * 440.0 * t) + 0.12 * np.sin(2 * np.pi * 880.0 * t)
    bed += 0.015 * np.random.default_rng(7).standard_normal(n)
    bed += 0.08 * np.sin(2 * np.pi * 6000.0 * t)

    left = bed.copy()
    right = bed.copy()
    injected: list[dict] = []

    def apply_gain(ch: np.ndarray, start_s: float, end_s: float, gain: float) -> None:
        a = int(start_s * sample_rate)
        b = int(end_s * sample_rate)
        ch[a:b] *= gain

    def apply_hf_cut(ch: np.ndarray, start_s: float, end_s: float) -> None:
        a = int(start_s * sample_rate)
        b = int(end_s * sample_rate)
        seg = ch[a:b].copy()
        kernel = np.ones(9) / 9.0
        filtered = np.convolve(seg, kernel, mode="same")
        ch[a:b] = 0.15 * seg + 0.85 * filtered

    apply_gain(left, 0.40, 0.44, 0.08)
    apply_gain(right, 0.40, 0.44, 0.08)
    injected.append({"type": "level_dip", "start_s": 0.40, "end_s": 0.44, "channel": "both"})

    apply_gain(left, 0.90, 0.925, 0.0)
    apply_gain(right, 0.90, 0.925, 0.0)
    injected.append({"type": "hard_mute", "start_s": 0.90, "end_s": 0.925, "channel": "both"})

    apply_hf_cut(left, 1.30, 1.36)
    apply_hf_cut(right, 1.30, 1.36)
    injected.append({"type": "hf_loss", "start_s": 1.30, "end_s": 1.36, "channel": "both"})

    apply_gain(left, 1.70, 1.75, 0.05)
    injected.append({"type": "level_dip", "start_s": 1.70, "end_s": 1.75, "channel": "L"})

    if stereo:
        # Inject azimuth: delay right by lag_samples (positive => R lags L)
        if abs(azimuth_lag_samples) > 1e-9:
            lag = azimuth_lag_samples
            idx = np.arange(n, dtype=np.float64) - lag
            idx0 = np.floor(idx).astype(np.int64)
            frac = idx - idx0
            idx0c = np.clip(idx0, 0, n - 1)
            idx1c = np.clip(idx0 + 1, 0, n - 1)
            delayed = (1.0 - frac) * right[idx0c] + frac * right[idx1c]
            delayed[idx < 0] = 0.0
            delayed[idx > (n - 1)] = 0.0
            right = delayed
            injected.append({"type": "azimuth_lag", "lag_samples": azimuth_lag_samples})

        if abs(level_offset_db_r) > 1e-9:
            right *= 10.0 ** (level_offset_db_r / 20.0)
            injected.append({"type": "level_offset_r_db", "db": level_offset_db_r})

        audio = np.stack([left, right], axis=1).astype(np.float32)
    else:
        audio = left.astype(np.float32)

    peak = float(np.max(np.abs(audio))) or 1.0
    audio = (audio / peak) * 0.89

    sf.write(str(path), audio, sample_rate, subtype=subtype)
    return {
        "path": str(path),
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "stereo": stereo,
        "subtype": subtype,
        "azimuth_lag_samples": azimuth_lag_samples if stereo else 0.0,
        "level_offset_db_r": level_offset_db_r if stereo else 0.0,
        "injected": injected,
    }
