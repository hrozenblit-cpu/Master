from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from mtdrop.wav_io import WavAudio, channel_matrix, write_wav_matching

CorrectTarget = Literal["azimuth", "level"]


@dataclass(slots=True)
class LevelMeasure:
    rms_l_db: float
    rms_r_db: float
    peak_l_db: float
    peak_r_db: float
    # Positive => left louder than right (L − R in dB).
    lr_rms_diff_db: float
    lr_peak_diff_db: float
    # Gain (linear) to apply to R so RMS matches L (1.0 = no change).
    suggested_gain_r: float
    suggested_gain_l: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AzimuthMeasure:
    # Positive lag_samples => R is late vs L (delay R / advance L to align).
    lag_samples: float
    lag_seconds: float
    lag_microseconds: float
    correlation_peak: float
    method: str = "cross_correlation"
    max_lag_samples: int = 0
    notes: str = (
        "Positive lag_samples means right channel lags left "
        "(typical tape head azimuth skew on stereo transfers)."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AlignmentReport:
    source: str
    sample_rate: int
    channels: int
    azimuth: AzimuthMeasure | None
    level: LevelMeasure | None
    stereo_relationship: str
    channel_correlation: float | None
    applied: dict[str, Any]
    output_wav: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "stereo_relationship": self.stereo_relationship,
            "channel_correlation": self.channel_correlation,
            "azimuth": self.azimuth.to_dict() if self.azimuth else None,
            "level": self.level.to_dict() if self.level else None,
            "applied": self.applied,
            "output_wav": self.output_wav,
            "policy": "Source masters are never overwritten; corrections write derived WAVs only.",
        }


def measure_level(wav: WavAudio) -> LevelMeasure | None:
    x = channel_matrix(wav.samples)
    if x.shape[1] < 2:
        return None
    l, r = x[:, 0], x[:, 1]
    rms_l = float(np.sqrt(np.mean(l * l) + 1e-20))
    rms_r = float(np.sqrt(np.mean(r * r) + 1e-20))
    peak_l = float(np.max(np.abs(l)) + 1e-20)
    peak_r = float(np.max(np.abs(r)) + 1e-20)

    def db(v: float) -> float:
        return 20.0 * np.log10(v)

    # Match both toward mid RMS so neither side clips as hard.
    mid = np.sqrt(rms_l * rms_r)
    gain_l = float(mid / rms_l) if rms_l > 1e-12 else 1.0
    gain_r = float(mid / rms_r) if rms_r > 1e-12 else 1.0
    return LevelMeasure(
        rms_l_db=db(rms_l),
        rms_r_db=db(rms_r),
        peak_l_db=db(peak_l),
        peak_r_db=db(peak_r),
        lr_rms_diff_db=db(rms_l) - db(rms_r),
        lr_peak_diff_db=db(peak_l) - db(peak_r),
        suggested_gain_l=gain_l,
        suggested_gain_r=gain_r,
    )


def measure_azimuth(wav: WavAudio, max_lag_ms: float = 2.0) -> AzimuthMeasure | None:
    """Estimate L/R time offset via normalized cross-correlation.

    Uses a mid-file window for stability on long reels. Sub-sample peak via parabolic fit.
    """
    x = channel_matrix(wav.samples)
    if x.shape[1] < 2:
        return None
    sr = wav.sample_rate
    max_lag = max(1, int(sr * max_lag_ms / 1000.0))
    n = x.shape[0]
    # Analyze up to ~2 s centered in file
    win = min(n, int(sr * 2.0))
    start = max(0, (n - win) // 2)
    left = x[start : start + win, 0].astype(np.float64)
    right = x[start : start + win, 1].astype(np.float64)
    left = left - left.mean()
    right = right - right.mean()

    # FFT-based cross-correlation
    size = 1 << int(np.ceil(np.log2(left.size + right.size - 1)))
    fl = np.fft.rfft(left, size)
    fr = np.fft.rfft(right, size)
    cc = np.fft.irfft(fl * np.conj(fr), size)
    # lags: 0..size/2 positive (R delayed?), rearrange
    cc = np.concatenate([cc[-(max_lag):], cc[: max_lag + 1]])
    lags = np.arange(-max_lag, max_lag + 1)
    # Normalize roughly
    denom = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-20)
    cc = cc / denom
    peak_i = int(np.argmax(cc))
    peak = float(cc[peak_i])
    lag = float(lags[peak_i])
    # Parabolic interpolation for fractional sample
    if 0 < peak_i < cc.size - 1:
        y0, y1, y2 = float(cc[peak_i - 1]), float(cc[peak_i]), float(cc[peak_i + 1])
        denom_p = (y0 - 2 * y1 + y2)
        if abs(denom_p) > 1e-12:
            delta = 0.5 * (y0 - y2) / denom_p
            lag = lag + float(delta)
    # irfft(L * conj(R)) peak lag is opposite our "R lags L" convention — flip sign.
    lag = -lag
    lag_s = lag / sr
    return AzimuthMeasure(
        lag_samples=lag,
        lag_seconds=lag_s,
        lag_microseconds=lag_s * 1e6,
        correlation_peak=peak,
        max_lag_samples=max_lag,
    )


def apply_corrections(
    wav: WavAudio,
    *,
    azimuth: AzimuthMeasure | None,
    level: LevelMeasure | None,
    correct: set[str],
    out_path: Path,
    lag_override: float | None = None,
    subtype: str | None = None,
) -> dict[str, Any]:
    """Write a derived corrected WAV. Never modifies the source path."""
    x = channel_matrix(wav.samples).astype(np.float64, copy=True)
    applied: dict[str, Any] = {"azimuth": None, "level": None}
    if x.shape[1] < 2:
        raise ValueError("azimuth/level correction requires stereo input")

    if "azimuth" in correct and (azimuth is not None or lag_override is not None):
        lag = float(lag_override if lag_override is not None else azimuth.lag_samples)  # type: ignore[union-attr]
        # Positive lag => R late. Split correction: delay L by +lag/2, advance R by lag/2.
        half = lag / 2.0
        x[:, 0] = _delay_channel(x[:, 0], half)
        x[:, 1] = _delay_channel(x[:, 1], -half)
        shift = int(np.ceil(abs(half))) + 1
        if shift > 0 and x.shape[0] > 2 * shift:
            x = x[shift:-shift]
        applied["azimuth"] = {
            "lag_samples_estimated": None if azimuth is None else azimuth.lag_samples,
            "lag_samples_applied": lag,
            "delay_l_samples": half,
            "delay_r_samples": -half,
            "lag_override": lag_override,
        }

    if "level" in correct and level is not None:
        x[:, 0] *= level.suggested_gain_l
        x[:, 1] *= level.suggested_gain_r
        peak = float(np.max(np.abs(x))) or 1.0
        if peak > 0.99:
            x *= 0.99 / peak
        applied["level"] = {
            "gain_l": level.suggested_gain_l,
            "gain_r": level.suggested_gain_r,
            "target": "match_mid_rms",
        }

    out_path = Path(out_path)
    # Hard rule: derived WAV matches input sr / bit depth / channels exactly.
    # Optional subtype override only if caller passes an explicit PCM subtype.
    like = wav
    if subtype is not None:
        from mtdrop.wav_io import WavFormat, _subtype_bit_depth

        like = WavFormat(
            sample_rate=wav.sample_rate,
            channels=wav.channels,
            subtype=subtype.upper(),
            bit_depth=_subtype_bit_depth(subtype.upper()),
        )
    written = write_wav_matching(out_path, x, like=like)
    applied["output_wav"] = str(out_path)
    applied["source_wav"] = str(wav.path)
    applied["frames_out"] = int(x.shape[0])
    applied["format"] = written.to_dict()
    return applied


def _delay_channel(sig: np.ndarray, delay_samples: float) -> np.ndarray:
    """Fractional delay via linear interpolation. Positive delay => push samples later."""
    if abs(delay_samples) < 1e-9:
        return sig
    n = sig.size
    idx = np.arange(n, dtype=np.float64) - delay_samples
    idx0 = np.floor(idx).astype(np.int64)
    frac = idx - idx0
    idx0c = np.clip(idx0, 0, n - 1)
    idx1c = np.clip(idx0 + 1, 0, n - 1)
    out = (1.0 - frac) * sig[idx0c] + frac * sig[idx1c]
    # Zero out regions that read past edges
    out[idx < 0] = 0.0
    out[idx > (n - 1)] = 0.0
    return out
