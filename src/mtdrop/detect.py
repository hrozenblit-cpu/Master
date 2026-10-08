from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mtdrop.models import (
    AnalysisReport,
    ChannelTag,
    DropoutEvent,
    DropoutType,
    StereoRelationship,
    duration_class,
)
from mtdrop.wav_io import WavAudio, channel_matrix


@dataclass(slots=True)
class DetectConfig:
    frame_ms: float = 5.0
    hop_ms: float = 2.5
    baseline_ms: float = 400.0
    # Relative RMS below adaptive baseline (linear ratio). 0.35 ≈ -9 dB.
    dip_ratio: float = 0.35
    # Absolute hard-mute floor (linear full-scale).
    mute_floor: float = 1e-4  # ≈ -80 dBFS
    # HF/LF ratio collapse vs local baseline (relative).
    hf_ratio_drop: float = 0.35
    # Minimum event length to keep (seconds).
    min_duration_s: float = 0.003
    # Merge gaps shorter than this (seconds).
    merge_gap_s: float = 0.008
    # Asymmetry: other channel must stay above this fraction of its baseline.
    asym_other_keep: float = 0.7
    severity_threshold: float = 0.15


def analyze(wav: WavAudio, config: DetectConfig | None = None) -> AnalysisReport:
    cfg = config or DetectConfig()
    x = channel_matrix(wav.samples)
    n_frames, n_ch = x.shape
    sr = wav.sample_rate

    frame = max(1, int(sr * cfg.frame_ms / 1000.0))
    hop = max(1, int(sr * cfg.hop_ms / 1000.0))
    baseline_frames = max(3, int(cfg.baseline_ms / cfg.hop_ms))

    events: list[DropoutEvent] = []
    per_ch_dips: list[list[tuple[int, int, float, DropoutType]]] = []

    for ch in range(n_ch):
        rms = _frame_rms(x[:, ch], frame, hop)
        baseline = _moving_percentile(rms, baseline_frames, q=0.6)
        baseline = np.maximum(baseline, 1e-8)

        dip_mask = rms < (baseline * cfg.dip_ratio)
        mute_mask = rms < cfg.mute_floor

        lf, hf = _band_energies(x[:, ch], sr, frame, hop)
        lf_safe = np.maximum(lf, 1e-10)
        hf_ratio = hf / lf_safe
        hf_base = _moving_percentile(hf_ratio, baseline_frames, q=0.6)
        hf_base = np.maximum(hf_base, 1e-8)
        # HF collapse while LF remains near baseline
        lf_base = _moving_percentile(lf, baseline_frames, q=0.6)
        lf_ok = lf > (np.maximum(lf_base, 1e-10) * 0.5)
        hf_mask = (hf_ratio < hf_base * cfg.hf_ratio_drop) & lf_ok & ~mute_mask

        ch_events: list[tuple[int, int, float, DropoutType]] = []
        ch_events.extend(_mask_to_spans(mute_mask, "hard_mute", severity_from=lambda s, e: 1.0))
        ch_events.extend(
            _mask_to_spans(
                dip_mask & ~mute_mask,
                "level_dip",
                severity_from=lambda s, e, rms=rms, baseline=baseline: _dip_severity(rms[s:e], baseline[s:e]),
            )
        )
        ch_events.extend(
            _mask_to_spans(
                hf_mask,
                "hf_loss",
                severity_from=lambda s, e, hr=hf_ratio, hb=hf_base: _hf_severity(hr[s:e], hb[s:e]),
            )
        )
        ch_events = _merge_spans(ch_events, cfg.merge_gap_s, sr, hop)
        per_ch_dips.append(ch_events)

    if n_ch == 1:
        for start_i, end_i, sev, etype in per_ch_dips[0]:
            ev = _make_event(start_i, end_i, hop, sr, "mono", etype, sev, cfg)
            if ev is not None:
                events.append(ev)
        relationship: StereoRelationship = "mono"
        corr: float | None = None
    else:
        # Never assume L==R: always keep per-channel hits, plus joint/asymmetry overlays.
        events.extend(_stereo_report(per_ch_dips[0], per_ch_dips[1], hop, sr, cfg))
        relationship, corr = _stereo_relationship(x[:, 0], x[:, 1])

    events = [e for e in events if e.severity >= cfg.severity_threshold and e.duration_s >= cfg.min_duration_s]
    events.sort(key=lambda e: (e.start_sample, e.channel, e.type))

    return AnalysisReport(
        source=str(wav.path),
        sample_rate=sr,
        channels=n_ch,
        frames=n_frames,
        bit_depth=wav.bit_depth,
        events=events,
        stereo_relationship=relationship,
        channel_correlation=corr,
    )


def _frame_rms(signal: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if signal.size < frame:
        return np.array([float(np.sqrt(np.mean(signal**2)))], dtype=np.float64)
    n = 1 + (signal.size - frame) // hop
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        sl = signal[i * hop : i * hop + frame]
        out[i] = np.sqrt(np.mean(sl * sl))
    return out


def _moving_percentile(x: np.ndarray, win: int, q: float) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    win = max(1, win)
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(x.size):
        out[i] = float(np.quantile(xp[i : i + win], q))
    return out


def _band_energies(signal: np.ndarray, sr: int, frame: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """Simple STFT-bin energy split: LF < 2 kHz, HF > 4 kHz (clamped to Nyquist)."""
    n_fft = max(256, 1 << int(np.ceil(np.log2(frame))))
    if signal.size < n_fft:
        # Degenerate short signals
        rms = _frame_rms(signal, frame, hop)
        return rms, rms * 0.5

    window = np.hanning(n_fft).astype(np.float64)
    n = 1 + (signal.size - n_fft) // hop
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lf_max = min(2000.0, sr / 2 * 0.45)
    hf_min = min(4000.0, sr / 2 * 0.55)
    lf_bins = freqs <= lf_max
    hf_bins = freqs >= hf_min
    lf = np.empty(n, dtype=np.float64)
    hf = np.empty(n, dtype=np.float64)
    for i in range(n):
        sl = signal[i * hop : i * hop + n_fft].astype(np.float64) * window
        spec = np.fft.rfft(sl)
        power = (spec.real**2 + spec.imag**2)
        lf[i] = np.sqrt(np.mean(power[lf_bins]) + 1e-20)
        hf[i] = np.sqrt(np.mean(power[hf_bins]) + 1e-20)
    # Align length with RMS hop framing used elsewhere
    target = _frame_rms(signal, frame, hop).size
    lf = _resample_frames(lf, target)
    hf = _resample_frames(hf, target)
    return lf, hf


def _resample_frames(x: np.ndarray, target: int) -> np.ndarray:
    if x.size == target:
        return x
    if x.size == 0:
        return np.zeros(target, dtype=np.float64)
    xp = np.linspace(0.0, 1.0, x.size)
    fp = np.linspace(0.0, 1.0, target)
    return np.interp(fp, xp, x)


def _mask_to_spans(
    mask: np.ndarray,
    etype: DropoutType,
    severity_from,
) -> list[tuple[int, int, float, DropoutType]]:
    spans: list[tuple[int, int, float, DropoutType]] = []
    if mask.size == 0:
        return spans
    in_run = False
    start = 0
    for i, flag in enumerate(mask):
        if flag and not in_run:
            in_run = True
            start = i
        elif not flag and in_run:
            in_run = False
            sev = float(severity_from(start, i))
            spans.append((start, i, sev, etype))
    if in_run:
        sev = float(severity_from(start, mask.size))
        spans.append((start, mask.size, sev, etype))
    return spans


def _dip_severity(rms_seg: np.ndarray, base_seg: np.ndarray) -> float:
    if rms_seg.size == 0:
        return 0.0
    ratio = float(np.median(rms_seg / np.maximum(base_seg, 1e-10)))
    # 0 = no dip, 1 = total mute relative to baseline
    return float(np.clip(1.0 - ratio, 0.0, 1.0))


def _hf_severity(ratio_seg: np.ndarray, base_seg: np.ndarray) -> float:
    if ratio_seg.size == 0:
        return 0.0
    r = float(np.median(ratio_seg / np.maximum(base_seg, 1e-10)))
    return float(np.clip(1.0 - r, 0.0, 1.0))


def _merge_spans(
    spans: list[tuple[int, int, float, DropoutType]],
    merge_gap_s: float,
    sr: int,
    hop: int,
) -> list[tuple[int, int, float, DropoutType]]:
    if not spans:
        return []
    gap_frames = max(1, int(merge_gap_s * sr / hop))
    # Merge within same type first
    by_type: dict[DropoutType, list[tuple[int, int, float, DropoutType]]] = {}
    for s in spans:
        by_type.setdefault(s[3], []).append(s)
    merged: list[tuple[int, int, float, DropoutType]] = []
    for etype, group in by_type.items():
        group = sorted(group, key=lambda t: t[0])
        cur_s, cur_e, cur_sev, _ = group[0]
        for s, e, sev, _ in group[1:]:
            if s <= cur_e + gap_frames:
                cur_e = max(cur_e, e)
                cur_sev = max(cur_sev, sev)
            else:
                merged.append((cur_s, cur_e, cur_sev, etype))
                cur_s, cur_e, cur_sev = s, e, sev
        merged.append((cur_s, cur_e, cur_sev, etype))
    return sorted(merged, key=lambda t: t[0])


def _make_event(
    start_i: int,
    end_i: int,
    hop: int,
    sr: int,
    channel: ChannelTag,
    etype: DropoutType,
    severity: float,
    cfg: DetectConfig,
    notes: str = "",
) -> DropoutEvent | None:
    start_sample = int(start_i * hop)
    end_sample = int(end_i * hop)
    start_s = start_sample / sr
    end_s = end_sample / sr
    dur = end_s - start_s
    if dur < cfg.min_duration_s:
        return None
    # Confidence: longer + deeper → higher, capped
    conf = float(np.clip(0.45 + 0.4 * severity + min(dur, 0.2) * 1.5, 0.0, 0.99))
    return DropoutEvent(
        start_s=start_s,
        end_s=end_s,
        start_sample=start_sample,
        end_sample=end_sample,
        channel=channel,
        type=etype,
        severity=float(severity),
        confidence=conf,
        duration_class=duration_class(dur),
        notes=notes,
    )


def _stereo_relationship(left: np.ndarray, right: np.ndarray) -> tuple[StereoRelationship, float]:
    """Correlation hint only — never used to skip a channel or assume shared content."""
    n = min(left.size, right.size)
    if n < 8:
        return "unknown", 0.0
    # Downsample for speed on long files
    step = max(1, n // 200_000)
    a = left[:n:step].astype(np.float64)
    b = right[:n:step].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return "unknown", 0.0
    corr = float(np.dot(a, b) / denom)
    # High correlation → dual-mono-like; lower → true stereo (different L/R content).
    if corr >= 0.92:
        return "dual_mono_like", corr
    return "true_stereo", corr


def _stereo_report(
    left: list[tuple[int, int, float, DropoutType]],
    right: list[tuple[int, int, float, DropoutType]],
    hop: int,
    sr: int,
    cfg: DetectConfig,
) -> list[DropoutEvent]:
    """Per-channel events always; joint + asymmetry overlays when spans relate.

    Detection never assumes L==R (dual-mono or true stereo). Each channel was
    scored against its own baseline; this only labels relationships for review.
    """
    events: list[DropoutEvent] = []

    # 1) Always emit per-channel morphology events.
    for ls, le, lsev, ltype in left:
        ev = _make_event(ls, le, hop, sr, "L", ltype, lsev, cfg, notes="per-channel")
        if ev is not None:
            events.append(ev)
    for rs, re, rsev, rtype in right:
        ev = _make_event(rs, re, hop, sr, "R", rtype, rsev, cfg, notes="per-channel")
        if ev is not None:
            events.append(ev)

    # 2) Joint / asymmetry overlays (same type, temporal overlap).
    used_r: set[int] = set()
    for ls, le, lsev, ltype in left:
        match = None
        for ri, (rs, re, rsev, rtype) in enumerate(right):
            if ri in used_r or rtype != ltype:
                continue
            if rs < le and re > ls:
                match = (ri, rs, re, rsev)
                break
        if match is None:
            asym = _make_event(
                ls,
                le,
                hop,
                sr,
                "L>R",
                "channel_asymmetry",
                lsev,
                cfg,
                notes=f"left-only {ltype}",
            )
            if asym is not None:
                events.append(asym)
            continue

        ri, rs, re, rsev = match
        used_r.add(ri)
        start_i = min(ls, rs)
        end_i = max(le, re)
        sev = max(lsev, rsev)
        if abs(lsev - rsev) < 0.2:
            joint = _make_event(
                start_i,
                end_i,
                hop,
                sr,
                "both",
                ltype,
                sev,
                cfg,
                notes="joint (overlapping L+R; not assuming identical content)",
            )
            if joint is not None:
                events.append(joint)
        else:
            ch: ChannelTag = "L>R" if lsev > rsev else "R>L"
            notes = "stronger on left" if ch == "L>R" else "stronger on right"
            joint = _make_event(start_i, end_i, hop, sr, ch, ltype, sev, cfg, notes=notes)
            if joint is not None:
                events.append(joint)
            asym = _make_event(
                start_i,
                end_i,
                hop,
                sr,
                ch,
                "channel_asymmetry",
                abs(lsev - rsev),
                cfg,
                notes=notes,
            )
            if asym is not None:
                events.append(asym)

    for ri, (rs, re, rsev, rtype) in enumerate(right):
        if ri in used_r:
            continue
        asym = _make_event(
            rs,
            re,
            hop,
            sr,
            "R>L",
            "channel_asymmetry",
            rsev,
            cfg,
            notes=f"right-only {rtype}",
        )
        if asym is not None:
            events.append(asym)

    return events
