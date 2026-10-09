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
    # Relative RMS below adaptive baseline (linear ratio). 0.34 ≈ -9.4 dB.
    # Balanced after exemplo_1_18 (FP flood at 0.35 HF) and pior.wav (FN on medium dips at 0.30).
    dip_ratio: float = 0.34
    # Absolute hard-mute floor (linear full-scale).
    mute_floor: float = 1e-4  # ≈ -80 dBFS
    # HF collapse vs local baseline (relative). Calibrated: 0.35 was far too sensitive
    # on real tape (musical HF variation → mass FP); 0.22 keeps strong clog events.
    hf_ratio_drop: float = 0.22
    # Also require absolute HF energy collapse (not only HF/LF ratio).
    hf_abs_drop: float = 0.30
    # Minimum event length to keep (seconds).
    min_duration_s: float = 0.004
    # Merge gaps shorter than this (seconds).
    merge_gap_s: float = 0.012
    # Asymmetry: other channel must stay above this fraction of its baseline.
    asym_other_keep: float = 0.7
    severity_threshold: float = 0.22
    # Correlation floor for dual_mono_like hint (full-track mono→two-track often ~0.90–0.98).
    dual_mono_corr: float = 0.90
    # Source impulse clicks/ticks/pops (not tape dropouts) — sample-accurate.
    # Tuned on Helio Samba ``15022_02_QG_do_Samba_OK.wav`` (~0:25 and similar).
    detect_impulse_clicks: bool = True
    # |sample − linear neighbors| must exceed this × local MAD.
    impulse_mad_k: float = 6.5
    # Absolute residual floor (full-scale). 0.005 catches Samba ~0:25 after azimuth
    # (corr softens residual below the old 0.007 floor).
    impulse_abs_floor: float = 0.005
    # Half-width of repair pad around peak (seconds). Wider pad keeps the peak
    # away from splice edges so declick doesn't leave edge ticks.
    impulse_pad_s: float = 0.0016
    # Max merged click length (seconds); longer → not a tick.
    impulse_max_dur_s: float = 0.008
    # Min severity to keep an impulse_click event.
    impulse_severity_threshold: float = 0.45
    # Hard cap per file (time-stratified keep) — avoids over-splicing bright music.
    impulse_max_events: int = 220
    # Bilateral low-mid "tok"/thump (soft knock) — not a sharp HF tick.
    # Calibrated on Helio Samba ``15022_02_QG_do_Samba_OK.wav`` ~24.438 s (survived spike declick).
    detect_bilateral_toks: bool = True
    tok_lo_hz: float = 120.0
    tok_hi_hz: float = 2600.0
    # Mid-band envelope jump vs ±40 ms context (excluding ±8 ms).
    tok_min_jump: float = 2.80
    # min(L_jump, R_jump) — both channels must knock.
    tok_min_bilat: float = 2.65
    tok_min_env: float = 0.016
    tok_nms_s: float = 0.025
    tok_min_w50_s: float = 0.0020
    tok_max_w50_s: float = 0.012
    tok_min_lr_corr: float = 0.90
    # Mid-band jump should exceed HF jump (tok/thump ≠ bright tick / consonant).
    tok_min_mid_hf_ratio: float = 1.35
    # Half-pad around env peak for event span (seconds); repair widens modestly.
    tok_pad_s: float = 0.0035
    tok_severity_threshold: float = 0.50
    # Count is L+R pairs×2; keep headroom so late Samba ~0:25 isn't starved.
    tok_max_events: int = 120


SENSITIVITY_PRESETS: dict[str, dict[str, float]] = {
    # Fewer FPs; may miss medium dips (seen on pior.wav with older 0.30 default).
    "conservative": {
        "dip_ratio": 0.30,
        "hf_ratio_drop": 0.20,
        "hf_abs_drop": 0.28,
        "severity_threshold": 0.25,
        "min_duration_s": 0.005,
    },
    # Default — compromise across exemplo_1_18 / 44.1_1_18 / pior.
    "balanced": {
        "dip_ratio": 0.34,
        "hf_ratio_drop": 0.22,
        "hf_abs_drop": 0.30,
        "severity_threshold": 0.22,
        "min_duration_s": 0.004,
    },
    # Hunt mild/partial dropouts on bad reels; expect more FPs — review markers.
    "aggressive": {
        "dip_ratio": 0.42,
        "hf_ratio_drop": 0.28,
        "hf_abs_drop": 0.40,
        "severity_threshold": 0.35,
        "min_duration_s": 0.005,
    },
}


def config_for_sensitivity(name: str) -> DetectConfig:
    if name not in SENSITIVITY_PRESETS:
        raise ValueError(f"unknown sensitivity {name!r}; choose from {sorted(SENSITIVITY_PRESETS)}")
    return DetectConfig(**SENSITIVITY_PRESETS[name])


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
        hf_abs_base = _moving_percentile(hf, baseline_frames, q=0.6)
        hf_abs_base = np.maximum(hf_abs_base, 1e-10)
        # HF collapse while LF remains near baseline — require BOTH ratio and abs HF drop
        # (ratio-only FPs when LF swells or musical HF dips without clog).
        lf_base = _moving_percentile(lf, baseline_frames, q=0.6)
        lf_ok = lf > (np.maximum(lf_base, 1e-10) * 0.5)
        hf_mask = (
            (hf_ratio < hf_base * cfg.hf_ratio_drop)
            & (hf < hf_abs_base * cfg.hf_abs_drop)
            & lf_ok
            & ~mute_mask
            & ~dip_mask
        )

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
        relationship, corr = _stereo_relationship(x[:, 0], x[:, 1], cfg.dual_mono_corr)

    if cfg.detect_impulse_clicks:
        events.extend(_detect_impulse_clicks(x, sr, cfg))
    if cfg.detect_bilateral_toks and n_ch >= 2:
        events.extend(_detect_bilateral_toks(x, sr, cfg))

    kept: list[DropoutEvent] = []
    impulses: list[DropoutEvent] = []
    toks: list[DropoutEvent] = []
    for e in events:
        if e.type == "impulse_click":
            if e.severity >= cfg.impulse_severity_threshold:
                impulses.append(e)
            continue
        if e.type == "bilateral_tok":
            if e.severity >= cfg.tok_severity_threshold:
                toks.append(e)
            continue
        if e.severity >= cfg.severity_threshold and e.duration_s >= cfg.min_duration_s:
            kept.append(e)
    if len(impulses) > cfg.impulse_max_events:
        # Time-stratified keep (1 s buckets): strongest local ticks first, then global fill.
        # Avoids dropping real ticks near 0:25 when earlier music has many bright peaks.
        bucket_s = 1.0
        by_bucket: dict[int, list[DropoutEvent]] = {}
        for e in impulses:
            by_bucket.setdefault(int(e.start_s // bucket_s), []).append(e)
        chosen: list[DropoutEvent] = []
        # ~2–3 per second keeps coverage across the timeline
        per_bucket = max(2, cfg.impulse_max_events // max(1, int(len(by_bucket) * 1.2)))
        for _b, group in sorted(by_bucket.items()):
            group.sort(key=lambda e: -e.severity)
            chosen.extend(group[:per_bucket])
        if len(chosen) < cfg.impulse_max_events:
            rest = sorted(impulses, key=lambda e: -e.severity)
            seen = {(e.start_sample, e.channel) for e in chosen}
            for e in rest:
                key = (e.start_sample, e.channel)
                if key in seen:
                    continue
                chosen.append(e)
                seen.add(key)
                if len(chosen) >= cfg.impulse_max_events:
                    break
        else:
            # Too many from buckets — trim by severity but keep ≥1 per occupied second
            chosen.sort(key=lambda e: -e.severity)
            keep: list[DropoutEvent] = []
            seen_b: set[int] = set()
            for e in chosen:
                b = int(e.start_s // bucket_s)
                if b not in seen_b:
                    keep.append(e)
                    seen_b.add(b)
            for e in chosen:
                if len(keep) >= cfg.impulse_max_events:
                    break
                if e in keep:
                    continue
                keep.append(e)
            chosen = keep[: cfg.impulse_max_events]
        impulses = chosen[: cfg.impulse_max_events]
    if len(toks) > cfg.tok_max_events:
        # Group L+R twins by start_sample; rank groups by severity; fair across time.
        # Truncating a flat time-ordered list starved Samba ~0:25 after early music toks.
        by_start: dict[int, list[DropoutEvent]] = {}
        for e in toks:
            by_start.setdefault(e.start_sample, []).append(e)
        groups = sorted(by_start.values(), key=lambda g: -max(e.severity for e in g))
        always_sev = 0.55
        strong_g = [g for g in groups if max(e.severity for e in g) >= always_sev]
        mild_g = [g for g in groups if max(e.severity for e in g) < always_sev]
        # Time-stratify mild groups (1.5 s slots); always keep strong groups.
        slot_s = 1.5
        slots: dict[int, list[list[DropoutEvent]]] = {}
        for g in mild_g:
            slots.setdefault(int(g[0].start_s // slot_s), []).append(g)
        chosen_g = list(strong_g)
        per_slot = 2
        for _s, gs in sorted(slots.items()):
            gs.sort(key=lambda g: -max(e.severity for e in g))
            chosen_g.extend(gs[:per_slot])
        # If still over budget, drop mildest strong first but keep ≥1 group per occupied 1.5 s of strong.
        flat = [e for g in chosen_g for e in g]
        if len(flat) > cfg.tok_max_events:
            # Keep strongest groups until cap (pairs stay together).
            chosen_g.sort(key=lambda g: -max(e.severity for e in g))
            tok_kept: list[DropoutEvent] = []
            for g in chosen_g:
                if len(tok_kept) + len(g) > cfg.tok_max_events:
                    break
                tok_kept.extend(g)
            toks = tok_kept
        else:
            toks = flat
    events = kept + impulses + toks
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


def _bandpass_fft(sig: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Zero-phase FFT bandpass (numpy only; no scipy dep). Reflect-pad to ease edges."""
    n = int(sig.size)
    if n < 16:
        return sig.astype(np.float64, copy=True)
    pad = min(max(8, n // 8), max(8, sr // 2))
    # reflect without duplicating endpoints
    left = sig[1 : pad + 1][::-1] if pad < n else sig[::-1]
    right = sig[-(pad + 1) : -1][::-1] if pad < n else sig[::-1]
    if left.size < pad:
        left = np.pad(left, (pad - left.size, 0), mode="edge")
    if right.size < pad:
        right = np.pad(right, (0, pad - right.size), mode="edge")
    xp = np.concatenate([left[:pad], sig, right[:pad]]).astype(np.float64, copy=False)
    spec = np.fft.rfft(xp)
    freqs = np.fft.rfftfreq(xp.size, d=1.0 / sr)
    tw = 40.0
    w = np.zeros_like(freqs)
    mid = (freqs >= (lo + tw)) & (freqs <= (hi - tw))
    w[mid] = 1.0
    lo_ramp = (freqs > (lo - tw)) & (freqs < (lo + tw))
    hi_ramp = (freqs > (hi - tw)) & (freqs < (hi + tw))
    if np.any(lo_ramp):
        w[lo_ramp] = 0.5 * (1.0 - np.cos(np.pi * (freqs[lo_ramp] - (lo - tw)) / (2.0 * tw)))
    if np.any(hi_ramp):
        w[hi_ramp] = 0.5 * (1.0 + np.cos(np.pi * (freqs[hi_ramp] - (hi - tw)) / (2.0 * tw)))
    y = np.fft.irfft(spec * w, n=xp.size)
    return y[pad : pad + n]


def _smooth_abs_env(sig: np.ndarray, sr: int, win_ms: float = 1.5) -> np.ndarray:
    w = max(3, int(round(win_ms * 0.001 * sr)))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float64) / float(w)
    return np.convolve(np.abs(sig), kernel, mode="same")


def _detect_bilateral_toks(x: np.ndarray, sr: int, cfg: DetectConfig) -> list[DropoutEvent]:
    """Find short bilateral low-mid knocks ("tok"/thump), not sharp HF ticks.

    Samba ~0:25 survived impulse residual declick: ~4–7 ms body, centroid ~2 kHz,
    L≈R, band-env jump ≫ local median. Repair uses a modestly wider Hermite bridge.
    """
    if x.ndim != 2 or x.shape[1] < 2 or x.shape[0] < int(0.1 * sr):
        return []

    mid = 0.5 * (x[:, 0] + x[:, 1])
    bp_m = _bandpass_fft(mid, sr, cfg.tok_lo_hz, cfg.tok_hi_hz)
    bp_l = _bandpass_fft(x[:, 0], sr, cfg.tok_lo_hz, cfg.tok_hi_hz)
    bp_r = _bandpass_fft(x[:, 1], sr, cfg.tok_lo_hz, cfg.tok_hi_hz)
    env_m = _smooth_abs_env(bp_m, sr)
    env_l = _smooth_abs_env(bp_l, sr)
    env_r = _smooth_abs_env(bp_r, sr)
    hf_lo = min(4000.0, sr * 0.45)
    hf_hi = min(12000.0, sr * 0.49)
    env_hf = (
        _smooth_abs_env(_bandpass_fft(mid, sr, hf_lo, hf_hi), sr)
        if cfg.tok_min_mid_hf_ratio > 0 and hf_hi > hf_lo + 100
        else None
    )

    half = max(32, int(round(0.040 * sr)))
    excl = max(8, int(round(0.008 * sr)))
    hop = max(1, int(round(0.0005 * sr)))
    refine_r = max(hop, int(round(0.0015 * sr)))
    n = int(env_m.size)
    cands: list[tuple[float, int, float, float, float]] = []
    for i0 in range(half, n - half, hop):
        if env_m[i0] < cfg.tok_min_env * 0.85:
            continue
        # Refine to local envelope peak — hop grid alone misses Samba ~24.587.
        lo = max(half, i0 - refine_r)
        hi = min(n - half, i0 + refine_r + 1)
        i = lo + int(np.argmax(env_m[lo:hi]))
        if env_m[i] < cfg.tok_min_env:
            continue
        ctx = np.concatenate([env_m[i - half : i - excl], env_m[i + excl : i + half]])
        ctx_l = np.concatenate([env_l[i - half : i - excl], env_l[i + excl : i + half]])
        ctx_r = np.concatenate([env_r[i - half : i - excl], env_r[i + excl : i + half]])
        if ctx.size < 8:
            continue
        jump = float(env_m[i] / (float(np.median(ctx)) + 1e-12))
        if jump < cfg.tok_min_jump:
            continue
        jl = float(env_l[i] / (float(np.median(ctx_l)) + 1e-12))
        jr = float(env_r[i] / (float(np.median(ctx_r)) + 1e-12))
        bilat = min(jl, jr)
        if bilat < cfg.tok_min_bilat:
            continue
        if env_hf is not None:
            ctx_hf = np.concatenate([env_hf[i - half : i - excl], env_hf[i + excl : i + half]])
            hf_jump = float(env_hf[i] / (float(np.median(ctx_hf)) + 1e-12))
            if jump / (hf_jump + 1e-12) < cfg.tok_min_mid_hf_ratio:
                continue
        # w50 of mid envelope peak
        thr = 0.5 * float(env_m[i])
        left_i = i
        right_i = i
        lim = int(round(0.012 * sr))
        while left_i > i - lim and env_m[left_i] > thr:
            left_i -= 1
        while right_i < i + lim and env_m[right_i] > thr:
            right_i += 1
        w50 = (right_i - left_i) / float(sr)
        if w50 < cfg.tok_min_w50_s or w50 > cfg.tok_max_w50_s:
            continue
        w = max(8, int(round(0.006 * sr)))
        a = max(0, i - w)
        b = min(n, i + w)
        seg_l = x[a:b, 0].astype(np.float64, copy=False)
        seg_r = x[a:b, 1].astype(np.float64, copy=False)
        if seg_l.size < 8:
            continue
        seg_l = seg_l - float(np.mean(seg_l))
        seg_r = seg_r - float(np.mean(seg_r))
        denom = float(np.linalg.norm(seg_l) * np.linalg.norm(seg_r)) + 1e-12
        corr = float(np.dot(seg_l, seg_r) / denom)
        if corr < cfg.tok_min_lr_corr:
            continue
        score = jump * bilat * float(env_m[i])
        cands.append((score, i, jump, bilat, w50))

    if not cands:
        return []
    cands.sort(key=lambda t: -t[0])
    nms = max(hop, int(round(cfg.tok_nms_s * sr)))
    chosen: list[tuple[float, int, float, float, float]] = []
    for row in cands:
        if any(abs(row[1] - c[1]) < nms for c in chosen):
            continue
        chosen.append(row)
        if len(chosen) >= 200:
            break
    chosen.sort(key=lambda t: t[1])

    out: list[DropoutEvent] = []
    pad = max(4, int(round(cfg.tok_pad_s * sr)))
    for _score, i, jump, bilat, w50 in chosen:
        # Span covers tok body (+ pad); repair uses modest Hermite over this window.
        half_body = max(pad, int(round(0.5 * w50 * sr)) + int(round(0.001 * sr)))
        a = max(0, i - half_body)
        b = min(n, i + half_body + 1)
        sev = float(np.clip((jump - 1.0) / 3.5, 0.0, 1.0))
        conf = float(np.clip(0.55 + 0.35 * sev + 0.05 * min(bilat, 4.0), 0.0, 0.99))
        note = (
            f"bilateral tok/thump (low-mid); jump={jump:.2f} bilat={bilat:.2f} "
            f"w50={w50*1000:.1f}ms; not a sharp HF tick"
        )
        for tag in ("L", "R"):
            out.append(
                DropoutEvent(
                    start_s=a / sr,
                    end_s=b / sr,
                    start_sample=a,
                    end_sample=b,
                    channel=tag,  # type: ignore[arg-type]
                    type="bilateral_tok",
                    severity=sev,
                    confidence=conf,
                    duration_class=duration_class((b - a) / sr),
                    notes=note,
                )
            )
    return out


def _detect_impulse_clicks(x: np.ndarray, sr: int, cfg: DetectConfig) -> list[DropoutEvent]:
    """Find source ticks/pops via residual vs local linear prediction (neighbors).

    Calibrated on Helio ``15022_02_QG_do_Samba_OK.wav`` (~0:25 and similar).
    These are *in the transfer*, not invented by repair — Master Tool should remove them.

    On stereo, also scan the mid mix so shared ticks aren't missed after azimuth, and
    mirror hits onto both L and R so dual-mono repairs stay balance-linked.
    """
    n_ch = x.shape[1]
    out: list[DropoutEvent] = []
    for ch_i in range(n_ch):
        tag: ChannelTag = "mono" if n_ch == 1 else ("L" if ch_i == 0 else "R")
        out.extend(_impulse_clicks_channel(x[:, ch_i], sr, cfg, tag))
    if n_ch >= 2:
        mid = 0.5 * (x[:, 0] + x[:, 1])
        mid_hits = _impulse_clicks_channel(mid, sr, cfg, "L")  # channel tag rewritten below
        # Index existing by approx start (2 ms bins) per channel
        have = {(e.channel, int(round(e.start_s * 500.0))) for e in out}
        for hit in mid_hits:
            for tag in ("L", "R"):
                key = (tag, int(round(hit.start_s * 500.0)))
                if key in have:
                    continue
                out.append(
                    DropoutEvent(
                        start_s=hit.start_s,
                        end_s=hit.end_s,
                        start_sample=hit.start_sample,
                        end_sample=hit.end_sample,
                        channel=tag,  # type: ignore[arg-type]
                        type="impulse_click",
                        severity=hit.severity,
                        confidence=hit.confidence,
                        duration_class=hit.duration_class,
                        notes="source tick/pop (mid-mix); linked L/R for dual-mono balance",
                    )
                )
                have.add(key)
        # Mirror any remaining single-channel hit onto the sibling within 2 ms
        by_ch: dict[str, list[DropoutEvent]] = {"L": [], "R": []}
        for e in out:
            if e.channel in by_ch:
                by_ch[e.channel].append(e)
        for src_ch, dst_ch in (("L", "R"), ("R", "L")):
            for e in by_ch[src_ch]:
                key = (dst_ch, int(round(e.start_s * 500.0)))
                if key in have:
                    continue
                # only mirror if sibling has no near hit already (±4 ms)
                if any(abs(o.start_s - e.start_s) < 0.004 for o in by_ch[dst_ch]):
                    continue
                out.append(
                    DropoutEvent(
                        start_s=e.start_s,
                        end_s=e.end_s,
                        start_sample=e.start_sample,
                        end_sample=e.end_sample,
                        channel=dst_ch,  # type: ignore[arg-type]
                        type="impulse_click",
                        severity=e.severity,
                        confidence=e.confidence,
                        duration_class=e.duration_class,
                        notes="linked sibling tick for dual-mono L/R balance",
                    )
                )
                have.add(key)
    return out


def _impulse_clicks_channel(
    ch: np.ndarray, sr: int, cfg: DetectConfig, channel: ChannelTag
) -> list[DropoutEvent]:
    n = int(ch.size)
    if n < 32:
        return []
    # Residual vs linear interp from ±1 sample (= |2nd difference|/2)
    err = np.abs(ch - 0.5 * (np.roll(ch, 1) + np.roll(ch, -1)))
    err[0] = 0.0
    err[-1] = 0.0
    # Local MAD via median of |err| in ~40 ms (stride for speed)
    win = max(64, int(0.04 * sr))
    hop = max(16, win // 4)
    local_mad = np.zeros(n, dtype=np.float64)
    for i in range(0, n, hop):
        a = max(0, i - win // 2)
        b = min(n, a + win)
        a = max(0, b - win)
        med = float(np.median(err[a:b]))
        mad = float(np.median(np.abs(err[a:b] - med))) + 1e-12
        local_mad[i : min(n, i + hop)] = mad
    # fill any trailing zeros
    if local_mad[-1] == 0:
        local_mad[local_mad == 0] = float(np.median(local_mad[local_mad > 0])) if np.any(local_mad > 0) else 1e-6

    thr = np.maximum(cfg.impulse_mad_k * local_mad, cfg.impulse_abs_floor)
    peaks = np.where(err > thr)[0]
    if peaks.size == 0:
        return []

    pad = max(2, int(round(cfg.impulse_pad_s * sr)))
    min_sep = max(pad, int(0.005 * sr))
    # Rank by residual, NMS
    order = peaks[np.argsort(-err[peaks])]
    chosen: list[int] = []
    for i in order:
        if any(abs(i - j) < min_sep for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) >= 400:
            break
    chosen.sort()

    events: list[DropoutEvent] = []
    # Merge peaks closer than 2*pad into one span
    if not chosen:
        return events
    spans: list[tuple[int, int, float]] = []
    cur_a = max(0, chosen[0] - pad)
    cur_b = min(n, chosen[0] + pad + 1)
    cur_sev = _impulse_severity(float(err[chosen[0]]), float(local_mad[chosen[0]]), cfg)
    for i in chosen[1:]:
        a = max(0, i - pad)
        b = min(n, i + pad + 1)
        sev = _impulse_severity(float(err[i]), float(local_mad[i]), cfg)
        if a <= cur_b + pad:
            cur_b = max(cur_b, b)
            cur_sev = max(cur_sev, sev)
        else:
            spans.append((cur_a, cur_b, cur_sev))
            cur_a, cur_b, cur_sev = a, b, sev
    spans.append((cur_a, cur_b, cur_sev))

    max_len = int(round(cfg.impulse_max_dur_s * sr))
    for a, b, sev in spans:
        if b - a > max_len:
            # Too long for a tick — keep center max_len
            mid = (a + b) // 2
            a = max(0, mid - max_len // 2)
            b = min(n, a + max_len)
        if b - a < 3:
            continue
        events.append(
            DropoutEvent(
                start_s=a / sr,
                end_s=b / sr,
                start_sample=a,
                end_sample=b,
                channel=channel,
                type="impulse_click",
                severity=float(sev),
                confidence=float(np.clip(0.55 + 0.4 * sev, 0.0, 0.99)),
                duration_class=duration_class((b - a) / sr),
                notes="source tick/pop (impulse); not a tape dropout",
            )
        )
    return events


def _impulse_severity(residual: float, mad: float, cfg: DetectConfig) -> float:
    # Map excess over threshold into 0..1
    thr = max(cfg.impulse_mad_k * mad, cfg.impulse_abs_floor)
    if residual <= thr:
        return 0.0
    # 1× over thr → ~0.5; 3× → ~1.0
    return float(np.clip((residual / thr - 1.0) / 2.0 + 0.5, 0.0, 1.0))


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


def _stereo_relationship(
    left: np.ndarray,
    right: np.ndarray,
    dual_mono_corr: float = 0.90,
) -> tuple[StereoRelationship, float]:
    """Correlation hint only — never used to skip a channel or assume shared content.

    dual_mono_like covers: intentional dual-mono masters AND full-track mono tape
    digitized as two-track (Studer A80), where L≈R program and differences are mostly
    azimuth, channel gain, and dropout asymmetry — not stereo imaging.

    Uses the 90th percentile of short window correlations so damaged sections on a
    longer full-track→two-track reel do not drag a dual-mono transfer into true_stereo
    (seen on Helio 44.1_1_18: early windows ~0.94, later damage pulls the mean down).
    """
    n = min(left.size, right.size)
    if n < 8:
        return "unknown", 0.0

    # Adaptive window: aim for ~8–24 windows across the file.
    win = max(2048, n // 12)
    hop = max(1024, win // 2)
    corrs: list[float] = []
    for start in range(0, max(1, n - win + 1), hop):
        a = left[start : start + win].astype(np.float64)
        b = right[start : start + win].astype(np.float64)
        # Skip near-silent windows
        if float(np.sqrt(np.mean(a * a))) < 1e-5 or float(np.sqrt(np.mean(b * b))) < 1e-5:
            continue
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-12:
            continue
        corrs.append(float(np.dot(a, b) / denom))

    if not corrs:
        return "unknown", 0.0
    corr = float(np.quantile(corrs, 0.90))
    if corr >= dual_mono_corr:
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
