from __future__ import annotations

"""Phase C — dropout repair apply (derived WAV only; never overwrites masters).

Conservative strategies:
- cubic / mirror interpolation for short level dips & hard mutes
- STFT magnitude fill for slightly longer gaps
- HF-band reconstruct for tape-clog (hf_loss)
- cross-channel borrow when the other channel is clean (dual-mono or true stereo)

Defaults calibrated on Helio exemplo_1_18 (full-track mono → Studer A80 two-track)
plus the synthetic fixture. Still not production-certified — more reels needed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf

from mtdrop.models import AnalysisReport, DropoutEvent
from mtdrop.wav_io import WavAudio, channel_matrix

RepairMode = Literal["off", "conservative", "preview"]

# Duration caps (seconds). preview is slightly more permissive.
# For dual_mono_like / full-track→two-track, longer asymmetric events can still
# borrow from the clean sibling — see _choose_strategy.
_MAX_DUR = {
    "conservative": {"hard_mute": 0.10, "level_dip": 0.12, "hf_loss": 0.15},
    "preview": {"hard_mute": 0.20, "level_dip": 0.25, "hf_loss": 0.30},
}
_MAX_DUR_BORROW = {
    "conservative": {"hard_mute": 0.25, "level_dip": 0.30, "hf_loss": 0.35},
    "preview": {"hard_mute": 0.40, "level_dip": 0.50, "hf_loss": 0.50},
}


@dataclass(slots=True)
class RepairPlan:
    mode: RepairMode
    events_selected: list[DropoutEvent]
    strategies: list[dict[str, Any]]
    status: str
    message: str
    repaired_count: int = 0
    deferred_count: int = 0
    output_wav: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "message": self.message,
            "event_count": len(self.events_selected),
            "repaired_count": self.repaired_count,
            "deferred_count": self.deferred_count,
            "output_wav": self.output_wav,
            "strategies": self.strategies,
            "events": [e.to_dict() for e in self.events_selected],
            "calibration_note": (
                "Calibrated on synthetic fixture + Helio exemplo_1_18 "
                "(full-track mono digitized as A80 two-track). More reels needed; not production-ready."
            ),
        }


@dataclass(slots=True)
class RepairResult:
    plan: RepairPlan
    samples: np.ndarray
    sample_rate: int
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = self.plan.to_dict()
        d["provenance"] = self.provenance
        return d


def plan_repairs(
    report: AnalysisReport,
    *,
    mode: RepairMode = "conservative",
) -> RepairPlan:
    """Select per-channel morphology events and assign strategies (no audio write)."""
    if mode == "off":
        return RepairPlan(
            mode=mode,
            events_selected=[],
            strategies=[],
            status="skipped",
            message="repair off",
        )

    caps = _MAX_DUR[mode]
    borrow_caps = _MAX_DUR_BORROW[mode]
    dual_mono = report.stereo_relationship == "dual_mono_like"
    selected: list[DropoutEvent] = []
    strategies: list[dict[str, Any]] = []
    deferred = 0

    # Repair only per-channel morphology events (avoid double-hit on joint overlays).
    candidates = [
        e
        for e in report.events
        if e.type in {"level_dip", "hard_mute", "hf_loss"} and e.channel in {"L", "R", "mono"}
    ]
    # Longest-first so we can skip nested overlaps on same channel
    candidates.sort(key=lambda e: (-e.duration_s, e.start_sample, e.channel))
    claimed: list[tuple[str, int, int]] = []

    for ev in candidates:
        if _overlaps_claimed(ev, claimed):
            strategies.append(_strat(ev, "skip_overlap", "planned", reason="overlaps larger event"))
            deferred += 1
            continue

        strat = _choose_strategy(ev, report)
        # Full-track→two-track / dual-mono: allow longer repairs when borrowing from clean sibling
        max_d = borrow_caps.get(ev.type, 0.25) if strat == "cross_channel_borrow" else caps.get(ev.type, 0.08)
        if ev.duration_s > max_d:
            strategies.append(
                _strat(ev, "defer_manual", "deferred", reason=f"duration {ev.duration_s:.4f}s > {max_d}s")
            )
            deferred += 1
            continue

        if dual_mono and strat == "cross_channel_borrow":
            notes_extra = "dual_mono_like prefer borrow"
        else:
            notes_extra = ""
        selected.append(ev)
        s = _strat(ev, strat, "planned")
        if notes_extra:
            s["note"] = notes_extra
        strategies.append(s)
        claimed.append((ev.channel, ev.start_sample, ev.end_sample))

    return RepairPlan(
        mode=mode,
        events_selected=selected,
        strategies=strategies,
        status="planned",
        message=f"{len(selected)} event(s) planned for apply; {deferred} deferred",
        repaired_count=0,
        deferred_count=deferred,
    )


def apply_repairs(
    wav: WavAudio,
    report: AnalysisReport,
    out_wav: Path,
    *,
    mode: RepairMode = "conservative",
    subtype: str | None = None,
) -> RepairResult:
    """Apply planned repairs and write a derived WAV. Never modifies the source path."""
    if mode == "off":
        plan = plan_repairs(report, mode=mode)
        return RepairResult(plan=plan, samples=channel_matrix(wav.samples), sample_rate=wav.sample_rate)

    plan = plan_repairs(report, mode=mode)
    x = channel_matrix(wav.samples).astype(np.float64, copy=True)
    sr = wav.sample_rate
    applied: list[dict[str, Any]] = []

    # Apply shortest-first within planned set for cleaner edge context
    order = sorted(plan.events_selected, key=lambda e: (e.start_sample, e.duration_s))
    strat_by_key = {
        (s.get("start_sample"), s.get("channel"), s.get("type")): s for s in plan.strategies if s.get("status") == "planned"
    }

    for ev in order:
        key = (ev.start_sample, ev.channel, ev.type)
        meta = strat_by_key.get(key) or _strat(ev, _choose_strategy(ev, report), "planned")
        strategy = meta["strategy"]
        ch_i = _channel_index(ev.channel, x.shape[1])
        if ch_i is None:
            meta["status"] = "skipped"
            meta["reason"] = "invalid channel"
            continue

        a, b = int(ev.start_sample), int(ev.end_sample)
        a = max(0, min(a, x.shape[0]))
        b = max(a, min(b, x.shape[0]))
        if b <= a:
            meta["status"] = "skipped"
            meta["reason"] = "empty span"
            continue

        try:
            if strategy == "cross_channel_borrow":
                donor = 1 - ch_i
                _cross_channel_borrow(x, ch_i, donor, a, b)
            elif strategy == "hf_band_reconstruct":
                _hf_band_reconstruct(x[:, ch_i], a, b, sr)
            elif strategy == "bilateral_context":
                _bilateral_context_fill(x[:, ch_i], a, b)
            elif strategy == "stft_interp":
                _stft_interp(x[:, ch_i], a, b)
            else:  # cubic_interp / mirror
                _mirror_interp(x[:, ch_i], a, b)
            meta["status"] = "applied"
            applied.append(
                {
                    "start_s": ev.start_s,
                    "end_s": ev.end_s,
                    "start_sample": a,
                    "end_sample": b,
                    "channel": ev.channel,
                    "type": ev.type,
                    "strategy": strategy,
                }
            )
        except Exception as exc:  # noqa: BLE001
            meta["status"] = "failed"
            meta["reason"] = str(exc)

    # Sync strategy list statuses
    for s in plan.strategies:
        if s.get("status") == "planned":
            # leave as-is if not updated via shared dict — update from applied list
            for ap in applied:
                if (
                    ap["start_sample"] == s.get("start_sample")
                    and ap["channel"] == s.get("channel")
                    and ap["type"] == s.get("type")
                ):
                    s["status"] = "applied"
                    s["strategy"] = ap["strategy"]

    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x *= 0.99 / peak

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    write_subtype = subtype or wav.subtype or "PCM_24"
    if write_subtype.upper() not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT"}:
        write_subtype = "PCM_24"
    sf.write(str(out_wav), x.astype(np.float32), sr, subtype=write_subtype)

    plan.repaired_count = len(applied)
    plan.output_wav = str(out_wav)
    plan.status = "applied"
    plan.message = (
        f"Wrote {out_wav.name}: {plan.repaired_count} repaired, {plan.deferred_count} deferred. "
        "Source master untouched."
    )

    provenance = {
        "source_wav": str(wav.path),
        "output_wav": str(out_wav),
        "mode": mode,
        "stereo_relationship": report.stereo_relationship,
        "repaired_events": applied,
        "policy": "Derived WAV only; masters never overwritten.",
        "calibration_note": (
            "Calibrated on synthetic fixture + Helio exemplo_1_18 "
            "(full-track mono → A80 two-track). More reels needed; not production-ready."
        ),
        "transfer_model": (
            "dual_mono_like prefers cross-channel borrow when donor is clean "
            "(typical of full-track mono digitized as two-track)."
        ),
    }
    return RepairResult(plan=plan, samples=x, sample_rate=sr, provenance=provenance)


def _strat(ev: DropoutEvent, strategy: str, status: str, reason: str = "") -> dict[str, Any]:
    d = {
        "event_start_s": ev.start_s,
        "event_end_s": ev.end_s,
        "start_sample": ev.start_sample,
        "end_sample": ev.end_sample,
        "channel": ev.channel,
        "type": ev.type,
        "strategy": strategy,
        "status": status,
    }
    if reason:
        d["reason"] = reason
    return d


def _overlaps_claimed(ev: DropoutEvent, claimed: list[tuple[str, int, int]]) -> bool:
    for ch, a, b in claimed:
        if ch != ev.channel:
            continue
        if ev.start_sample < b and ev.end_sample > a:
            return True
    return False


def _choose_strategy(ev: DropoutEvent, report: AnalysisReport) -> str:
    dual_mono = report.stereo_relationship == "dual_mono_like"
    # Full-track mono→two-track / dual-mono: prefer cross-channel borrow whenever donor is usable.
    if ev.channel in {"L", "R"} and _donor_usable(ev, report, dual_mono=dual_mono):
        return "cross_channel_borrow"
    if ev.type == "hf_loss":
        return "hf_band_reconstruct"
    # Bilateral damage (common on full-track debris): long context fill beats short STFT.
    if dual_mono and ev.channel in {"L", "R"} and _bilateral_damage(ev, report):
        return "bilateral_context"
    if ev.duration_s >= 0.012:
        return "stft_interp"
    return "cubic_interp"


def _bilateral_damage(ev: DropoutEvent, report: AnalysisReport) -> bool:
    """True when the sibling channel also has an overlapping morphology hit."""
    other = "R" if ev.channel == "L" else "L"
    for e in report.events:
        if e.channel != other:
            continue
        if e.type not in {"level_dip", "hard_mute", "hf_loss"}:
            continue
        if e.start_sample < ev.end_sample and e.end_sample > ev.start_sample:
            return True
    return False


def _donor_usable(ev: DropoutEvent, report: AnalysisReport, *, dual_mono: bool) -> bool:
    """True if the other channel is safe enough to borrow from.

    Strict: no overlapping morphology on donor.
    dual_mono_like (incl. full-track→two-track): also allow if donor overlap is much milder.
    """
    if report.channels < 2 or ev.channel not in {"L", "R"}:
        return False
    other = "R" if ev.channel == "L" else "L"
    overlapping = [
        e
        for e in report.events
        if e.channel == other
        and e.type in {"level_dip", "hard_mute", "hf_loss"}
        and e.start_sample < ev.end_sample
        and e.end_sample > ev.start_sample
    ]
    if not overlapping:
        return True
    if not dual_mono:
        return False
    # Donor usable if every overlapping hit is clearly milder
    return all(e.severity <= ev.severity * 0.55 for e in overlapping)


def _channel_index(tag: str, n_ch: int) -> int | None:
    if tag == "mono" and n_ch >= 1:
        return 0
    if tag == "L" and n_ch >= 1:
        return 0
    if tag == "R" and n_ch >= 2:
        return 1
    return None


def _fade_weights(n: int, fade: int) -> tuple[np.ndarray, np.ndarray]:
    fade = max(1, min(fade, n // 2 if n >= 2 else 1))
    w_new = np.ones(n, dtype=np.float64)
    if n >= 2:
        ramp = np.linspace(0.0, 1.0, fade)
        w_new[:fade] = ramp
        w_new[-fade:] = ramp[::-1]
    return w_new, 1.0 - w_new


def _mirror_interp(ch: np.ndarray, a: int, b: int, ctx: int = 128) -> None:
    """Replace [a:b) with crossfade of mirrored left/right context (cubic_interp family)."""
    n = b - a
    if n <= 0:
        return
    left = ch[max(0, a - ctx) : a]
    right = ch[b : min(ch.size, b + ctx)]
    if left.size == 0 and right.size == 0:
        ch[a:b] = 0.0
        return
    if left.size == 0:
        fill = _tile_to(right[::-1], n)
    elif right.size == 0:
        fill = _tile_to(left[::-1], n)
    else:
        from_l = _tile_to(left[::-1], n)
        from_r = _tile_to(right[::-1], n)
        t = np.linspace(0.0, 1.0, n)
        # Equal-power-ish crossfade
        fill = from_l * np.cos(0.5 * np.pi * t) + from_r * np.sin(0.5 * np.pi * t)
    # Match edge samples exactly
    if a > 0:
        fill = fill - fill[0] + ch[a - 1]
    w_new, w_old = _fade_weights(n, max(8, min(64, n // 4)))
    ch[a:b] = w_new * fill + w_old * ch[a:b]


def _bilateral_context_fill(ch: np.ndarray, a: int, b: int) -> None:
    """Longer-context fill for dual-mono bilateral dropouts (both channels damaged).

    Cross-channel borrow is unavailable; use extended mirror from each side of the
    gap plus a light STFT magnitude blend so short full-track hits are less 'holey'.
    """
    n = b - a
    if n <= 0:
        return
    ctx = max(256, min(2048, n * 4))
    _mirror_interp(ch, a, b, ctx=ctx)
    # Refine with STFT if gap is long enough for a stable transform
    if n >= 64:
        backup = ch[a:b].copy()
        try:
            _stft_interp(ch, a, b, n_fft=min(512, 1 << int(np.ceil(np.log2(max(64, n))))), hop=64)
            # Blend mirror (transient continuity) with STFT (tonal fill)
            t = np.linspace(0.0, 1.0, n)
            # Prefer mirror at edges, STFT in the middle
            w_stft = np.sin(np.pi * t) ** 2
            ch[a:b] = (1.0 - w_stft) * backup + w_stft * ch[a:b]
        except Exception:  # noqa: BLE001
            ch[a:b] = backup


def _tile_to(seg: np.ndarray, n: int) -> np.ndarray:
    if seg.size == 0:
        return np.zeros(n, dtype=np.float64)
    reps = int(np.ceil(n / seg.size))
    return np.tile(seg, reps)[:n].astype(np.float64)


def _cross_channel_borrow(x: np.ndarray, target: int, donor: int, a: int, b: int) -> None:
    n = b - a
    ctx = min(256, a, x.shape[0] - b)
    src = x[a:b, donor].copy()
    # RMS-match donor segment to target pre/post context
    tgt_ctx = np.concatenate([x[max(0, a - ctx) : a, target], x[b : min(x.shape[0], b + ctx), target]])
    don_ctx = np.concatenate([x[max(0, a - ctx) : a, donor], x[b : min(x.shape[0], b + ctx), donor]])
    rms_t = float(np.sqrt(np.mean(tgt_ctx**2) + 1e-20)) if tgt_ctx.size else 1.0
    rms_d = float(np.sqrt(np.mean(don_ctx**2) + 1e-20)) if don_ctx.size else 1.0
    scaled = src * (rms_t / rms_d)
    w_new, w_old = _fade_weights(n, max(8, min(64, n // 4)))
    x[a:b, target] = w_new * scaled + w_old * x[a:b, target]


def _stft_interp(ch: np.ndarray, a: int, b: int, n_fft: int = 512, hop: int = 128) -> None:
    """Interpolate STFT frames across the gap; overlap-add back into ch[a:b]."""
    pad = n_fft * 2
    start = max(0, a - pad)
    end = min(ch.size, b + pad)
    seg = ch[start:end].copy()
    if seg.size < n_fft:
        _mirror_interp(ch, a, b)
        return

    window = np.hanning(n_fft).astype(np.float64)
    n_frames = 1 + (seg.size - n_fft) // hop
    frames = np.stack([seg[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    specs = np.fft.rfft(frames, axis=1)

    # Gap relative to seg
    gap_a = a - start
    gap_b = b - start
    bad = []
    good = []
    for i in range(n_frames):
        fa = i * hop
        fb = fa + n_fft
        if fb <= gap_a or fa >= gap_b:
            good.append(i)
        else:
            bad.append(i)

    if not bad:
        return
    if not good:
        _mirror_interp(ch, a, b)
        return

    mags = np.abs(specs)
    phases = np.angle(specs)
    for i in bad:
        # Nearest good frames left/right
        left = max((g for g in good if g < i), default=None)
        right = min((g for g in good if g > i), default=None)
        if left is None and right is None:
            continue
        if left is None:
            mags[i] = mags[right]
            phases[i] = phases[right]
        elif right is None:
            mags[i] = mags[left]
            phases[i] = phases[left]
        else:
            t = (i - left) / max(1, right - left)
            mags[i] = (1 - t) * mags[left] + t * mags[right]
            # Phase: follow left with progression toward right
            phases[i] = (1 - t) * phases[left] + t * phases[right]

    specs_new = mags * np.exp(1j * phases)
    # iSTFT overlap-add
    out = np.zeros(seg.size, dtype=np.float64)
    norm = np.zeros(seg.size, dtype=np.float64)
    for i in range(n_frames):
        frame = np.fft.irfft(specs_new[i], n=n_fft).real * window
        fa = i * hop
        out[fa : fa + n_fft] += frame
        norm[fa : fa + n_fft] += window**2
    norm = np.maximum(norm, 1e-8)
    recon = out / norm

    # Only write the gap region back (with short fade)
    ga, gb = gap_a, gap_b
    fill = recon[ga:gb]
    n = fill.size
    w_new, w_old = _fade_weights(n, max(8, min(64, n // 4)))
    ch[a:b] = w_new * fill + w_old * ch[a:b]


def _hf_band_reconstruct(ch: np.ndarray, a: int, b: int, sr: int, n_fft: int = 512, hop: int = 128) -> None:
    """Keep LF from the dip region; rebuild HF from neighboring frames."""
    pad = n_fft * 2
    start = max(0, a - pad)
    end = min(ch.size, b + pad)
    seg = ch[start:end].copy()
    if seg.size < n_fft:
        _mirror_interp(ch, a, b)
        return

    window = np.hanning(n_fft).astype(np.float64)
    n_frames = 1 + (seg.size - n_fft) // hop
    frames = np.stack([seg[i * hop : i * hop + n_fft] * window for i in range(n_frames)])
    specs = np.fft.rfft(frames, axis=1)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    lf_max = min(2000.0, sr / 2 * 0.45)
    hf_min = min(4000.0, sr / 2 * 0.55)
    hf_bins = freqs >= hf_min
    lf_bins = freqs <= lf_max
    mid_bins = ~hf_bins & ~lf_bins

    gap_a = a - start
    gap_b = b - start
    bad = []
    good = []
    for i in range(n_frames):
        fa = i * hop
        fb = fa + n_fft
        if fb <= gap_a or fa >= gap_b:
            good.append(i)
        else:
            bad.append(i)

    if not bad or not good:
        _mirror_interp(ch, a, b)
        return

    mags = np.abs(specs)
    phases = np.angle(specs)
    for i in bad:
        left = max((g for g in good if g < i), default=None)
        right = min((g for g in good if g > i), default=None)
        if left is None and right is None:
            continue
        if left is None:
            donor_mag, donor_phase = mags[right], phases[right]
        elif right is None:
            donor_mag, donor_phase = mags[left], phases[left]
        else:
            t = (i - left) / max(1, right - left)
            donor_mag = (1 - t) * mags[left] + t * mags[right]
            donor_phase = (1 - t) * phases[left] + t * phases[right]
        # Keep LF from current (dulled) frame; replace HF (+ blend mid)
        mags[i, hf_bins] = donor_mag[hf_bins]
        phases[i, hf_bins] = donor_phase[hf_bins]
        mags[i, mid_bins] = 0.4 * mags[i, mid_bins] + 0.6 * donor_mag[mid_bins]
        phases[i, mid_bins] = 0.4 * phases[i, mid_bins] + 0.6 * donor_phase[mid_bins]

    specs_new = mags * np.exp(1j * phases)
    out = np.zeros(seg.size, dtype=np.float64)
    norm = np.zeros(seg.size, dtype=np.float64)
    for i in range(n_frames):
        frame = np.fft.irfft(specs_new[i], n=n_fft).real * window
        fa = i * hop
        out[fa : fa + n_fft] += frame
        norm[fa : fa + n_fft] += window**2
    recon = out / np.maximum(norm, 1e-8)
    fill = recon[gap_a:gap_b]
    n = fill.size
    w_new, w_old = _fade_weights(n, max(8, min(64, n // 4)))
    ch[a:b] = w_new * fill + w_old * ch[a:b]
