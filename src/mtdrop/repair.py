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

from mtdrop.models import AnalysisReport, DropoutEvent
from mtdrop.wav_io import WavAudio, WavFormat, channel_matrix, write_wav_matching, _subtype_bit_depth

# conservative = vocal-safe default (Helio: don't chew lyrics / "comeu as palavras")
# aggressive = more invasive fills (alias: preview, kept for CLI back-compat)
RepairMode = Literal["off", "conservative", "aggressive", "preview"]


def _norm_mode(mode: RepairMode) -> RepairMode:
    if mode == "preview":
        return "aggressive"
    return mode


# Duration caps (seconds). Conservative keeps replacements short so consonants survive.
# For dual_mono_like / full-track→two-track, longer asymmetric events can still
# borrow from the clean sibling — see _choose_strategy.
_MAX_DUR = {
    "conservative": {"hard_mute": 0.080, "level_dip": 0.055, "hf_loss": 0.035, "impulse_click": 0.004},
    "aggressive": {"hard_mute": 0.20, "level_dip": 0.25, "hf_loss": 0.30, "impulse_click": 0.012},
}
_MAX_DUR_BORROW = {
    "conservative": {"hard_mute": 0.15, "level_dip": 0.12, "hf_loss": 0.12},
    "aggressive": {"hard_mute": 0.40, "level_dip": 0.50, "hf_loss": 0.50},
}
# Floor for *apply* (detection may still list milder markers).
# Conservative is intentionally high — prefer markers over inventing syllables.
_MIN_SEVERITY = {
    # level_dip 0.72 keeps pior.wav real holes; apply-time content gate blocks voiced FPs
    "conservative": {"hard_mute": 0.55, "level_dip": 0.72, "hf_loss": 0.92, "impulse_click": 0.60},
    "aggressive": {"hard_mute": 0.40, "level_dip": 0.60, "hf_loss": 0.82, "impulse_click": 0.48},
}
# Skip micro repairs that only create edge clicks (seconds).
# impulse_click is intentionally sub-ms…few-ms — do not treat as skip_micro.
_MIN_APPLY_DUR = {
    "conservative": {"hard_mute": 0.003, "level_dip": 0.008, "hf_loss": 0.025, "impulse_click": 0.00015},
    "aggressive": {"hard_mute": 0.003, "level_dip": 0.005, "hf_loss": 0.015, "impulse_click": 0.0001},
}
# Merge same-channel planned spans closer than this (seconds).
_MERGE_GAP_S = {"conservative": 0.040, "aggressive": 0.020}
# Cap unique dropout time-spans per second of audio (keep highest severity).
_MAX_SPANS_PER_S = {"conservative": 0.6, "aggressive": 2.0}
# Max fraction of fill vs original for non-mute strategies (vocal safety).
_MAX_NEW_BLEND = {"conservative": 0.45, "aggressive": 0.95}


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

    mode = _norm_mode(mode)
    caps = _MAX_DUR[mode]
    borrow_caps = _MAX_DUR_BORROW[mode]
    sev_floor = _MIN_SEVERITY[mode]
    dur_floor = _MIN_APPLY_DUR[mode]
    merge_gap = _MERGE_GAP_S[mode]
    dual_mono = report.stereo_relationship == "dual_mono_like"
    selected: list[DropoutEvent] = []
    strategies: list[dict[str, Any]] = []
    deferred = 0

    # Per-channel morphology + source impulse clicks (ticks/pops in the transfer).
    candidates = [
        e
        for e in report.events
        if e.type in {"level_dip", "hard_mute", "hf_loss", "impulse_click"}
        and e.channel in {"L", "R", "mono"}
    ]
    # Strongest / longest first so weak dense HF hits yield to real dropouts.
    candidates.sort(key=lambda e: (-e.severity, -e.duration_s, e.start_sample, e.channel))
    claimed: list[tuple[str, int, int]] = []

    for ev in candidates:
        min_sev = sev_floor.get(ev.type, 0.7)
        if ev.severity < min_sev:
            strategies.append(
                _strat(ev, "skip_mild", "deferred", reason=f"severity {ev.severity:.3f} < {min_sev:.2f}")
            )
            deferred += 1
            continue

        min_d = dur_floor.get(ev.type, 0.008)
        if ev.duration_s < min_d:
            strategies.append(
                _strat(ev, "skip_micro", "deferred", reason=f"duration {ev.duration_s*1000:.1f}ms < {min_d*1000:.0f}ms")
            )
            deferred += 1
            continue

        if _overlaps_claimed(ev, claimed, merge_gap_s=merge_gap, sr=report.sample_rate):
            strategies.append(
                _strat(ev, "skip_overlap", "deferred", reason="overlaps / within merge gap of stronger event")
            )
            deferred += 1
            continue

        strat = _choose_strategy(ev, report, mode=mode)
        if strat == "defer_bilateral_hf":
            strategies.append(
                _strat(
                    ev,
                    "defer_bilateral_hf",
                    "deferred",
                    reason="bilateral HF on dual_mono — skip splice (markers only; avoids music clicks)",
                )
            )
            deferred += 1
            continue

        # Conservative / vocal-safe: never invent bilateral STFT/context (eats consonants).
        # Downgrade to short cubic soft-blend instead of full bilateral_context.
        if mode == "conservative" and strat == "bilateral_context":
            if ev.type == "hard_mute" or (ev.severity >= 0.72 and ev.duration_s <= 0.055):
                strat = "cubic_interp"
            else:
                strategies.append(
                    _strat(
                        ev,
                        "defer_bilateral_vocal",
                        "deferred",
                        reason="conservative vocal-safe: no bilateral full-band fill (use markers / aggressive)",
                    )
                )
                deferred += 1
                continue

        # Bilateral full-band fills invent crackle unless the dip is strong + short.
        if strat == "bilateral_context":
            if ev.severity < 0.78 or ev.duration_s > 0.080:
                strategies.append(
                    _strat(
                        ev,
                        "defer_bilateral_risky",
                        "deferred",
                        reason=(
                            f"bilateral fill needs sev≥0.78 and dur≤80ms "
                            f"(got sev={ev.severity:.2f}, dur={ev.duration_s*1000:.0f}ms)"
                        ),
                    )
                )
                deferred += 1
                continue

        # Conservative: skip STFT / HF reconstruct — they replace voiced texture.
        if mode == "conservative" and strat in {"stft_interp", "hf_band_reconstruct"}:
            if strat == "hf_band_reconstruct" or ev.type != "hard_mute":
                strategies.append(
                    _strat(
                        ev,
                        "defer_spectral_vocal",
                        "deferred",
                        reason="conservative vocal-safe: no STFT/HF replace (keeps syllables)",
                    )
                )
                deferred += 1
                continue

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
        elif mode == "conservative":
            notes_extra = "vocal-safe conservative"
        else:
            notes_extra = ""
        selected.append(ev)
        s = _strat(ev, strat, "planned")
        if notes_extra:
            s["note"] = notes_extra
        strategies.append(s)
        claimed.append((ev.channel, ev.start_sample, ev.end_sample))

    # Density throttle on dropout splices only (impulse de-clicks are tiny + must stay).
    selected, strategies, extra_def = _throttle_density(
        selected, strategies, report.frames / max(1, report.sample_rate), mode=mode
    )
    deferred += extra_def

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

    mode = _norm_mode(mode)
    plan = plan_repairs(report, mode=mode)
    x = channel_matrix(wav.samples).astype(np.float64, copy=True)
    sr = wav.sample_rate
    applied: list[dict[str, Any]] = []
    max_new = _MAX_NEW_BLEND[mode]

    # Apply shortest-first within planned set for cleaner edge context
    order = sorted(plan.events_selected, key=lambda e: (e.start_sample, e.duration_s))
    strat_by_key = {
        (s.get("start_sample"), s.get("channel"), s.get("type")): s for s in plan.strategies if s.get("status") == "planned"
    }

    for ev in order:
        key = (ev.start_sample, ev.channel, ev.type)
        meta = strat_by_key.get(key) or _strat(ev, _choose_strategy(ev, report, mode=mode), "planned")
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

        # Vocal / music preservation: don't replace spans that still carry energy
        # (partial dips / consonants) unless hard_mute or impulse tick.
        if (
            mode == "conservative"
            and ev.type in {"level_dip", "hf_loss"}
            and strategy != "cross_channel_borrow"
            and not _is_true_dropout_span(x[:, ch_i], a, b, max_ratio=0.22)
        ):
            meta["status"] = "deferred"
            meta["strategy"] = "defer_has_content"
            meta["reason"] = "span still has energy — skip replace (protect voiced content)"
            plan.deferred_count += 1
            continue

        try:
            if strategy == "declick_interp":
                # De-click may be stronger than dropout fills — ticks aren't lyrics.
                _declick_interp(
                    x[:, ch_i], a, b, sr, max_new=0.85 if mode == "conservative" else 0.95
                )
            elif strategy == "cross_channel_borrow":
                donor = 1 - ch_i
                # Borrow is safer for vocals (real donor audio) — allow more replacement.
                borrow_new = min(1.0, max_new + 0.35) if mode == "conservative" else max_new
                _cross_channel_borrow(x, ch_i, donor, a, b, sr, max_new=borrow_new)
            elif strategy == "hf_band_reconstruct":
                _hf_band_reconstruct(x[:, ch_i], a, b, sr)
            elif strategy == "bilateral_context":
                _bilateral_context_fill(x[:, ch_i], a, b, sr)
            elif strategy == "stft_interp":
                _stft_interp(x[:, ch_i], a, b, sr=sr)
            else:  # cubic_interp / mirror — keep soft on level dips (lyrics)
                fill_new = 0.30 if (mode == "conservative" and ev.type == "level_dip") else max_new
                _mirror_interp(x[:, ch_i], a, b, sr=sr, max_new=fill_new)
            _seal_boundaries(x[:, ch_i], a, b, sr)
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
    # Hard rule: derived WAV matches input sr / bit depth / channels exactly.
    like: WavAudio | WavFormat = wav
    if subtype is not None:
        like = WavFormat(
            sample_rate=wav.sample_rate,
            channels=wav.channels,
            subtype=subtype.upper(),
            bit_depth=_subtype_bit_depth(subtype.upper()),
        )
    written = write_wav_matching(out_wav, x, like=like)

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
        "format": written.to_dict(),
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


def _overlaps_claimed(
    ev: DropoutEvent,
    claimed: list[tuple[str, int, int]],
    *,
    merge_gap_s: float = 0.0,
    sr: int = 48000,
) -> bool:
    pad = int(round(max(0.0, merge_gap_s) * sr))
    for ch, a, b in claimed:
        if ch != ev.channel:
            continue
        if ev.start_sample < b + pad and ev.end_sample + pad > a:
            return True
    return False


def _throttle_density(
    selected: list[DropoutEvent],
    strategies: list[dict[str, Any]],
    duration_s: float,
    *,
    mode: RepairMode,
) -> tuple[list[DropoutEvent], list[dict[str, Any]], int]:
    """Keep at most N unique dropout time-spans per second (impulse clicks exempt)."""
    if not selected or duration_s <= 0:
        return selected, strategies, 0
    clicks = [e for e in selected if e.type == "impulse_click"]
    drops = [e for e in selected if e.type != "impulse_click"]
    if not drops:
        return selected, strategies, 0
    max_spans = max(3, int(round(_MAX_SPANS_PER_S[mode] * duration_s)))
    # Group by approximate start (10 ms bins) ignoring channel — L+R same hit counts once.
    bins: dict[int, list[DropoutEvent]] = {}
    for ev in drops:
        key = int(round(ev.start_s * 100.0))  # 10 ms
        bins.setdefault(key, []).append(ev)
    ranked = sorted(bins.items(), key=lambda kv: -max(e.severity for e in kv[1]))
    keep_keys = {k for k, _ in ranked[:max_spans]}
    if len(keep_keys) >= len(bins):
        return selected, strategies, 0

    keep_ids = {(e.start_sample, e.channel, e.type) for k in keep_keys for e in bins[k]}
    # Always keep impulse clicks
    keep_ids |= {(e.start_sample, e.channel, e.type) for e in clicks}
    new_selected = [e for e in selected if (e.start_sample, e.channel, e.type) in keep_ids]
    deferred = 0
    for s in strategies:
        if s.get("status") != "planned":
            continue
        if s.get("type") == "impulse_click":
            continue
        key = (s.get("start_sample"), s.get("channel"), s.get("type"))
        if key not in keep_ids:
            s["status"] = "deferred"
            s["strategy"] = "skip_density"
            s["reason"] = f"density cap ~{_MAX_SPANS_PER_S[mode]:.1f} spans/s"
            deferred += 1
    return new_selected, strategies, deferred


def _choose_strategy(ev: DropoutEvent, report: AnalysisReport, *, mode: RepairMode = "conservative") -> str:
    mode = _norm_mode(mode)
    if ev.type == "impulse_click":
        return "declick_interp"
    dual_mono = report.stereo_relationship == "dual_mono_like"
    # Full-track mono→two-track / dual-mono: prefer cross-channel borrow whenever donor is usable.
    if ev.channel in {"L", "R"} and _donor_usable(ev, report, dual_mono=dual_mono):
        return "cross_channel_borrow"
    # Bilateral HF clog on dual-mono: full-band bilateral fill invents LF clicks in music.
    # Mark for defer — markers stay in dropouts.json; do not splice.
    if (
        dual_mono
        and ev.type == "hf_loss"
        and ev.channel in {"L", "R"}
        and _bilateral_damage(ev, report)
    ):
        return "defer_bilateral_hf"
    if ev.type == "hf_loss":
        return "hf_band_reconstruct"
    # Bilateral level damage (debris): long context fill beats short STFT.
    # (Conservative plan_repairs will defer this for vocal safety.)
    if (
        dual_mono
        and ev.type in {"level_dip", "hard_mute"}
        and ev.channel in {"L", "R"}
        and _bilateral_damage(ev, report)
    ):
        return "bilateral_context"
    # Vocal-safe: prefer short cubic over STFT (STFT replaces consonants).
    if mode == "conservative" or ev.duration_s < 0.012:
        return "cubic_interp"
    if ev.duration_s >= 0.012:
        return "stft_interp"
    return "cubic_interp"


def _is_true_dropout_span(ch: np.ndarray, a: int, b: int, *, max_ratio: float = 0.22) -> bool:
    """True when [a:b) is much quieter than neighbors — safe to replace.

    Partial dips that still hold voiced/musical energy return False so we don't
    chew consonants/syllables (Helio: “comeu um pouco as palavras” @ ~1:15).
    """
    n = b - a
    if n <= 0 or ch.size < 8:
        return False
    ctx = max(n, min(2048, n * 4))
    left = ch[max(0, a - ctx) : a]
    right = ch[b : min(ch.size, b + ctx)]
    if left.size < 8 and right.size < 8:
        return True
    span_rms = float(np.sqrt(np.mean(ch[a:b] ** 2) + 1e-20))
    neigh = np.concatenate([left, right]) if left.size and right.size else (left if left.size else right)
    neigh_rms = float(np.sqrt(np.mean(neigh**2) + 1e-20))
    # Require deep hole: default ≤ ~22% of neighbor (~−13 dB)
    return span_rms <= neigh_rms * max_ratio


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


def _declick_interp(ch: np.ndarray, a: int, b: int, sr: int, *, max_new: float = 0.9) -> None:
    """Remove a short source tick/pop by cubic-ish mirror fill + equal-power crossfade."""
    n = b - a
    if n <= 0:
        return
    left = float(ch[a - 1]) if a > 0 else float(ch[a])
    right = float(ch[b]) if b < ch.size else float(ch[b - 1])
    t = np.linspace(0.0, 1.0, n)
    w = t * t * (3.0 - 2.0 * t)
    bridge = (1.0 - w) * left + w * right
    if n >= 8:
        ctx = max(16, min(128, n * 3))
        left_ctx = ch[max(0, a - ctx) : a]
        right_ctx = ch[b : min(ch.size, b + ctx)]
        if left_ctx.size and right_ctx.size:
            from_l = _tile_to(left_ctx[::-1], n)
            from_r = _tile_to(right_ctx[::-1], n)
            tex = from_l * np.cos(0.5 * np.pi * t) + from_r * np.sin(0.5 * np.pi * t)
            fill = 0.65 * bridge + 0.35 * tex
        else:
            fill = bridge
    else:
        fill = bridge
    fill = _match_endpoints(fill, left, right)
    fade = max(1, min(n // 3, int(round(sr * 0.0004))))
    w_new, w_old = _fade_weights(n, fade)
    # Cap replacement strength — conservative keeps more original around vocals
    strength = float(np.clip(max_new, 0.2, 1.0))
    w_new = np.clip(w_new, 0.0, 1.0) * strength
    # Boost center of the tick a bit more than edges
    mid = np.sin(np.pi * t) ** 2
    w_new = np.clip(w_new + 0.25 * mid * strength, 0.0, strength)
    w_old = 1.0 - w_new
    ch[a:b] = w_new * fill + w_old * ch[a:b]


def _fade_len(n: int, sr: int, ms: float = 8.0) -> int:
    """Crossfade length in samples — long enough to avoid audible clicks in music."""
    if n <= 2:
        return 1
    want = int(round(sr * ms / 1000.0))
    # At least ~3 ms, at most 40% of the span (need room for both edges)
    lo = max(32, int(round(sr * 0.003)))
    hi = max(lo, int(n * 0.4))
    return int(np.clip(want, lo, hi))


def _fade_weights(n: int, fade: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power (cosine) edge fades — linear ramps cause clicks on bright music."""
    fade = max(1, min(fade, n // 2 if n >= 2 else 1))
    w_new = np.ones(n, dtype=np.float64)
    if n >= 2 and fade >= 1:
        # raised-cosine / equal-power: sin^2 in, cos^2 out
        t = np.linspace(0.0, 1.0, fade, endpoint=True)
        ramp_in = np.sin(0.5 * np.pi * t) ** 2
        w_new[:fade] = ramp_in
        w_new[-fade:] = ramp_in[::-1]
    return w_new, 1.0 - w_new


def _match_endpoints(fill: np.ndarray, left: float, right: float) -> np.ndarray:
    """Affine-correct fill so first/last samples match neighbors (kills step discontinuities)."""
    if fill.size == 0:
        return fill
    if fill.size == 1:
        return np.array([(left + right) * 0.5], dtype=np.float64)
    out = fill.astype(np.float64, copy=True)
    # Remove linear trend between endpoints, then re-add target endpoints
    t = np.linspace(0.0, 1.0, out.size)
    old_l, old_r = float(out[0]), float(out[-1])
    out = out - ((1.0 - t) * old_l + t * old_r)
    out = out + ((1.0 - t) * left + t * right)
    return out


def _seal_boundaries(ch: np.ndarray, a: int, b: int, sr: int) -> None:
    """Post-repair micro-crossfade around event edges to kill residual clicks/pops."""
    n = ch.size
    if n < 4 or b <= a:
        return
    fade = _fade_len(max(8, b - a), sr, ms=4.0)
    # Blend a short neighborhood straddling each edge back toward continuity
    for edge in (a, b):
        lo = max(0, edge - fade)
        hi = min(n, edge + fade)
        if hi - lo < 4:
            continue
        # Local linear bridge across the edge neighborhood
        left = float(ch[lo])
        right = float(ch[hi - 1])
        t = np.linspace(0.0, 1.0, hi - lo)
        bridge = (1.0 - t) * left + t * right
        # Equal-power mix: keep most of signal, pull edges toward bridge
        w = np.sin(np.pi * t) ** 2  # peaks at center/edge
        # Stronger correction right at the boundary index
        mid = edge - lo
        w = np.clip(w * 0.35, 0.0, 0.35)
        if 0 <= mid < w.size:
            w[mid] = min(0.55, w[mid] + 0.25)
        ch[lo:hi] = (1.0 - w) * ch[lo:hi] + w * bridge


def _blend_fill(
    ch: np.ndarray,
    a: int,
    b: int,
    fill: np.ndarray,
    sr: int,
    ms: float = 8.0,
    *,
    max_new: float = 1.0,
) -> None:
    n = b - a
    if n <= 0 or fill.size != n:
        return
    left = float(ch[a - 1]) if a > 0 else float(fill[0])
    right = float(ch[b]) if b < ch.size else float(fill[-1])
    fill = _match_endpoints(fill, left, right)
    fade = _fade_len(n, sr, ms=ms)
    w_new, w_old = _fade_weights(n, fade)
    strength = float(np.clip(max_new, 0.0, 1.0))
    w_new = w_new * strength
    w_old = 1.0 - w_new
    ch[a:b] = w_new * fill + w_old * ch[a:b]


def _mirror_interp(
    ch: np.ndarray, a: int, b: int, ctx: int = 128, sr: int = 48000, *, max_new: float = 1.0
) -> None:
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
        fill = from_l * np.cos(0.5 * np.pi * t) + from_r * np.sin(0.5 * np.pi * t)
    _blend_fill(ch, a, b, fill, sr, ms=10.0, max_new=max_new)


def _bilateral_context_fill(ch: np.ndarray, a: int, b: int, sr: int = 48000) -> None:
    """Longer-context fill for dual-mono bilateral dropouts (both channels damaged).

    Cross-channel borrow is unavailable; use extended mirror from each side of the
    gap plus a light STFT magnitude blend so short full-track hits are less 'holey'.
    """
    n = b - a
    if n <= 0:
        return
    ctx = max(256, min(2048, n * 4))
    _mirror_interp(ch, a, b, ctx=ctx, sr=sr)
    # Refine with STFT if gap is long enough for a stable transform
    if n >= 64:
        backup = ch[a:b].copy()
        try:
            _stft_interp(ch, a, b, sr=sr, n_fft=min(512, 1 << int(np.ceil(np.log2(max(64, n))))), hop=64)
            # Prefer mirror at edges (continuity), STFT in the middle (tonal)
            t = np.linspace(0.0, 1.0, n)
            w_stft = np.sin(np.pi * t) ** 2
            ch[a:b] = (1.0 - w_stft) * backup + w_stft * ch[a:b]
            _seal_boundaries(ch, a, b, sr)
        except Exception:  # noqa: BLE001
            ch[a:b] = backup


def _tile_to(seg: np.ndarray, n: int) -> np.ndarray:
    if seg.size == 0:
        return np.zeros(n, dtype=np.float64)
    reps = int(np.ceil(n / seg.size))
    return np.tile(seg, reps)[:n].astype(np.float64)


def _cross_channel_borrow(
    x: np.ndarray, target: int, donor: int, a: int, b: int, sr: int = 48000, *, max_new: float = 1.0
) -> None:
    n = b - a
    ctx = min(256, a, x.shape[0] - b)
    src = x[a:b, donor].copy()
    # RMS-match donor segment to target pre/post context
    tgt_ctx = np.concatenate([x[max(0, a - ctx) : a, target], x[b : min(x.shape[0], b + ctx), target]])
    don_ctx = np.concatenate([x[max(0, a - ctx) : a, donor], x[b : min(x.shape[0], b + ctx), donor]])
    rms_t = float(np.sqrt(np.mean(tgt_ctx**2) + 1e-20)) if tgt_ctx.size else 1.0
    rms_d = float(np.sqrt(np.mean(don_ctx**2) + 1e-20)) if don_ctx.size else 1.0
    scaled = src * (rms_t / max(rms_d, 1e-12))
    _blend_fill(x[:, target], a, b, scaled, sr, ms=10.0, max_new=max_new)


def _stft_interp(ch: np.ndarray, a: int, b: int, sr: int = 48000, n_fft: int = 512, hop: int = 128) -> None:
    """Interpolate STFT frames across the gap; overlap-add back into ch[a:b]."""
    pad = n_fft * 2
    start = max(0, a - pad)
    end = min(ch.size, b + pad)
    seg = ch[start:end].copy()
    if seg.size < n_fft:
        _mirror_interp(ch, a, b, sr=sr)
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
        _mirror_interp(ch, a, b, sr=sr)
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

    fill = recon[gap_a:gap_b]
    _blend_fill(ch, a, b, fill, sr, ms=10.0)


def _hf_band_reconstruct(ch: np.ndarray, a: int, b: int, sr: int, n_fft: int = 512, hop: int = 128) -> None:
    """Keep LF + local phase; rebuild HF magnitude from neighbors (phase swap → clicks)."""
    pad = n_fft * 2
    start = max(0, a - pad)
    end = min(ch.size, b + pad)
    seg = ch[start:end].copy()
    if seg.size < n_fft:
        _mirror_interp(ch, a, b, sr=sr)
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
        _mirror_interp(ch, a, b, sr=sr)
        return

    mags = np.abs(specs)
    phases = np.angle(specs)  # keep local phase — interpolated donor phase causes estalos
    for i in bad:
        left = max((g for g in good if g < i), default=None)
        right = min((g for g in good if g > i), default=None)
        if left is None and right is None:
            continue
        if left is None:
            donor_mag = mags[right]
        elif right is None:
            donor_mag = mags[left]
        else:
            t = (i - left) / max(1, right - left)
            donor_mag = (1 - t) * mags[left] + t * mags[right]
        # Soft HF lift toward donor (never hard-replace); keep mid mostly local
        mags[i, hf_bins] = 0.35 * mags[i, hf_bins] + 0.65 * donor_mag[hf_bins]
        mags[i, mid_bins] = 0.7 * mags[i, mid_bins] + 0.3 * donor_mag[mid_bins]

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
    _blend_fill(ch, a, b, fill, sr, ms=12.0)
