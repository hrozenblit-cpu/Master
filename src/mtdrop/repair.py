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
    # impulse_click cap allows coalesced neighboring ticks into one OLA span
    # bilateral_tok ~4–8 ms body + pad; keep well under vocal-chew widths
    "conservative": {
        "hard_mute": 0.080,
        "level_dip": 0.055,
        "hf_loss": 0.035,
        "impulse_click": 0.022,
        "bilateral_tok": 0.014,
    },
    "aggressive": {
        "hard_mute": 0.20,
        "level_dip": 0.25,
        "hf_loss": 0.30,
        "impulse_click": 0.030,
        "bilateral_tok": 0.016,
    },
}
_MAX_DUR_BORROW = {
    "conservative": {"hard_mute": 0.15, "level_dip": 0.12, "hf_loss": 0.12},
    "aggressive": {"hard_mute": 0.40, "level_dip": 0.50, "hf_loss": 0.50},
}
# Floor for *apply* (detection may still list milder markers).
# Conservative is intentionally high — prefer markers over inventing syllables.
_MIN_SEVERITY = {
    # level_dip 0.72 keeps pior.wav real holes; apply-time content gate blocks voiced FPs
    # impulse_click 0.58 keeps Samba ~0:25 (sev≈0.66–0.73 after azimuth) while cutting mild FPs
    # bilateral_tok: high floor — only clear knocks (Samba ~24.438 sev≈1.0).
    # Milder mid onsets in vocals are left alone (Helio: don't chew words).
    "conservative": {
        "hard_mute": 0.55,
        "level_dip": 0.72,
        "hf_loss": 0.92,
        "impulse_click": 0.58,
        "bilateral_tok": 0.80,
    },
    "aggressive": {
        "hard_mute": 0.40,
        "level_dip": 0.60,
        "hf_loss": 0.82,
        "impulse_click": 0.48,
        # 0.70 ≈ jump≳3.45 — Samba primary; secondary ~24.59 (sev≈0.58) stays marker-only
        "bilateral_tok": 0.70,
    },
}
# Skip micro repairs that only create edge clicks (seconds).
# impulse_click is intentionally sub-ms…few-ms — do not treat as skip_micro.
_MIN_APPLY_DUR = {
    "conservative": {
        "hard_mute": 0.003,
        "level_dip": 0.008,
        "hf_loss": 0.025,
        "impulse_click": 0.00015,
        "bilateral_tok": 0.002,
    },
    "aggressive": {
        "hard_mute": 0.003,
        "level_dip": 0.005,
        "hf_loss": 0.015,
        "impulse_click": 0.0001,
        "bilateral_tok": 0.002,
    },
}
# Merge same-channel planned spans closer than this (seconds).
_MERGE_GAP_S = {"conservative": 0.040, "aggressive": 0.020}
# Cap unique dropout time-spans per second of audio (keep highest severity).
_MAX_SPANS_PER_S = {"conservative": 0.6, "aggressive": 2.0}
# Impulse de-clicks used to be density-exempt → ~6 splices/s carpet of crossfade ticks.
_MAX_IMPULSE_PER_S = {"conservative": 0.45, "aggressive": 1.5}
# Coalesce nearby impulse peaks into one OLA span (seconds).
_IMPULSE_COALESCE_S = {"conservative": 0.018, "aggressive": 0.012}
# Coalesce nearby bilateral toks into one modest Hermite span (seconds).
_TOK_COALESCE_S = {"conservative": 0.016, "aggressive": 0.014}
# Tok Hermite blend strength (partial — preserves underlying RMS; full=1 ducks/mush).
_TOK_STRENGTH = {"conservative": 0.62, "aggressive": 0.72}
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

    # Per-channel morphology + source impulse clicks + bilateral toks (ticks/pops/toks).
    candidates = [
        e
        for e in report.events
        if e.type in {"level_dip", "hard_mute", "hf_loss", "impulse_click", "bilateral_tok"}
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

        # Impulses/toks: allow close neighbors into the plan — apply coalesces them.
        # Dropouts still use the wider merge gap so dense HF doesn't stack splices.
        if ev.type == "impulse_click":
            gap = 0.001
        elif ev.type == "bilateral_tok":
            gap = 0.002
        else:
            gap = merge_gap
        if _overlaps_claimed(ev, claimed, merge_gap_s=gap, sr=report.sample_rate):
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

    # Density throttle: dropouts first, then impulse clicks (was exempt → splice carpet).
    dur_s = report.frames / max(1, report.sample_rate)
    selected, strategies, extra_def = _throttle_density(selected, strategies, dur_s, mode=mode)
    deferred += extra_def
    selected, strategies, extra_imp = _throttle_impulses(selected, strategies, dur_s, mode=mode)
    deferred += extra_imp
    selected, strategies, extra_tok = _throttle_toks(selected, strategies, dur_s, mode=mode)
    deferred += extra_tok

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
    dual_mono = report.stereo_relationship == "dual_mono_like"

    # Apply shortest-first within planned set for cleaner edge context
    order = sorted(plan.events_selected, key=lambda e: (e.start_sample, e.duration_s))
    strat_by_key = {
        (s.get("start_sample"), s.get("channel"), s.get("type")): s for s in plan.strategies if s.get("status") == "planned"
    }

    # Coalesce nearby impulse clicks into one OLA span (cuts join count; Helio crossfade carpet).
    coalesce_gap = _IMPULSE_COALESCE_S[mode]
    impulse_spans = _coalesce_impulse_spans(
        [e for e in order if e.type == "impulse_click"],
        gap_s=coalesce_gap,
        sr=sr,
        n_frames=x.shape[0],
    )
    absorbed_impulse: set[tuple[int, str]] = set()
    for span in impulse_spans:
        for ev in span["events"]:
            absorbed_impulse.add((ev.start_sample, ev.channel))
        a, b = span["a"], span["b"]
        ch_set = []
        if dual_mono and x.shape[1] >= 2:
            ch_set = [0, 1]
        else:
            for e in span["events"]:
                ci = _channel_index(e.channel, x.shape[1])
                if ci is not None and ci not in ch_set:
                    ch_set.append(ci)
        try:
            for ci in ch_set:
                _declick_interp(x[:, ci], a, b, sr, max_new=1.0)
            applied.append(
                {
                    "start_s": a / sr,
                    "end_s": b / sr,
                    "start_sample": a,
                    "end_sample": b,
                    "channel": "L+R" if len(ch_set) > 1 else ("L" if ch_set[0] == 0 else "R"),
                    "type": "impulse_click",
                    "strategy": "declick_interp",
                    "note": f"per-peak declick n={len(span['events'])}",
                }
            )
            for ev in span["events"]:
                key = (ev.start_sample, ev.channel, ev.type)
                meta = strat_by_key.get(key)
                if meta is not None:
                    meta["status"] = "applied"
                    meta["strategy"] = "declick_interp"
                    meta["note"] = "ola coalesce"
        except Exception as exc:  # noqa: BLE001
            for ev in span["events"]:
                key = (ev.start_sample, ev.channel, ev.type)
                meta = strat_by_key.get(key)
                if meta is not None:
                    meta["status"] = "failed"
                    meta["reason"] = str(exc)

    # Bilateral toks: modestly wider Hermite over tok body (not residual-peak hunting).
    tok_spans = _coalesce_impulse_spans(
        [e for e in order if e.type == "bilateral_tok"],
        gap_s=_TOK_COALESCE_S[mode],
        sr=sr,
        n_frames=x.shape[0],
    )
    absorbed_tok: set[tuple[int, str]] = set()
    tok_strength = _TOK_STRENGTH[mode]
    for span in tok_spans:
        for ev in span["events"]:
            absorbed_tok.add((ev.start_sample, ev.channel))
        a, b = span["a"], span["b"]
        # Clamp span to modest tok width so coalesced neighbors don't chew vocals.
        max_tok = int(round(_MAX_DUR[mode]["bilateral_tok"] * sr))
        if b - a > max_tok:
            mid = (a + b) // 2
            a = max(0, mid - max_tok // 2)
            b = min(x.shape[0], a + max_tok)
        ch_set = [0, 1] if x.shape[1] >= 2 else [0]
        try:
            for ci in ch_set:
                _declick_tok(x[:, ci], a, b, sr, strength=tok_strength)
            applied.append(
                {
                    "start_s": a / sr,
                    "end_s": b / sr,
                    "start_sample": a,
                    "end_sample": b,
                    "channel": "L+R" if len(ch_set) > 1 else "mono",
                    "type": "bilateral_tok",
                    "strategy": "declick_tok",
                    "note": f"tok/thump Hermite n={len(span['events'])} strength={tok_strength:.2f}",
                }
            )
            for ev in span["events"]:
                key = (ev.start_sample, ev.channel, ev.type)
                meta = strat_by_key.get(key)
                if meta is not None:
                    meta["status"] = "applied"
                    meta["strategy"] = "declick_tok"
                    meta["note"] = "tok coalesce"
        except Exception as exc:  # noqa: BLE001
            for ev in span["events"]:
                key = (ev.start_sample, ev.channel, ev.type)
                meta = strat_by_key.get(key)
                if meta is not None:
                    meta["status"] = "failed"
                    meta["reason"] = str(exc)

    for ev in order:
        key = (ev.start_sample, ev.channel, ev.type)
        meta = strat_by_key.get(key) or _strat(ev, _choose_strategy(ev, report, mode=mode), "planned")
        if (
            ev.type == "impulse_click"
            or ev.type == "bilateral_tok"
            or (ev.start_sample, ev.channel) in absorbed_impulse
            or (ev.start_sample, ev.channel) in absorbed_tok
        ):
            if meta.get("status") == "planned":
                meta["status"] = "applied"
                meta["note"] = "ola coalesce" if ev.type == "impulse_click" else "tok coalesce"
            continue
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

        # Dual-mono: never apply single-channel dropout fills (causes L/R pump).
        if (
            dual_mono
            and ev.type in {"level_dip", "hf_loss", "hard_mute"}
            and strategy not in {"cross_channel_borrow", "declick_interp", "declick_tok"}
            and ev.channel in {"L", "R"}
        ):
            meta["status"] = "deferred"
            meta["strategy"] = "defer_asymmetric_dual_mono"
            meta["reason"] = "dual_mono_like: skip single-channel fill (avoids L/R pump)"
            plan.deferred_count += 1
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
            # No _seal_boundaries after these — _blend_fill already does long equal-power
            # edges; a second seal was inventing ticks/swish (Helio crossfade feedback).
            if strategy == "cross_channel_borrow":
                donor = 1 - ch_i
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

    # Dual-mono / full-track→two-track: re-match L/R RMS after edits so repairs
    # never leave one channel ducked while the other is hotter (Helio "pump").
    if dual_mono and x.shape[1] >= 2:
        _balance_dual_mono_rms(x)

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
    """Keep at most N unique dropout time-spans per second (impulse/tok clicks exempt)."""
    if not selected or duration_s <= 0:
        return selected, strategies, 0
    clicks = [e for e in selected if e.type in {"impulse_click", "bilateral_tok"}]
    drops = [e for e in selected if e.type not in {"impulse_click", "bilateral_tok"}]
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
    # Always keep impulse clicks / bilateral toks (their own throttle applies)
    keep_ids |= {(e.start_sample, e.channel, e.type) for e in clicks}
    new_selected = [e for e in selected if (e.start_sample, e.channel, e.type) in keep_ids]
    deferred = 0
    for s in strategies:
        if s.get("status") != "planned":
            continue
        if s.get("type") in {"impulse_click", "bilateral_tok"}:
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
    if ev.type == "bilateral_tok":
        return "declick_tok"
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


def _balance_dual_mono_rms(x: np.ndarray) -> None:
    """Match L/R full-file RMS to their mid — kills residual L/R pump after edits."""
    if x.ndim != 2 or x.shape[1] < 2:
        return
    l_rms = float(np.sqrt(np.mean(x[:, 0] ** 2) + 1e-20))
    r_rms = float(np.sqrt(np.mean(x[:, 1] ** 2) + 1e-20))
    mid = 0.5 * (l_rms + r_rms)
    if l_rms > 1e-12:
        x[:, 0] *= mid / l_rms
    if r_rms > 1e-12:
        x[:, 1] *= mid / r_rms
    peak = float(np.max(np.abs(x))) or 1.0
    if peak > 0.99:
        x *= 0.99 / peak


def _residual_impulse_cleanup(
    x: np.ndarray,
    sr: int,
    *,
    dual_mono: bool,
    near_times_s: list[float] | None = None,
    abs_floor: float = 0.0055,
    mad_k: float = 7.0,
    max_hits: int = 24,
    near_radius_s: float = 0.012,
) -> list[dict[str, Any]]:
    """Extra de-click near already-repaired impulse spans (splice-edge residuals only)."""
    if x.ndim != 2 or x.shape[0] < 64:
        return []
    if not near_times_s:
        return []
    mid = 0.5 * (x[:, 0] + x[:, 1]) if x.shape[1] >= 2 else x[:, 0]
    err = np.abs(mid - 0.5 * (np.roll(mid, 1) + np.roll(mid, -1)))
    err[0] = 0.0
    err[-1] = 0.0
    # Mask: only search ±near_radius around prior impulse centers
    allow = np.zeros(mid.size, dtype=bool)
    rad = int(round(near_radius_s * sr))
    for t in near_times_s:
        c = int(round(float(t) * sr))
        allow[max(0, c - rad) : min(mid.size, c + rad + 1)] = True
    if not np.any(allow):
        return []
    win = max(64, int(0.04 * sr))
    hop = max(16, win // 4)
    mad = np.zeros(mid.size, dtype=np.float64)
    for i in range(0, mid.size, hop):
        a = max(0, i - win // 2)
        b = min(mid.size, a + win)
        a = max(0, b - win)
        med = float(np.median(err[a:b]))
        mad[i : min(mid.size, i + hop)] = float(np.median(np.abs(err[a:b] - med))) + 1e-12
    if mad[-1] == 0:
        mad[mad == 0] = float(np.median(mad[mad > 0])) if np.any(mad > 0) else 1e-6
    thr = np.maximum(mad_k * mad, abs_floor)
    peaks = np.where(allow & (err > thr))[0]
    if peaks.size == 0:
        return []
    pad = max(4, int(round(0.0016 * sr)))
    min_sep = max(pad, int(0.006 * sr))
    order = peaks[np.argsort(-err[peaks])]
    chosen: list[int] = []
    for i in order:
        if any(abs(i - j) < min_sep for j in chosen):
            continue
        chosen.append(int(i))
        if len(chosen) >= max_hits:
            break
    applied: list[dict[str, Any]] = []
    channels = [0, 1] if (dual_mono and x.shape[1] >= 2) else list(range(x.shape[1]))
    for i in sorted(chosen):
        a = max(0, i - pad)
        b = min(x.shape[0], i + pad + 1)
        for ci in channels:
            _declick_interp(x[:, ci], a, b, sr, max_new=1.0)
        applied.append(
            {
                "start_s": a / sr,
                "end_s": b / sr,
                "start_sample": a,
                "end_sample": b,
                "channel": "L+R" if len(channels) > 1 else ("L" if channels[0] == 0 else "R"),
                "type": "impulse_click",
                "strategy": "declick_interp_residual",
            }
        )
    return applied


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


def _coalesce_impulse_spans(
    events: list[DropoutEvent],
    *,
    gap_s: float,
    sr: int,
    n_frames: int,
) -> list[dict[str, Any]]:
    """Merge impulse events within gap_s into union spans for a single OLA apply."""
    if not events:
        return []
    ordered = sorted(events, key=lambda e: (e.start_sample, e.channel))
    gap = max(1, int(round(gap_s * sr)))
    spans: list[dict[str, Any]] = []
    cur_events = [ordered[0]]
    cur_a = int(ordered[0].start_sample)
    cur_b = int(ordered[0].end_sample)
    for ev in ordered[1:]:
        a, b = int(ev.start_sample), int(ev.end_sample)
        if a <= cur_b + gap:
            cur_b = max(cur_b, b)
            cur_a = min(cur_a, a)
            cur_events.append(ev)
        else:
            spans.append({"a": max(0, cur_a), "b": min(n_frames, cur_b), "events": cur_events})
            cur_events = [ev]
            cur_a, cur_b = a, b
    spans.append({"a": max(0, cur_a), "b": min(n_frames, cur_b), "events": cur_events})
    return spans


def _expand_impulse_core(
    x: np.ndarray, a: int, b: int, sr: int, *, radius_s: float = 0.020, abs_floor: float = 0.0055
) -> tuple[int, int]:
    """Grow [a,b) to include nearby mid-mix residual peaks (twin ticks)."""
    if x.ndim != 2 or x.shape[0] < 8:
        return a, b
    mid = 0.5 * (x[:, 0] + x[:, 1]) if x.shape[1] >= 2 else x[:, 0]
    rad = max(4, int(round(radius_s * sr)))
    lo = max(0, a - rad)
    hi = min(mid.size, b + rad)
    err = np.abs(mid[lo:hi] - 0.5 * (np.roll(mid[lo:hi], 1) + np.roll(mid[lo:hi], -1)))
    if err.size < 3:
        return a, b
    err[0] = 0.0
    err[-1] = 0.0
    peaks = np.where(err >= abs_floor)[0]
    if peaks.size == 0:
        return a, b
    pad = max(4, int(round(0.0016 * sr)))
    a2 = min(a, lo + int(peaks.min()) - pad)
    b2 = max(b, lo + int(peaks.max()) + pad + 1)
    return max(0, a2), min(x.shape[0], b2)


def _throttle_impulses(
    selected: list[DropoutEvent],
    strategies: list[dict[str, Any]],
    duration_s: float,
    *,
    mode: RepairMode,
) -> tuple[list[DropoutEvent], list[dict[str, Any]], int]:
    """Cap impulse de-clicks per second, fair across the timeline.

    Global top-N by severity starved late ticks (Samba ~0:25). Instead:
    - always keep high-severity hits (true pops),
    - density-cap milder ones in ~1.5 s slots (best severity per slot).
    """
    if not selected or duration_s <= 0:
        return selected, strategies, 0
    clicks = [e for e in selected if e.type == "impulse_click"]
    if not clicks:
        return selected, strategies, 0

    always_sev = 0.70 if mode == "conservative" else 0.58
    strong = [e for e in clicks if e.severity >= always_sev]
    mild = [e for e in clicks if e.severity < always_sev]

    # Slot ALL clicks (strong + mild) so early FPs don't erase later real ticks,
    # but strong ones always win their slot and are never dropped for density.
    slot_s = 1.5
    per_slot = max(1, int(round(_MAX_IMPULSE_PER_S[mode] * slot_s)))
    slot_bins: dict[int, list[DropoutEvent]] = {}
    for ev in clicks:
        slot_bins.setdefault(int(ev.start_s // slot_s), []).append(ev)
    kept_clicks: list[DropoutEvent] = []
    for _slot, evs in slot_bins.items():
        by20: dict[int, list[DropoutEvent]] = {}
        for ev in evs:
            by20.setdefault(int(round(ev.start_s * 50.0)), []).append(ev)
        ranked = sorted(by20.values(), key=lambda group: -max(e.severity for e in group))
        # Always include every strong group in this slot, plus top mild up to per_slot
        strong_groups = [g for g in ranked if max(e.severity for e in g) >= always_sev]
        mild_groups = [g for g in ranked if max(e.severity for e in g) < always_sev]
        # At most one strong time-key per slot (+ expand covers twin peaks nearby)
        for group in strong_groups[:1]:
            kept_clicks.extend(group)
        for group in mild_groups[: max(0, per_slot)]:
            kept_clicks.extend(group)

    # Revive twin ticks within 20 ms of a kept impulse (Samba ~24.429 + ~24.445).
    # Density may drop the milder twin; keep it so tight per-peak declick can hit both.
    keep_click_ids = {(e.start_sample, e.channel, e.type) for e in kept_clicks}
    kept_times = [e.start_s for e in kept_clicks]
    for ev in clicks:
        key = (ev.start_sample, ev.channel, ev.type)
        if key in keep_click_ids:
            continue
        if any(abs(ev.start_s - t) <= 0.020 for t in kept_times):
            kept_clicks.append(ev)
            keep_click_ids.add(key)
            kept_times.append(ev.start_s)

    if len(keep_click_ids) >= len(clicks):
        return selected, strategies, 0

    non_clicks = [e for e in selected if e.type != "impulse_click"]
    new_selected = non_clicks + [e for e in clicks if (e.start_sample, e.channel, e.type) in keep_click_ids]
    deferred = 0
    for s in strategies:
        if s.get("status") != "planned" or s.get("type") != "impulse_click":
            continue
        key = (s.get("start_sample"), s.get("channel"), s.get("type"))
        if key not in keep_click_ids:
            s["status"] = "deferred"
            s["strategy"] = "skip_impulse_density"
            s["reason"] = f"impulse density cap ~{_MAX_IMPULSE_PER_S[mode]:.1f}/s (time-stratified)"
            deferred += 1
        elif s.get("strategy") == "skip_impulse_density":
            # revived after an earlier mark — shouldn't happen in one pass
            pass
        else:
            # Ensure revived twins are planned
            if s.get("status") == "deferred" and key in keep_click_ids:
                s["status"] = "planned"
                s["strategy"] = "declick_interp"
                s["reason"] = "revived twin impulse"
                deferred = max(0, deferred - 1)
    return new_selected, strategies, deferred


def _throttle_toks(
    selected: list[DropoutEvent],
    strategies: list[dict[str, Any]],
    duration_s: float,
    *,
    mode: RepairMode,
) -> tuple[list[DropoutEvent], list[dict[str, Any]], int]:
    """Cap bilateral tok repairs — always keep strong knocks, sparse milder ones."""
    if not selected or duration_s <= 0:
        return selected, strategies, 0
    toks = [e for e in selected if e.type == "bilateral_tok"]
    if not toks:
        return selected, strategies, 0
    always_sev = 0.85 if mode == "conservative" else 0.78
    # Group L+R by start
    by_start: dict[int, list[DropoutEvent]] = {}
    for e in toks:
        by_start.setdefault(e.start_sample, []).append(e)
    groups = list(by_start.values())
    strong = [g for g in groups if max(e.severity for e in g) >= always_sev]
    mild = [g for g in groups if max(e.severity for e in g) < always_sev]
    slot_s = 2.0
    slots: dict[int, list[list[DropoutEvent]]] = {}
    for g in mild:
        slots.setdefault(int(g[0].start_s // slot_s), []).append(g)
    kept_g = list(strong)
    for _s, gs in slots.items():
        gs.sort(key=lambda g: -max(e.severity for e in g))
        kept_g.extend(gs[:1])
    keep_ids = {(e.start_sample, e.channel, e.type) for g in kept_g for e in g}
    if len(keep_ids) >= len(toks):
        return selected, strategies, 0
    non = [e for e in selected if e.type != "bilateral_tok"]
    new_selected = non + [e for e in toks if (e.start_sample, e.channel, e.type) in keep_ids]
    deferred = 0
    for s in strategies:
        if s.get("status") != "planned" or s.get("type") != "bilateral_tok":
            continue
        key = (s.get("start_sample"), s.get("channel"), s.get("type"))
        if key not in keep_ids:
            s["status"] = "deferred"
            s["strategy"] = "skip_tok_density"
            s["reason"] = "tok density cap (keep strong knocks; sparse milder)"
            deferred += 1
    return new_selected, strategies, deferred


def _declick_tok(ch: np.ndarray, a: int, b: int, sr: int, *, strength: float = 0.72) -> None:
    """Modest Hermite replace for bilateral tok/thump body (~5–6 ms per peak).

    Unlike residual-peak ``_declick_interp``, this targets the low-mid knock envelope
    that survives spike declick (Samba ~24.438). Searches ±10 ms for neighbor tok
    bodies (24.438+24.446 cluster) and replaces each with a short partial Hermite so
    underlying RMS stays up — too-wide full OLA ducks music.
    """
    if b <= a or ch.size < 8:
        return
    a = max(0, min(int(a), ch.size))
    b = max(a, min(int(b), ch.size))
    # Find envelope peaks in a modest search window around the planned span.
    search = max(int(round(0.010 * sr)), (b - a) // 2)
    mid = (a + b) // 2
    lo = max(0, mid - search)
    hi = min(ch.size, mid + search)
    seg = ch[lo:hi].astype(np.float64, copy=False)
    if seg.size < 16:
        centers = [mid]
    else:
        # Cheap |x| smooth ≈ tok body tracker (no FFT in the hot repair path).
        win = max(3, int(round(0.0015 * sr)))
        env = np.convolve(np.abs(seg), np.ones(win) / win, mode="same")
        # Local maxima above median*1.35
        med = float(np.median(env) + 1e-12)
        peaks: list[int] = []
        for i in range(2, env.size - 2):
            if env[i] >= env[i - 1] and env[i] >= env[i + 1] and env[i] >= 1.35 * med:
                peaks.append(i)
        if not peaks:
            peaks = [int(np.argmax(env))]
        # Rank by env, NMS ~4 ms, keep up to 3 (tok bursts)
        peaks.sort(key=lambda i: -env[i])
        chosen: list[int] = []
        min_sep = max(4, int(round(0.004 * sr)))
        for i in peaks:
            if any(abs(i - j) < min_sep for j in chosen):
                continue
            chosen.append(i)
            if len(chosen) >= 3:
                break
        centers = sorted(lo + i for i in chosen)

    core_half = max(4, int(round(0.00275 * sr)))  # ~5.5 ms core
    wing = max(16, int(round(0.0025 * sr)))
    strength = float(np.clip(strength, 0.35, 1.0))

    for c in centers:
        ca = max(0, c - core_half)
        cb = min(ch.size, c + core_half)
        if cb - ca < 4:
            continue
        a0 = max(0, ca - wing)
        b0 = min(ch.size, cb + wing)
        n = b0 - a0
        left = float(ch[ca - 1]) if ca > 0 else float(ch[ca])
        right = float(ch[cb]) if cb < ch.size else float(ch[cb - 1])
        left_slope = float(ch[ca - 1] - ch[ca - 2]) if ca >= 2 else 0.0
        right_slope = float(ch[cb + 1] - ch[cb]) if cb + 1 < ch.size else 0.0
        t_core = np.linspace(0.0, 1.0, max(1, cb - ca))
        t2 = t_core * t_core
        t3 = t2 * t_core
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t_core
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        core_fill = h00 * left + h10 * left_slope + h01 * right + h11 * right_slope
        core_fill = _match_endpoints(core_fill, left, right)

        fill = ch[a0:b0].astype(np.float64, copy=True)
        fill[ca - a0 : cb - a0] = core_fill
        w = np.zeros(n, dtype=np.float64)
        fade_l = ca - a0
        fade_r = b0 - cb
        if fade_l > 0:
            tl = np.linspace(0.0, 1.0, fade_l, endpoint=True)
            w[:fade_l] = np.sin(0.5 * np.pi * tl) ** 2
        w[ca - a0 : cb - a0] = 1.0
        if fade_r > 0:
            tr = np.linspace(0.0, 1.0, fade_r, endpoint=True)
            w[cb - a0 :] = np.cos(0.5 * np.pi * tr) ** 2
        w = w * strength
        orig = ch[a0:b0].astype(np.float64, copy=False)
        ch[a0:b0] = w * fill + (1.0 - w) * orig


def _declick_interp(ch: np.ndarray, a: int, b: int, sr: int, *, max_new: float = 0.9) -> None:
    """Surgical per-peak declick inside [a,b) — preserve music between ticks.

    Earlier OLA bridged the whole coalesced span (~20 ms) with a near-flat Hermite,
    ducking energy ~50% (Helio still heard ~0:25 as unfixed). Now: find high-crest
    residual peaks and replace only ~1.2 ms around each, with ~2 ms equal-power wings.
    """
    if b <= a or ch.size < 8:
        return
    a = max(0, min(int(a), ch.size))
    b = max(a, min(int(b), ch.size))
    seg = ch[a:b].astype(np.float64, copy=False)
    if seg.size < 3:
        return
    err = np.abs(seg - 0.5 * (np.roll(seg, 1) + np.roll(seg, -1)))
    err[0] = 0.0
    err[-1] = 0.0
    med = float(np.median(err))
    mad = float(np.median(np.abs(err - med))) + 1e-12
    thr = max(0.005, 5.5 * mad)
    peaks = np.where(err >= thr)[0]
    if peaks.size == 0:
        # still hit the worst sample if clearly impulsive vs neighbors
        i = int(np.argmax(err))
        if err[i] < max(0.004, 4.0 * mad):
            return
        peaks = np.array([i])

    # Crest gate: |sample| / local RMS — reject bright music texture
    half_ctx = max(8, int(round(0.0015 * sr)))
    candidates: list[int] = []
    order = peaks[np.argsort(-err[peaks])]
    for i in order:
        abs_i = a + int(i)
        lo = max(0, abs_i - half_ctx)
        hi = min(ch.size, abs_i + half_ctx)
        local = ch[lo:hi]
        rms = float(np.sqrt(np.mean(local**2)) + 1e-12)
        crest = float(abs(ch[abs_i]) / rms)
        # Samba ticks ~1.4–1.6 crest on mid; allow slightly lower after azimuth
        if crest < 1.15 and err[i] < 0.007:
            continue
        candidates.append(int(i))
    if not candidates:
        candidates = [int(np.argmax(err))]

    # Cluster peaks within ~2.5 ms of each seed (click bursts), one core per cluster.
    # Prior min_sep=4 ms skipped Samba's 24.429/24.430 neighbors and left the tick.
    cluster_r = max(4, int(round(0.0025 * sr)))
    used = np.zeros(len(candidates), dtype=bool)
    clusters: list[tuple[int, int]] = []  # (ca, cb) absolute
    for idx, i_rel in enumerate(candidates):
        if used[idx]:
            continue
        # primary cluster: tight burst around this peak
        members = [i_rel]
        used[idx] = True
        for jdx in range(idx + 1, len(candidates)):
            if used[jdx]:
                continue
            if abs(candidates[jdx] - i_rel) <= cluster_r:
                members.append(candidates[jdx])
                used[jdx] = True
        pad = max(3, int(round(0.0005 * sr)))
        ca = max(0, a + min(members) - pad)
        cb = min(ch.size, a + max(members) + pad + 1)
        clusters.append((ca, cb))
        if len(clusters) >= 4:
            break

    wing = max(20, int(round(0.0022 * sr)))  # ~2.2 ms OLA wings
    strength = float(np.clip(max_new, 0.5, 1.0))

    for ca, cb in clusters:
        a0 = max(0, ca - wing)
        b0 = min(ch.size, cb + wing)
        n = b0 - a0
        if n < 4 or cb <= ca:
            continue
        left = float(ch[ca - 1]) if ca > 0 else float(ch[ca])
        right = float(ch[cb]) if cb < ch.size else float(ch[cb - 1])
        left_slope = float(ch[ca - 1] - ch[ca - 2]) if ca >= 2 else 0.0
        right_slope = float(ch[cb + 1] - ch[cb]) if cb + 1 < ch.size else 0.0
        t_core = np.linspace(0.0, 1.0, max(1, cb - ca))
        t2 = t_core * t_core
        t3 = t2 * t_core
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t_core
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        core_fill = h00 * left + h10 * left_slope + h01 * right + h11 * right_slope
        core_fill = _match_endpoints(core_fill, left, right)

        fill = ch[a0:b0].astype(np.float64, copy=True)
        fill[ca - a0 : cb - a0] = core_fill
        w = np.zeros(n, dtype=np.float64)
        fade_l = ca - a0
        fade_r = b0 - cb
        if fade_l > 0:
            tl = np.linspace(0.0, 1.0, fade_l, endpoint=True)
            w[:fade_l] = np.sin(0.5 * np.pi * tl) ** 2
        w[ca - a0 : cb - a0] = 1.0
        if fade_r > 0:
            tr = np.linspace(0.0, 1.0, fade_r, endpoint=True)
            w[cb - a0 :] = np.cos(0.5 * np.pi * tr) ** 2
        w = w * strength
        orig = ch[a0:b0].astype(np.float64, copy=False)
        ch[a0:b0] = w * fill + (1.0 - w) * orig


def _fade_len(n: int, sr: int, ms: float = 12.0) -> int:
    """Crossfade length in samples — prefer ≥~3 ms equal-power edges on music."""
    if n <= 2:
        return 1
    want = int(round(sr * ms / 1000.0))
    # At least ~3 ms when span allows; at most 45% of the span (both edges)
    lo = min(max(24, int(round(sr * 0.003))), max(1, n // 2))
    hi = max(lo, int(n * 0.45))
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
    """Deprecated no-op.

    Older builds mixed a linear bridge over edges *after* equal-power blends,
    inventing ticks/swish (Helio). Callers use `_blend_fill` / OLA declick only.
    """
    return


def _blend_fill(
    ch: np.ndarray,
    a: int,
    b: int,
    fill: np.ndarray,
    sr: int,
    ms: float = 12.0,
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
    _blend_fill(ch, a, b, fill, sr, ms=12.0, max_new=max_new)


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
    _blend_fill(x[:, target], a, b, scaled, sr, ms=12.0, max_new=max_new)


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
    _blend_fill(ch, a, b, fill, sr, ms=12.0)


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
