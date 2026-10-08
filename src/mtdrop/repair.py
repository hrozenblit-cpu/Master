from __future__ import annotations

"""Phase C scaffold — dropout repair (not fully implemented in v0.1).

Planned strategies (see docs plan Phase C):
- micro / short level_dip + hard_mute → cubic / AR / STFT interpolation
- hf_loss → high-band reconstruct from temporal neighbors
- channel_asymmetry → optional cross-channel borrow when stereo_relationship allows

v0.1 exposes the API surface and CLI flag but refuses to silently invent audio
until Helio validates detection FP rate on real A80 transfers.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mtdrop.models import AnalysisReport, DropoutEvent

RepairMode = Literal["off", "conservative", "preview"]


@dataclass(slots=True)
class RepairPlan:
    mode: RepairMode
    events_selected: list[DropoutEvent]
    strategies: list[dict[str, Any]]
    status: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "message": self.message,
            "event_count": len(self.events_selected),
            "strategies": self.strategies,
            "events": [e.to_dict() for e in self.events_selected],
        }


def plan_repairs(
    report: AnalysisReport,
    *,
    mode: RepairMode = "conservative",
    max_duration_s: float = 0.05,
) -> RepairPlan:
    """Select candidate events and assign intended strategies (no audio write yet)."""
    if mode == "off":
        return RepairPlan(mode=mode, events_selected=[], strategies=[], status="skipped", message="repair off")

    selected: list[DropoutEvent] = []
    strategies: list[dict[str, Any]] = []
    for ev in report.events:
        if ev.type == "channel_asymmetry":
            continue  # overlay tag; repair the underlying morphology events
        if ev.duration_s > max_duration_s and mode == "conservative":
            strategies.append(
                {
                    "event_start_s": ev.start_s,
                    "type": ev.type,
                    "strategy": "defer_manual",
                    "reason": f"duration {ev.duration_s:.4f}s > conservative max {max_duration_s}s",
                }
            )
            continue
        selected.append(ev)
        if ev.type == "hard_mute" or ev.type == "level_dip":
            strat = "stft_interp" if ev.duration_s >= 0.01 else "cubic_interp"
        elif ev.type == "hf_loss":
            strat = "hf_band_reconstruct"
        else:
            strat = "defer_manual"
        strategies.append(
            {
                "event_start_s": ev.start_s,
                "event_end_s": ev.end_s,
                "channel": ev.channel,
                "type": ev.type,
                "strategy": strat,
                "status": "planned",
            }
        )

    return RepairPlan(
        mode=mode,
        events_selected=selected,
        strategies=strategies,
        status="planned_only",
        message=(
            "Phase C scaffold: repair plan generated but audio not written in v0.1. "
            "Use markers + RX/CEDAR for heal until --repair gains an apply path."
        ),
    )


def apply_repairs(
    source_wav: Path,
    plan: RepairPlan,
    out_wav: Path,
) -> dict[str, Any]:
    """Reserved for Phase C apply. Raises until implemented."""
    raise NotImplementedError(
        "Dropout audio repair apply is Phase C — not yet implemented. "
        f"Plan has {len(plan.strategies)} strateg(ies); source={source_wav}, out={out_wav}"
    )
