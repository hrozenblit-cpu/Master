from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

DropoutType = Literal[
    "level_dip",
    "hard_mute",
    "hf_loss",
    "channel_asymmetry",
    "impulse_click",
    "bilateral_tok",
]
DurationClass = Literal["micro", "short", "medium", "long"]
ChannelTag = Literal["mono", "L", "R", "both", "L>R", "R>L"]
StereoRelationship = Literal["mono", "dual_mono_like", "true_stereo", "unknown"]


def duration_class(duration_s: float) -> DurationClass:
    ms = duration_s * 1000.0
    if ms < 5.0:
        return "micro"
    if ms < 50.0:
        return "short"
    if ms < 300.0:
        return "medium"
    return "long"


@dataclass(slots=True)
class DropoutEvent:
    start_s: float
    end_s: float
    start_sample: int
    end_sample: int
    channel: ChannelTag
    type: DropoutType
    severity: float
    confidence: float
    duration_class: DurationClass
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["duration_s"] = self.duration_s
        return d


@dataclass(slots=True)
class AnalysisReport:
    source: str
    sample_rate: int
    channels: int
    frames: int
    bit_depth: int | None
    events: list[DropoutEvent] = field(default_factory=list)
    stereo_relationship: StereoRelationship = "unknown"
    channel_correlation: float | None = None
    tool: str = "mtdrop"
    tool_version: str = "0.2.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_version": self.tool_version,
            "source": self.source,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "bit_depth": self.bit_depth,
            "stereo_relationship": self.stereo_relationship,
            "channel_correlation": self.channel_correlation,
            "notes": (
                "Per-channel detection never assumes L==R. "
                "stereo_relationship is a correlation hint only: "
                "dual_mono_like includes full-track mono tape digitized as two-track "
                "(L≈R program; differences from azimuth/gain/dropout asymmetry)."
            ),
            "event_count": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }
