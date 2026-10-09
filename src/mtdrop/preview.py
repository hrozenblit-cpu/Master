from __future__ import annotations

"""Listen-friendly preview export — padded clips (default ≥2.5 s) + optional loudnorm MP3."""

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


@dataclass(slots=True)
class PreviewClip:
    label: str
    start_s: float
    end_s: float
    wav_path: Path
    mp3_path: Path | None = None
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.end_s - self.start_s,
            "wav": str(self.wav_path),
            "mp3": str(self.mp3_path) if self.mp3_path else None,
            "source": self.source,
        }


def export_padded_clip(
    source_wav: Path,
    out_wav: Path,
    *,
    center_s: float | None = None,
    start_s: float | None = None,
    end_s: float | None = None,
    pad_s: float = 1.25,
    min_duration_s: float = 2.5,
    subtype: str = "PCM_16",
) -> tuple[float, float]:
    """Write a padded excerpt. Returns (start_s, end_s) actually written.

    Always enforces at least ``min_duration_s`` (default 2.5 s) so IDE players can audition.
    """
    source_wav = Path(source_wav)
    out_wav = Path(out_wav)
    info = sf.info(str(source_wav))
    dur = info.frames / info.samplerate

    if start_s is not None and end_s is not None:
        a, b = float(start_s), float(end_s)
    elif center_s is not None:
        a = float(center_s) - pad_s
        b = float(center_s) + pad_s
    else:
        raise ValueError("provide center_s or start_s+end_s")

    # Expand to minimum duration, centered on the requested span
    span = max(0.0, b - a)
    if span < min_duration_s:
        mid = 0.5 * (a + b) if span > 0 else (center_s if center_s is not None else a)
        half = min_duration_s / 2.0
        a, b = mid - half, mid + half

    a = max(0.0, a)
    b = min(dur, b)
    # If clamped at an edge, extend the other side to keep min duration when possible
    if b - a < min_duration_s - 1e-6:
        need = min_duration_s - (b - a)
        a = max(0.0, a - need)
        b = min(dur, a + min_duration_s)

    data, sr = sf.read(
        str(source_wav),
        start=int(a * info.samplerate),
        stop=max(int(a * info.samplerate) + 1, int(b * info.samplerate)),
        always_2d=True,
    )
    # If the source file itself is shorter than min_duration, pad with silence so
    # IDE players still get a ≥min_duration buffer (avoids "inaudible" sub-second clips).
    target = int(round(min_duration_s * sr))
    if data.shape[0] < target:
        pad = target - data.shape[0]
        data = np.pad(data, ((0, pad), (0, 0)), mode="constant")
        b = a + data.shape[0] / sr

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), data, sr, subtype=subtype)
    return a, b


def loudnorm_mp3(wav_path: Path, mp3_path: Path | None = None, *, i: float = -16.0, tp: float = -1.5) -> Path:
    """Export loudnorm MP3 via ffmpeg for easy IDE/browser playback."""
    wav_path = Path(wav_path)
    mp3_path = Path(mp3_path) if mp3_path else wav_path.with_suffix(".mp3")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — required for MP3 preview export")
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(wav_path),
        "-af",
        f"loudnorm=I={i}:TP={tp}:LRA=11",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        "4",
        str(mp3_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return mp3_path


def export_event_previews(
    sources: dict[str, Path],
    events: list[dict[str, Any]],
    out_dir: Path,
    *,
    stem: str,
    pad_s: float = 1.25,
    min_duration_s: float = 2.5,
    max_events: int = 3,
    mp3: bool = True,
) -> list[PreviewClip]:
    """Export padded previews for the top events across labeled sources (original/repaired/…)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Rank by severity, unique-ish centers
    ranked = sorted(events, key=lambda e: -float(e.get("severity", 0.0)))
    picked: list[dict[str, Any]] = []
    for ev in ranked:
        mid = 0.5 * (float(ev["start_s"]) + float(ev["end_s"]))
        if any(abs(mid - 0.5 * (float(p["start_s"]) + float(p["end_s"]))) < 0.4 for p in picked):
            continue
        picked.append(ev)
        if len(picked) >= max_events:
            break

    clips: list[PreviewClip] = []
    for i, ev in enumerate(picked):
        mid = 0.5 * (float(ev["start_s"]) + float(ev["end_s"]))
        tag = f"e{i+1}_{mid:.1f}s".replace(".", "p")
        for label, src in sources.items():
            if src is None or not Path(src).exists():
                continue
            wav_out = out_dir / f"{stem}_preview_{label}_{tag}.wav"
            a, b = export_padded_clip(
                Path(src),
                wav_out,
                center_s=mid,
                pad_s=pad_s,
                min_duration_s=min_duration_s,
            )
            mp3_out = None
            if mp3:
                try:
                    mp3_out = loudnorm_mp3(wav_out)
                except Exception:  # noqa: BLE001
                    mp3_out = None
            clips.append(
                PreviewClip(
                    label=f"{label}_{tag}",
                    start_s=a,
                    end_s=b,
                    wav_path=wav_out,
                    mp3_path=mp3_out,
                    source=str(src),
                )
            )
    return clips


def events_from_dropout_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("events") or [])
