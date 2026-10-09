from __future__ import annotations

import json
from pathlib import Path

import soundfile as sf

from mtdrop.fixture import synthesize_dropout_wav
from mtdrop.preview import export_padded_clip, export_event_previews


def test_export_padded_clip_min_duration(tmp_path: Path) -> None:
    wav = tmp_path / "src.wav"
    synthesize_dropout_wav(wav, sample_rate=48000, duration_s=2.0, stereo=True)
    out = tmp_path / "clip.wav"
    a, b = export_padded_clip(wav, out, center_s=0.9, pad_s=0.05, min_duration_s=2.5)
    assert out.exists()
    info = sf.info(str(out))
    dur = info.frames / info.samplerate
    assert dur >= 2.45  # allow tiny rounding
    assert (b - a) >= 2.45


def test_cli_preview_command(tmp_path: Path) -> None:
    from mtdrop.cli import main

    wav = tmp_path / "src.wav"
    synthesize_dropout_wav(wav, sample_rate=48000, duration_s=2.0, stereo=True)
    events = tmp_path / "e.json"
    events.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "start_s": 0.90,
                        "end_s": 0.925,
                        "severity": 1.0,
                        "type": "hard_mute",
                        "channel": "both",
                    }
                ]
            }
        )
    )
    out = tmp_path / "prev"
    rc = main(
        [
            "preview",
            "--wav",
            f"original={wav}",
            "--events",
            str(events),
            "--out",
            str(out),
            "--stem",
            "demo",
            "--min-duration",
            "2.5",
            "--no-mp3",
        ]
    )
    assert rc == 0
    clips = list(out.glob("*.wav"))
    assert clips
    for c in clips:
        info = sf.info(str(c))
        assert info.frames / info.samplerate >= 2.45
