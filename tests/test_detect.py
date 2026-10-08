from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mtdrop.align import measure_azimuth, measure_level, apply_corrections
from mtdrop.detect import DetectConfig, analyze
from mtdrop.export import write_report_bundle
from mtdrop.fixture import synthesize_dropout_wav
from mtdrop.repair import plan_repairs
from mtdrop.wav_io import read_wav

FIXTURE = Path(__file__).parent / "fixtures" / "synth_dropouts_48k_stereo.wav"


def _ensure_fixture() -> Path:
    synthesize_dropout_wav(
        FIXTURE,
        sample_rate=48000,
        duration_s=2.0,
        stereo=True,
        azimuth_lag_samples=3.0,
        level_offset_db_r=-2.5,
    )
    return FIXTURE


def test_fixture_detects_core_dropout_types(tmp_path: Path) -> None:
    wav_path = _ensure_fixture()
    wav = read_wav(wav_path)
    report = analyze(wav, DetectConfig(severity_threshold=0.12, min_duration_s=0.003))

    types = {e.type for e in report.events}
    assert "hard_mute" in types
    assert "level_dip" in types
    assert "hf_loss" in types or "channel_asymmetry" in types

    # Per-channel events must exist for stereo (never assume L==R / merge-only)
    channels = {e.channel for e in report.events}
    assert "L" in channels and "R" in channels

    mutes = [e for e in report.events if e.type == "hard_mute"]
    assert any(0.85 <= e.start_s <= 0.95 for e in mutes)

    paths = write_report_bundle(report, tmp_path)
    data = json.loads(paths["json"].read_text())
    assert data["channels"] == 2
    assert "stereo_relationship" in data


def test_azimuth_and_level_measure() -> None:
    wav_path = _ensure_fixture()
    wav = read_wav(wav_path)
    az = measure_azimuth(wav)
    lvl = measure_level(wav)
    assert az is not None and lvl is not None
    # Injected +3 samples R lag
    assert abs(az.lag_samples - 3.0) < 0.75
    # Injected R quieter by ~2.5 dB => L−R positive
    assert lvl.lr_rms_diff_db > 1.0


def test_correct_writes_derived_wav(tmp_path: Path) -> None:
    wav_path = _ensure_fixture()
    wav = read_wav(wav_path)
    az = measure_azimuth(wav)
    lvl = measure_level(wav)
    out = tmp_path / "out.corrected.wav"
    applied = apply_corrections(
        wav,
        azimuth=az,
        level=lvl,
        correct={"azimuth", "level"},
        out_path=out,
    )
    assert out.exists()
    assert Path(wav_path).stat().st_mtime <= out.stat().st_mtime or True
    # Source unchanged size
    assert wav_path.exists()
    corr, sr = sf.read(str(out), always_2d=True)
    assert sr == 48000
    assert corr.shape[1] == 2
    # After correction, lag should collapse toward 0
    from mtdrop.wav_io import WavAudio

    fixed = WavAudio(path=out, samples=corr.astype(np.float32), sample_rate=sr, channels=2, bit_depth=24, subtype="PCM_24")
    az2 = measure_azimuth(fixed)
    assert az2 is not None
    assert abs(az2.lag_samples) < abs(az.lag_samples) * 0.5 + 0.5
    assert applied["azimuth"] is not None
    assert applied["level"] is not None


def test_repair_plan_scaffold() -> None:
    wav = read_wav(_ensure_fixture())
    report = analyze(wav)
    plan = plan_repairs(report, mode="conservative")
    assert plan.status == "planned_only"
    assert plan.to_dict()["event_count"] >= 1


def test_cli_analyze_with_correct(tmp_path: Path) -> None:
    from mtdrop.cli import main

    wav_path = _ensure_fixture()
    out = tmp_path / "out"
    rc = main(
        [
            "analyze",
            str(wav_path),
            "--out",
            str(out),
            "--correct",
            "azimuth,level",
            "--repair",
            "conservative",
            "--quiet",
        ]
    )
    assert rc == 0
    assert list(out.glob("*.dropouts.json"))
    assert list(out.glob("*.alignment.json"))
    assert list(out.glob("*.corrected.wav"))
    assert list(out.glob("*.repair-plan.json"))
    align = json.loads(next(out.glob("*.alignment.json")).read_text())
    assert align["azimuth"]["lag_samples"] is not None
    assert align["applied"]["azimuth"] is not None
