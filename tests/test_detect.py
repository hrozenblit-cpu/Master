from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from mtdrop.align import measure_azimuth, measure_level, apply_corrections
from mtdrop.detect import DetectConfig, analyze
from mtdrop.export import write_report_bundle
from mtdrop.fixture import synthesize_dropout_wav
from mtdrop.repair import apply_repairs, plan_repairs
from mtdrop.wav_io import WavAudio, read_wav

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
    report = analyze(wav, DetectConfig(severity_threshold=0.12, min_duration_s=0.003, hf_ratio_drop=0.35, dip_ratio=0.35))

    types = {e.type for e in report.events}
    assert "hard_mute" in types
    assert "level_dip" in types
    assert "hf_loss" in types or "channel_asymmetry" in types

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
    assert abs(az.lag_samples - 3.0) < 0.75
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
    assert wav_path.exists()
    corr, sr = sf.read(str(out), always_2d=True)
    assert sr == 48000
    assert corr.shape[1] == 2
    fixed = WavAudio(
        path=out, samples=corr.astype(np.float32), sample_rate=sr, channels=2, bit_depth=24, subtype="PCM_24"
    )
    az2 = measure_azimuth(fixed)
    assert az2 is not None
    assert abs(az2.lag_samples) < abs(az.lag_samples) * 0.5 + 0.5
    assert applied["azimuth"] is not None
    assert applied["level"] is not None


def test_repair_applies_and_fills_hard_mute(tmp_path: Path) -> None:
    wav_path = _ensure_fixture()
    wav = read_wav(wav_path)
    report = analyze(wav, DetectConfig(hf_ratio_drop=0.35, dip_ratio=0.35, min_duration_s=0.003, severity_threshold=0.12))
    plan = plan_repairs(report, mode="conservative")
    assert plan.status == "planned"
    assert len(plan.events_selected) >= 1

    out = tmp_path / "out.repaired.wav"
    result = apply_repairs(wav, report, out, mode="conservative")
    assert out.exists()
    assert result.plan.status == "applied"
    assert result.plan.repaired_count >= 1
    assert result.provenance["repaired_events"]

    # Source untouched: repaired is a different path
    assert Path(result.provenance["source_wav"]).resolve() != out.resolve()

    # Hard mute region (~0.90–0.925) should have higher energy after repair
    sr = wav.sample_rate
    a, b = int(0.90 * sr), int(0.925 * sr)
    raw = wav.samples if wav.samples.ndim == 2 else wav.samples.reshape(-1, 1)
    rep, _ = sf.read(str(out), always_2d=True)
    raw_e = float(np.mean(raw[a:b] ** 2))
    rep_e = float(np.mean(rep[a:b] ** 2))
    assert rep_e > raw_e * 5.0


def test_cli_analyze_correct_and_repair(tmp_path: Path) -> None:
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
            "--sensitivity",
            "aggressive",
            "--quiet",
        ]
    )
    assert rc == 0
    assert list(out.glob("*.dropouts.json"))
    assert list(out.glob("*.alignment.json"))
    assert list(out.glob("*.corrected.wav"))
    assert list(out.glob("*.repaired.wav"))
    assert list(out.glob("*.repair.json"))
    repair = json.loads(next(out.glob("*.repair.json")).read_text())
    assert repair["status"] == "applied"
    assert repair["repaired_count"] >= 1
    assert "calibration_note" in repair
