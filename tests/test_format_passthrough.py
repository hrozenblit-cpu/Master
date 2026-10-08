from __future__ import annotations

from pathlib import Path

import pytest
import soundfile as sf

from mtdrop.align import apply_corrections, measure_azimuth, measure_level
from mtdrop.detect import analyze, config_for_sensitivity
from mtdrop.fixture import synthesize_dropout_wav
from mtdrop.repair import apply_repairs
from mtdrop.wav_io import formats_match, probe_format, read_wav, write_wav_matching


MEDIA = Path("/cursor/stores/self/media")
EXEMPLO = MEDIA / "exemplo_1_18.wav"
PIOR = MEDIA / "pior.wav"


def _assert_match(src: Path, *outs: Path) -> None:
    want = probe_format(src)
    for out in outs:
        got = probe_format(out)
        assert formats_match(want, got), f"{out.name}: wanted {want.to_dict()} got {got.to_dict()}"


def test_write_wav_matching_preserves_pcm16_and_pcm24(tmp_path: Path) -> None:
    p16 = tmp_path / "a16.wav"
    p24 = tmp_path / "a24.wav"
    synthesize_dropout_wav(p16, sample_rate=44100, duration_s=0.5, stereo=True, subtype="PCM_16")
    synthesize_dropout_wav(p24, sample_rate=88200, duration_s=0.5, stereo=True, subtype="PCM_24")
    for src in (p16, p24):
        wav = read_wav(src)
        out = tmp_path / f"out_{src.name}"
        write_wav_matching(out, wav.samples, like=wav)
        _assert_match(src, out)


def test_correct_and_repair_match_input_format_synthetic(tmp_path: Path) -> None:
    src = tmp_path / "src.wav"
    synthesize_dropout_wav(src, sample_rate=44100, duration_s=2.0, stereo=True, subtype="PCM_16")
    wav = read_wav(src)
    corr = tmp_path / "c.wav"
    apply_corrections(
        wav,
        azimuth=measure_azimuth(wav),
        level=measure_level(wav),
        correct={"azimuth", "level"},
        out_path=corr,
    )
    work = read_wav(corr)
    rep = tmp_path / "r.wav"
    apply_repairs(work, analyze(work, config_for_sensitivity("balanced")), rep, mode="conservative")
    _assert_match(src, corr, rep)


@pytest.mark.skipif(not EXEMPLO.exists(), reason="Helio exemplo_1_18.wav not in store")
def test_helio_exemplo_88k24_passthrough(tmp_path: Path) -> None:
    wav = read_wav(EXEMPLO)
    assert wav.sample_rate == 88200 and wav.subtype == "PCM_24" and wav.channels == 2
    corr = tmp_path / "exemplo.corrected.wav"
    apply_corrections(
        wav,
        azimuth=measure_azimuth(wav),
        level=measure_level(wav),
        correct={"azimuth", "level"},
        out_path=corr,
    )
    work = read_wav(corr)
    rep = tmp_path / "exemplo.repaired.wav"
    apply_repairs(work, analyze(work, config_for_sensitivity("balanced")), rep, mode="conservative")
    _assert_match(EXEMPLO, corr, rep)
    for p in (corr, rep):
        info = sf.info(str(p))
        assert info.samplerate == 88200
        assert info.subtype == "PCM_24"
        assert info.channels == 2


@pytest.mark.skipif(not PIOR.exists(), reason="Helio pior.wav not in store")
def test_helio_pior_44k16_passthrough(tmp_path: Path) -> None:
    wav = read_wav(PIOR)
    assert wav.sample_rate == 44100 and wav.subtype == "PCM_16" and wav.channels == 2
    corr = tmp_path / "pior.corrected.wav"
    apply_corrections(
        wav,
        azimuth=measure_azimuth(wav),
        level=measure_level(wav),
        correct={"azimuth", "level"},
        out_path=corr,
    )
    work = read_wav(corr)
    rep = tmp_path / "pior.repaired.wav"
    apply_repairs(work, analyze(work, config_for_sensitivity("balanced")), rep, mode="conservative")
    _assert_match(PIOR, corr, rep)
    for p in (corr, rep):
        info = sf.info(str(p))
        assert info.samplerate == 44100
        assert info.subtype == "PCM_16"
        assert info.channels == 2
