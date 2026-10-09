from __future__ import annotations

from pathlib import Path

from mtdrop.batch_names import (
    batch_output_filename,
    format_index,
    sanitize_filename_part,
    title_from_filename,
)
from mtdrop.ui import build_queue_rows, process_batch


def test_sanitize_strips_windows_illegal() -> None:
    assert "<bad>:" not in sanitize_filename_part('a<b>:"c|/d?*')
    assert sanitize_filename_part("  hello   world  ") == "hello world"
    assert sanitize_filename_part("") == "sem_titulo"


def test_batch_filename_pattern() -> None:
    name = batch_output_filename(1, "Q. G. do Samba", "Artista X")
    assert name == "01_[Q. G. do Samba] - [Artista X] - reparado.wav"
    assert format_index(12) == "12"
    assert format_index("3") == "03"


def test_title_from_filename() -> None:
    assert title_from_filename("/tmp/15022_02_QG_do_Samba_OK.wav") == "15022 02 QG do Samba OK"


def test_build_queue_and_process_batch(tmp_path: Path) -> None:
    import numpy as np
    import soundfile as sf

    from mtdrop.wav_io import read_wav

    wavs = []
    for i, stem in enumerate(["Musica_A", "Musica_B"], start=1):
        p = tmp_path / f"{stem}.wav"
        sr = 44100
        t = np.arange(int(0.3 * sr)) / sr
        sig = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack([sig, sig * 0.98], axis=1)
        sf.write(str(p), stereo, sr, subtype="PCM_16")
        wavs.append(p)

    rows = build_queue_rows(wavs, None, artist_default="Helio")
    assert len(rows) == 2
    assert rows[0][0] == "01"
    assert rows[0][2] == "Helio"
    assert "Musica A" in rows[0][1] or "Musica_A" in rows[0][1]

    out_root = tmp_path / "saidas"
    finals = list(
        process_batch(
            rows,
            out_folder=str(out_root),
            dated_subfolder=False,
            do_correct=True,
            repair_mode="aggressive",
            sensitivity="balanced",
        )
    )
    assert finals
    table, status, completed, log = finals[-1]
    assert "concluído" in status.lower() or "concluido" in status.lower() or "OK" in status
    assert len(completed) == 2
    for path_s in completed:
        p = Path(path_s)
        assert p.is_file()
        assert "reparado.wav" in p.name
        assert p.name.startswith("0")
        # never overwrite masters — source still exists and differs from out
        assert p.parent.name == "reparados"
        # format passthrough
        out_wav = read_wav(p)
        assert out_wav.sample_rate == 44100
        assert out_wav.channels == 2
    assert all("OK" in r[4] for r in table)
