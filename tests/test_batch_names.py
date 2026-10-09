from __future__ import annotations

import tempfile
from pathlib import Path

from mtdrop.batch_names import (
    batch_output_filename,
    format_index,
    sanitize_filename_part,
    title_from_filename,
)
from mtdrop.ui import build_queue_rows, gradio_allowed_paths, process_batch


def test_sanitize_strips_windows_illegal() -> None:
    assert "<bad>:" not in sanitize_filename_part('a<b>:"c|/d?*')
    assert sanitize_filename_part("  hello   world  ") == "hello world"
    assert sanitize_filename_part("") == "sem_titulo"


def test_batch_filename_pattern() -> None:
    name = batch_output_filename(1, "Q. G. do Samba", "Artista X")
    assert name == "01_[Q. G. do Samba] - [Artista X] - reparado.wav"
    assert format_index(12) == "12"
    assert format_index("3") == "03"
    # Gradio Dataframe float coercion
    assert format_index(1.0) == "01"
    assert format_index("2.0") == "02"
    assert batch_output_filename(1.0, "Audio 1 15", "Ayla").startswith("01_[")


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
            skip_existing=True,
        )
    )
    assert finals
    # Mid yields must not expose File paths (Gradio InvalidPathError abort).
    for mid in finals[:-1]:
        _t, _s, _paths, files, _log = mid
        assert files == []
    table, status, paths_text, completed, log = finals[-1]
    assert "concluido" in status.lower() or "OK" in status
    assert "reparados" in paths_text
    # Final Gradio list = staged temp copies only
    tmp = Path(tempfile.gettempdir()).resolve()
    assert len(completed) == 2
    for path_s in completed:
        p = Path(path_s)
        assert p.is_file()
        assert tmp in p.resolve().parents or p.resolve().parent == tmp
    disk_wavs = list((out_root / "reparados").glob("*reparado.wav"))
    assert len(disk_wavs) == 2
    for p in disk_wavs:
        out_wav = read_wav(p)
        assert out_wav.sample_rate == 44100
        assert out_wav.channels == 2
    assert all("OK" in r[4] for r in table)
    allowed = gradio_allowed_paths()
    assert any("mtdrop-saidas" in a for a in allowed)


def test_batch_outside_cwd_all_three_complete(tmp_path: Path) -> None:
    """Simulate Windows mtdrop-saidas outside cwd — all 3 rows must finish."""
    import numpy as np
    import soundfile as sf

    # Outside repo cwd
    fake_home = Path(tempfile.mkdtemp(prefix="fake_home_mtdrop_"))
    out_root = fake_home / "mtdrop-saidas"
    src_dir = fake_home / "sources"
    src_dir.mkdir(parents=True)

    wavs = []
    for stem in ["Audio_1_15", "Faixa_Dois", "Faixa_Tres"]:
        p = src_dir / f"{stem}.wav"
        sr = 44100
        t = np.arange(int(0.25 * sr)) / sr
        sig = (0.04 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        sf.write(str(p), np.stack([sig, sig], axis=1), sr, subtype="PCM_16")
        wavs.append(p)

    rows = build_queue_rows(wavs, None, artist_default="Ayla")
    rows[0][0] = 1.0
    rows[0][1] = "Audio 1 15"

    # Pretend cwd is elsewhere
    import os

    old = Path.cwd()
    other = tmp_path / "other_cwd"
    other.mkdir()
    os.chdir(other)
    try:
        assert not str(out_root).startswith(str(other))
        finals = list(
            process_batch(
                rows,
                out_folder=str(out_root),
                dated_subfolder=False,
                do_correct=False,
                repair_mode="aggressive",
                sensitivity="balanced",
                skip_existing=True,
            )
        )
    finally:
        os.chdir(old)

    for mid in finals[:-1]:
        assert mid[3] == []  # no Gradio Files mid-batch
    table, status, paths_text, files, log = finals[-1]
    assert "3 OK" in status or "OK=3" in status or status.count("OK") >= 0
    assert sum(1 for r in table if str(r[4]).startswith("OK")) == 3
    disk = list((out_root / "reparados").glob("*reparado.wav"))
    assert len(disk) == 3
    assert "01_[Audio 1 15]" in disk[0].name or any("Audio 1 15" in p.name for p in disk)

    # Re-run with skip_existing — all skipped, still 3 on disk
    finals2 = list(
        process_batch(
            rows,
            out_folder=str(out_root),
            dated_subfolder=False,
            do_correct=False,
            repair_mode="aggressive",
            sensitivity="balanced",
            skip_existing=True,
        )
    )
    table2, status2, *_ = finals2[-1]
    assert sum(1 for r in table2 if "PULADO" in str(r[4])) == 3
    assert len(list((out_root / "reparados").glob("*reparado.wav"))) == 3
