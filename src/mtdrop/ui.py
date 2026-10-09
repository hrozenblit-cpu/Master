"""Local Gradio UI for full-file mtdrop listen / A-B testing + batch queue.

Never overwrites the uploaded source — all outputs go to a work dir or chosen folder.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from mtdrop.align import apply_corrections, measure_azimuth, measure_level
from mtdrop.batch_names import batch_output_filename, format_index, title_from_filename
from mtdrop.detect import analyze, config_for_sensitivity
from mtdrop.export import write_json_obj, write_report_bundle
from mtdrop.repair import apply_repairs
from mtdrop.wav_io import read_wav

# Gradio File downloads must live under temp/cwd/allowed_paths. Real deliverables
# still go to the user-chosen folder; we mirror copies here for the UI component.
_DOWNLOAD_STAGE = Path(tempfile.mkdtemp(prefix="mtdrop_gradio_dl_"))
# Stable copies of Gradio multi-upload temps (they can vanish between queue build & run).
_UPLOAD_STAGE = Path(tempfile.mkdtemp(prefix="mtdrop_gradio_up_"))


def default_batch_out_root() -> Path:
    return (Path.home() / "mtdrop-saidas").expanduser()


def gradio_allowed_paths() -> list[str]:
    """Paths Gradio may expose via File/Audio components (Windows InvalidPathError fix)."""
    roots = [
        default_batch_out_root(),
        default_batch_out_root() / "reparados",
        _DOWNLOAD_STAGE,
        _UPLOAD_STAGE,
        Path(tempfile.gettempdir()),
        Path.cwd(),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for p in roots:
        try:
            s = str(p.expanduser().resolve())
        except OSError:
            s = str(p.expanduser())
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _stage_for_gradio_download(disk_path: Path) -> Path | None:
    """Copy a user-folder deliverable into the Gradio-safe download stage.

    Returns the staged path, or None if copy fails (disk save still counts as OK).
    """
    try:
        _DOWNLOAD_STAGE.mkdir(parents=True, exist_ok=True)
        dest = _DOWNLOAD_STAGE / disk_path.name
        # Unique if same name already staged this session
        if dest.exists():
            stem, ext = dest.stem, dest.suffix
            n = 2
            while True:
                cand = _DOWNLOAD_STAGE / f"{stem}_{n}{ext}"
                if not cand.exists():
                    dest = cand
                    break
                n += 1
        shutil.copy2(disk_path, dest)
        return dest
    except OSError:
        return None


def _format_exc(exc: BaseException) -> str:
    """Short + full exception text for row status / log."""
    short = f"{type(exc).__name__}: {exc}"
    return short


def _run_pipeline(
    wav_path: str | Path,
    *,
    do_correct: bool,
    repair_mode: str,
    sensitivity: str,
    progress: Any = None,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    wav_path = Path(wav_path)
    work = Path(work_dir) if work_dir is not None else Path(tempfile.mkdtemp(prefix="mtdrop_ui_"))
    work.mkdir(parents=True, exist_ok=True)
    logs: list[str] = []

    def tick(frac: float, msg: str) -> None:
        logs.append(msg)
        if progress is not None:
            try:
                progress(frac, desc=msg)
            except TypeError:
                progress(frac)

    tick(0.05, f"Reading {wav_path.name}")
    wav = read_wav(wav_path)
    cfg = config_for_sensitivity(sensitivity)

    tick(0.2, "Detecting dropouts…")
    report = analyze(wav, cfg)
    write_report_bundle(report, work)

    az = measure_azimuth(wav) if wav.channels == 2 else None
    lvl = measure_level(wav) if wav.channels == 2 else None
    align = {
        "source": str(wav_path),
        "stereo_relationship": report.stereo_relationship,
        "channel_correlation": report.channel_correlation,
        "azimuth": az.to_dict() if az else None,
        "level": lvl.to_dict() if lvl else None,
        "policy": "Source never overwritten; outputs are derived copies.",
    }
    write_json_obj(align, work / f"{wav_path.stem}.alignment.json")

    corrected_path = None
    work_wav = wav
    repair_report = report
    if do_correct and wav.channels == 2:
        tick(0.45, "Applying azimuth + level correction…")
        corrected_path = work / f"{wav_path.stem}.corrected.wav"
        apply_corrections(
            wav,
            azimuth=az,
            level=lvl,
            correct={"azimuth", "level"},
            out_path=corrected_path,
        )
        work_wav = read_wav(corrected_path)
        repair_report = analyze(work_wav, cfg)
    elif do_correct and wav.channels == 1:
        logs.append("Correct skipped (mono)")

    repaired_path = None
    repair_meta = None
    if repair_mode != "off":
        tick(0.7, f"Repairing dropouts ({repair_mode})…")
        repaired_path = work / f"{wav_path.stem}.repaired.wav"
        result = apply_repairs(work_wav, repair_report, repaired_path, mode=repair_mode)  # type: ignore[arg-type]
        repair_meta = result.to_dict()
        write_json_obj(repair_meta, work / f"{wav_path.stem}.repair.json")

    # Copy original into work for convenient download bundle (read-only source preserved)
    original_copy = work / f"{wav_path.stem}.original.wav"
    if not original_copy.exists():
        shutil.copy2(wav_path, original_copy)

    tick(1.0, "Done")
    summary = {
        "work_dir": str(work),
        "source": str(wav_path),
        "sample_rate": wav.sample_rate,
        "channels": wav.channels,
        "duration_s": wav.samples.shape[0] / wav.sample_rate,
        "stereo_relationship": report.stereo_relationship,
        "event_count": len(report.events),
        "azimuth_us": None if az is None else az.lag_microseconds,
        "lr_db": None if lvl is None else lvl.lr_rms_diff_db,
        "repaired_count": None if repair_meta is None else repair_meta.get("repaired_count"),
        "original_wav": str(original_copy),
        "corrected_wav": str(corrected_path) if corrected_path else None,
        "repaired_wav": str(repaired_path) if repaired_path else None,
        "dropouts_json": str(work / f"{wav_path.stem}.dropouts.json"),
        "alignment_json": str(work / f"{wav_path.stem}.alignment.json"),
        "repair_json": str(work / f"{wav_path.stem}.repair.json") if repair_meta else None,
        "log": "\n".join(logs),
    }
    write_json_obj(summary, work / f"{wav_path.stem}.ui-summary.json")
    return summary


def _file_obj_path(file_obj: Any) -> Path | None:
    if file_obj is None:
        return None
    if isinstance(file_obj, (str, Path)):
        p = Path(file_obj)
        return p if p.suffix.lower() == ".wav" else None
    name = getattr(file_obj, "name", None) or getattr(file_obj, "path", None)
    if name:
        p = Path(str(name))
        return p if p.suffix.lower() == ".wav" else None
    return None


def _persist_upload(path: Path) -> Path:
    """Copy Gradio temp uploads into a stable stage so they survive until Processar lote."""
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    # Already outside Gradio temp — keep as-is
    tmp = Path(tempfile.gettempdir()).resolve()
    try:
        if tmp not in resolved.parents and resolved.parent != tmp:
            if resolved.is_file():
                return resolved
    except OSError:
        pass
    _UPLOAD_STAGE.mkdir(parents=True, exist_ok=True)
    dest = _UPLOAD_STAGE / resolved.name
    if dest.exists() and dest.stat().st_size == resolved.stat().st_size:
        return dest
    if dest.exists():
        stem, ext = dest.stem, dest.suffix
        n = 2
        while True:
            cand = _UPLOAD_STAGE / f"{stem}_{n}{ext}"
            if not cand.exists():
                dest = cand
                break
            n += 1
    shutil.copy2(resolved, dest)
    return dest


def _collect_wav_paths(
    files: Any,
    folder_path: str | None,
    *,
    persist_uploads: bool = True,
) -> list[Path]:
    """Merge multi-file upload + optional folder glob; stable order, unique paths."""
    found: list[Path] = []
    if files:
        items = files if isinstance(files, (list, tuple)) else [files]
        for item in items:
            p = _file_obj_path(item)
            if p is not None and p.is_file():
                found.append(_persist_upload(p) if persist_uploads else p.resolve())
    folder = (folder_path or "").strip().strip('"').strip("'")
    if folder:
        root = Path(folder).expanduser()
        if root.is_dir():
            found.extend(sorted(p.resolve() for p in root.rglob("*.wav") if p.is_file()))
        elif root.is_file() and root.suffix.lower() == ".wav":
            found.append(root.resolve())
    # unique, preserve order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        key = p.resolve() if p.exists() else p
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def build_queue_rows(files: Any, folder_path: str | None, artist_default: str = "") -> list[list[str]]:
    """Build dataframe rows: Nº | Título | Artista | Arquivo | Status."""
    paths = _collect_wav_paths(files, folder_path, persist_uploads=True)
    rows: list[list[str]] = []
    for i, p in enumerate(paths, start=1):
        rows.append(
            [
                f"{i:02d}",
                title_from_filename(p),
                (artist_default or "").strip(),
                str(p),
                "na fila",
            ]
        )
    return rows


def _resolve_batch_out_dir(out_folder: str, *, dated_subfolder: bool) -> Path:
    raw = (out_folder or "").strip().strip('"').strip("'")
    base = Path(raw or str(default_batch_out_root())).expanduser()
    if dated_subfolder:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = base / f"reparados_{stamp}"
    else:
        dest = base / "reparados"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Não foi possível criar a pasta de saída {dest}: {exc}. "
            "Escolha outra Pasta de saída (ex.: Documents\\mtdrop-saidas)."
        ) from exc
    return dest


def _unique_path(path: Path) -> Path:
    """Never overwrite: if exists, append _2, _3, …"""
    if not path.exists():
        return path
    stem, ext = path.stem, path.suffix
    n = 2
    while True:
        cand = path.with_name(f"{stem}_{n}{ext}")
        if not cand.exists():
            return cand
        n += 1


def _paths_text(disk_paths: list[str]) -> str:
    if not disk_paths:
        return "(nenhum ainda)"
    return "\n".join(disk_paths)


def process_batch(
    rows: list[list[Any]] | Any,
    *,
    out_folder: str,
    dated_subfolder: bool,
    do_correct: bool,
    repair_mode: str,
    sensitivity: str,
    skip_existing: bool = True,
    progress: Any = None,
) -> Iterator[tuple[list[list[str]], str, str, list[str], str]]:
    """Yield (table, status, disk_paths_text, gradio_files, log) per file.

    Critical (Windows / Gradio InvalidPathError):
    - Disk write under the user folder is the source of truth.
    - Mid-batch yields return **empty** ``gradio_files`` so Gradio postprocess never
      sees ``mtdrop-saidas\\reparados\\...`` and cannot abort the remaining rows.
    - Only the **final** yield may attach Gradio-safe temp copies for optional download.
    """
    empty_files: list[str] = []

    # Normalize dataframe input (pandas / list / numpy)
    if rows is None:
        yield [], "Fila vazia — carregue WAVs ou informe uma pasta.", "", empty_files, ""
        return
    if hasattr(rows, "values"):
        data = [list(r) for r in rows.values.tolist()]
    else:
        data = [list(r) for r in rows]
    if not data:
        yield [], "Fila vazia — carregue WAVs ou informe uma pasta.", "", empty_files, ""
        return

    try:
        dest = _resolve_batch_out_dir(out_folder, dated_subfolder=dated_subfolder)
    except Exception as exc:  # noqa: BLE001
        msg = _format_exc(exc)
        tb = traceback.format_exc()
        yield [], f"ERRO ao preparar pasta de saida: {msg}", "", empty_files, tb
        return

    disk_saved: list[str] = []
    staged_for_final: list[str] = []
    log_lines: list[str] = [
        f"Pasta de saida (disco): {dest}",
        "Cada faixa e gravada em disco imediatamente. O download da UI so aparece no fim "
        "(copias em pasta temp) — falha no download NAO cancela o lote.",
    ]
    total = len(data)
    ok = 0
    err = 0
    skipped = 0

    def _mid_yield(status: str) -> tuple[list[list[str]], str, str, list[str], str]:
        # Never pass user-folder paths to Gradio Files mid-batch.
        return (
            [list(r) for r in data],
            status,
            _paths_text(disk_saved),
            empty_files,
            "\n".join(log_lines),
        )

    for i, row in enumerate(data):
        while len(row) < 5:
            row.append("")
        num = format_index(row[0])
        data[i][0] = num
        title, artist, src_s = str(row[1]), str(row[2]), str(row[3])
        src = Path(src_s.strip().strip('"').strip("'")).expanduser()
        data[i][4] = "processando..."
        overall = (i / max(1, total)) * 0.95
        if progress is not None:
            try:
                progress(overall, desc=f"[{i+1}/{total}] {src.name}")
            except TypeError:
                progress(overall)
        yield _mid_yield(f"Processando {i+1}/{total}: {src.name}")

        if not src.is_file():
            msg = (
                f"arquivo nao encontrado: {src} "
                "(upload temporario do Gradio pode ter expirado — use Montar fila de novo "
                "ou informe o caminho da pasta no disco)"
            )
            data[i][4] = f"ERRO: {msg}"
            err += 1
            log_lines.append(f"ERRO [{num}] {msg}")
            yield _mid_yield(f"Erro em {i+1}/{total} — continuando…")
            continue

        out_name = batch_output_filename(num, title, artist, suffix="reparado", ext=".wav")
        planned = dest / out_name
        if skip_existing and planned.is_file():
            skipped += 1
            data[i][4] = f"PULADO (ja existe) → {planned}"
            disk_saved.append(str(planned))
            log_lines.append(f"PULADO [{num}] ja existe: {planned}")
            yield _mid_yield(
                f"Progresso {i+1}/{total} — OK={ok} · pulados={skipped} · erros={err}"
            )
            continue

        work: Path | None = None
        try:
            work = Path(tempfile.mkdtemp(prefix="mtdrop_batch_"))
            summary = _run_pipeline(
                src,
                do_correct=bool(do_correct),
                repair_mode=repair_mode,
                sensitivity=sensitivity,
                work_dir=work,
            )
            deliverable = summary.get("repaired_wav") or summary.get("corrected_wav")
            if not deliverable:
                raise RuntimeError("Nenhum WAV de saida (ligue repair ou correct)")

            out_path = planned if (skip_existing or not planned.exists()) else _unique_path(planned)
            try:
                shutil.copy2(deliverable, out_path)
            except OSError as exc:
                raise RuntimeError(
                    f"Falha ao gravar em disco {out_path}: {exc} "
                    "(pasta bloqueada / sem permissao / caminho longo?)"
                ) from exc

            stem_base = out_path.with_suffix("")
            for key, label in (
                ("dropouts_json", ".dropouts.json"),
                ("alignment_json", ".alignment.json"),
                ("repair_json", ".repair.json"),
            ):
                src_json = summary.get(key)
                if src_json and Path(src_json).is_file():
                    try:
                        shutil.copy2(src_json, Path(str(stem_base) + label))
                    except OSError as side_exc:
                        log_lines.append(f"aviso sidecar [{num}]: {side_exc}")

            ok += 1
            disk_saved.append(str(out_path))
            data[i][4] = f"OK → {out_path}"
            log_lines.append(
                f"OK [{num}] {src.name} → {out_path} "
                f"(events={summary.get('event_count')}, repaired={summary.get('repaired_count')})"
            )
            # Stage for *final* Gradio Files only — never mid-batch.
            staged = _stage_for_gradio_download(out_path)
            if staged is not None:
                staged_for_final.append(str(staged))
        except Exception as exc:  # noqa: BLE001 — continue batch
            err += 1
            msg = _format_exc(exc)
            data[i][4] = f"ERRO: {msg}"
            log_lines.append(f"ERRO [{num}] {src.name}: {msg}")
            log_lines.append(traceback.format_exc())
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)

        yield _mid_yield(
            f"Progresso {i+1}/{total} — OK={ok} · pulados={skipped} · erros={err} · pasta={dest}"
        )

    if progress is not None:
        try:
            progress(1.0, desc="Lote concluido")
        except TypeError:
            progress(1.0)
    final = (
        f"Lote concluido: {ok} OK, {skipped} pulado(s), {err} erro(s), total {total}. "
        f"Arquivos em disco: {dest}"
    )
    log_lines.append(final)
    # Final yield may include Gradio-safe temp copies. If Gradio still chokes, all
    # disk writes already finished — Helio still has the WAVs.
    yield (
        [list(r) for r in data],
        final,
        _paths_text(disk_saved),
        list(staged_for_final),
        "\n".join(log_lines),
    )


def build_app():
    import gradio as gr

    def run_single(file_obj, do_correct, repair_mode, sensitivity, progress=gr.Progress()):
        if file_obj is None:
            raise gr.Error("Load a WAV first")
        path = file_obj if isinstance(file_obj, str) else getattr(file_obj, "name", None) or str(file_obj)
        summary = _run_pipeline(
            path,
            do_correct=bool(do_correct),
            repair_mode=repair_mode,
            sensitivity=sensitivity,
            progress=progress,
        )
        if summary["azimuth_us"] is not None:
            status = (
                f"events={summary['event_count']} · {summary['stereo_relationship']} · "
                f"az={summary['azimuth_us']:.1f}µs · L−R={summary['lr_db']:+.2f} dB · "
                f"repaired={summary['repaired_count']}"
            )
        else:
            status = f"events={summary['event_count']} · repaired={summary['repaired_count']}"

        downloads = [
            summary["dropouts_json"],
            summary["alignment_json"],
        ]
        if summary["repair_json"]:
            downloads.append(summary["repair_json"])
        if summary["corrected_wav"]:
            downloads.append(summary["corrected_wav"])
        if summary["repaired_wav"]:
            downloads.append(summary["repaired_wav"])

        return (
            status,
            summary["log"],
            summary["original_wav"],
            summary["corrected_wav"],
            summary["repaired_wav"],
            downloads,
            json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=2),
        )

    def load_queue(files, folder, artist_default):
        rows = build_queue_rows(files, folder, artist_default or "")
        if not rows:
            return [], "Nenhum WAV encontrado."
        return rows, f"{len(rows)} arquivo(s) na fila."

    def run_batch(
        table,
        out_folder,
        dated,
        do_correct,
        repair_mode,
        sensitivity,
        skip_existing,
        progress=gr.Progress(),
    ):
        # Never let a Gradio download/path issue abort the generator mid-batch.
        try:
            yield from process_batch(
                table,
                out_folder=out_folder or "",
                dated_subfolder=bool(dated),
                do_correct=bool(do_correct),
                repair_mode=repair_mode,
                sensitivity=sensitivity,
                skip_existing=bool(skip_existing),
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            msg = _format_exc(exc)
            yield (
                table if table is not None else [],
                f"ERRO no lote (UI): {msg}. Se os WAVs ja aparecem na pasta reparados, ignore o download da UI.",
                "",
                [],
                tb,
            )

    default_out = str(default_batch_out_root())

    with gr.Blocks(title="mtdrop — tape dropout tool") as demo:
        gr.Markdown(
            """
# mtdrop — interface local / local listen UI

Pipeline: detect → **corrigido/corrected** (só azimuth + nível L/R) → **reparado/repaired** (resultado final: dropouts + estalos/clicks + tok/thump).

| Faixa / Player | PT | EN |
|---|---|---|
| **Original** | Arquivo de entrada (pode já ter estalos) | Source WAV (may already contain ticks/pops) |
| **Corrigido** | Só alinhamento (azimuth + ganho L/R) — **não** é o resultado final | Alignment only — **not** the final result |
| **Reparado** | **Resultado final** (alinhamento + dropouts + de-click/tok) | **Final output** (alignment + dropout + de-click) |

**Arquivo único:** reparo padrão = `conservative` (protege voz).  
**Lote:** reparo padrão = `aggressive` (Helio / Samba) — mude se precisar.  
O WAV de origem **nunca** é sobrescrito. Saídas mantêm taxa / bits / canais do input.
            """
        )

        with gr.Tabs():
            with gr.Tab("Arquivo único"):
                with gr.Row():
                    inp = gr.File(label="WAV (≤192 kHz / 24-bit, mono or stereo)", file_types=[".wav"])
                    with gr.Column():
                        do_correct = gr.Checkbox(
                            value=True,
                            label="Corrigir azimuth + nível L/R / Correct azimuth + L/R level",
                        )
                        repair_mode = gr.Radio(
                            choices=["conservative", "aggressive", "off"],
                            value="conservative",
                            label=(
                                "Reparo / Repair — conservative = protege voz/letra (padrão); "
                                "aggressive = mais invasivo"
                            ),
                        )
                        sensitivity = gr.Radio(
                            choices=["balanced", "aggressive", "conservative"],
                            value="balanced",
                            label="Sensibilidade / Sensitivity",
                        )
                        run_btn = gr.Button("Rodar mtdrop / Run", variant="primary")

                status = gr.Textbox(label="Resumo / Summary", interactive=False)
                log = gr.Textbox(label="Log", lines=4, interactive=False)

                gr.Markdown(
                    "### Ouça o arquivo inteiro / Full-file A–B\n"
                    "**Use REPARADO como resultado final.** Corrigido = só azimuth/nível."
                )
                with gr.Row():
                    aud_orig = gr.Audio(
                        label="1) Original (entrada / source)",
                        type="filepath",
                        interactive=False,
                    )
                    aud_corr = gr.Audio(
                        label="2) Corrigido / Corrected — só azimuth+nível (NÃO é o final)",
                        type="filepath",
                        interactive=False,
                    )
                    aud_rep = gr.Audio(
                        label="3) REPARADO / REPAIRED — resultado final ★",
                        type="filepath",
                        interactive=False,
                    )

                downloads = gr.Files(
                    label="Download — use o *.repaired.wav como resultado final / final deliverable"
                )
                summary_json = gr.Code(label="JSON do run", language="json")

                run_btn.click(
                    run_single,
                    inputs=[inp, do_correct, repair_mode, sensitivity],
                    outputs=[status, log, aud_orig, aud_corr, aud_rep, downloads, summary_json],
                )

            with gr.Tab("Lote / Batch"):
                gr.Markdown(
                    """
### Fila em lote
1. Selecione **vários WAVs** e/ou informe o caminho de uma **pasta**.
2. Clique **Montar fila** — edite **Nº**, **Título**, **Artista** na tabela.
3. Escolha a **pasta de saída** (os arquivos vão para `reparados/` dentro dela).
4. Clique **Processar lote** — cada arquivo é **salvo na hora** (não espera o lote inteiro).

Nome de saída: `01_[Nome da Musica] - [Artista] - reparado.wav`
                    """
                )
                with gr.Row():
                    batch_files = gr.File(
                        label="WAVs (múltiplos) / Multiple WAVs",
                        file_types=[".wav"],
                        file_count="multiple",
                    )
                    with gr.Column():
                        batch_folder = gr.Textbox(
                            label="Ou pasta com WAVs (caminho absoluto) / Or folder path",
                            placeholder=r"C:\Transfers\Samba  ou  /home/…/wavs",
                        )
                        batch_artist = gr.Textbox(
                            label="Artista padrão (preenche a fila) / Default artist",
                            placeholder="Nome do artista",
                        )
                        load_btn = gr.Button("Montar fila / Build queue", variant="secondary")

                batch_table = gr.Dataframe(
                    headers=["Nº", "Título", "Artista", "Arquivo", "Status"],
                    datatype=["str", "str", "str", "str", "str"],
                    column_count=(5, "fixed"),
                    label="Fila (editável) — Nº / Título / Artista",
                    interactive=True,
                    wrap=True,
                )
                batch_queue_status = gr.Textbox(label="Fila", interactive=False)

                with gr.Row():
                    batch_out = gr.Textbox(
                        label="Pasta de saída / Output folder",
                        value=default_out,
                        info="Cria subpasta reparados/ (ou reparados_AAAAMMDD_HHMMSS se marcado)",
                    )
                    batch_dated = gr.Checkbox(
                        value=False,
                        label="Subpasta datada / Dated subfolder (reparados_YYYYMMDD_HHMMSS)",
                    )
                    batch_skip = gr.Checkbox(
                        value=True,
                        label="Pular se ja existe 01_[…] - reparado.wav / Skip existing",
                    )

                with gr.Row():
                    batch_correct = gr.Checkbox(
                        value=True,
                        label="Corrigir azimuth + nível L/R",
                    )
                    batch_repair = gr.Radio(
                        choices=["aggressive", "conservative", "off"],
                        value="aggressive",
                        label="Reparo no lote (padrão aggressive — Samba/Helio)",
                    )
                    batch_sens = gr.Radio(
                        choices=["balanced", "aggressive", "conservative"],
                        value="balanced",
                        label="Sensibilidade",
                    )

                batch_run = gr.Button("Processar lote / Run batch", variant="primary")
                batch_status = gr.Textbox(label="Progresso do lote / Batch progress", interactive=False)
                batch_paths = gr.Textbox(
                    label="Arquivos salvos em disco (caminhos) / Saved on disk (paths)",
                    lines=6,
                    interactive=False,
                )
                batch_log = gr.Textbox(label="Log do lote", lines=8, interactive=False)
                batch_downloads = gr.Files(
                    label=(
                        "Download auxiliar no FIM do lote (copias temp). "
                        "O WAV real ja esta em reparados/ — se este painel falhar, ignore."
                    )
                )

                load_btn.click(
                    load_queue,
                    inputs=[batch_files, batch_folder, batch_artist],
                    outputs=[batch_table, batch_queue_status],
                )
                batch_run.click(
                    run_batch,
                    inputs=[
                        batch_table,
                        batch_out,
                        batch_dated,
                        batch_correct,
                        batch_repair,
                        batch_sens,
                        batch_skip,
                    ],
                    outputs=[batch_table, batch_status, batch_paths, batch_downloads, batch_log],
                )

    return demo


def _port_available(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def pick_listen_port(host: str = "127.0.0.1", preferred: int = 7860, span: int = 11) -> int:
    """Return preferred port, or the next free port in [preferred, preferred+span).

    ``preferred=0`` asks the OS for any free ephemeral port.
    """
    import socket

    if preferred == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    last = preferred + max(1, span)
    for port in range(preferred, last):
        if _port_available(host, port):
            return port
    raise OSError(
        f"Cannot find empty port in range: {preferred}-{last - 1}. "
        "Feche outra janela do mtdrop/Gradio ou use: mtdrop ui --port 0"
    )


def launch(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
    *,
    port_span: int = 11,
) -> int:
    """Launch Gradio UI. Returns the port actually bound."""
    chosen = pick_listen_port(host, preferred=port, span=port_span)
    if chosen != port and port != 0:
        print(
            f"Porta {port} ocupada — usando {chosen} em vez disso.\n"
            f"Port {port} busy — using {chosen} instead.",
            flush=True,
        )
    url = f"http://{host}:{chosen}"
    print(
        "\n"
        "============================================\n"
        f"  mtdrop UI → {url}\n"
        "============================================\n"
        "Abra este URL no navegador / Open this URL in your browser.\n"
        "Aba Lote: selecione vários WAVs → Montar fila → Processar lote.\n",
        flush=True,
    )
    demo = build_app()
    allowed = gradio_allowed_paths()
    # Also allow the configured default out root's parent tree for custom subfolders.
    print(f"Gradio allowed_paths: {allowed}", flush=True)
    demo.queue().launch(
        server_name=host,
        server_port=chosen,
        share=share,
        show_error=True,
        inbrowser=False,
        allowed_paths=allowed,
    )
    return chosen
