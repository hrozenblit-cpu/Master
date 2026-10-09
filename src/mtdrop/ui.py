"""Local Gradio UI for full-file mtdrop listen / A-B testing.

Never overwrites the uploaded source — all outputs go to a temp work dir.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from mtdrop.align import apply_corrections, measure_azimuth, measure_level
from mtdrop.detect import analyze, config_for_sensitivity
from mtdrop.export import write_json_obj, write_report_bundle
from mtdrop.repair import apply_repairs
from mtdrop.wav_io import read_wav


def _run_pipeline(
    wav_path: str | Path,
    *,
    do_correct: bool,
    repair_mode: str,
    sensitivity: str,
    progress: Any = None,
) -> dict[str, Any]:
    wav_path = Path(wav_path)
    work = Path(tempfile.mkdtemp(prefix="mtdrop_ui_"))
    logs: list[str] = []

    def tick(frac: float, msg: str) -> None:
        logs.append(msg)
        if progress is not None:
            progress(frac, desc=msg)

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


def build_app():
    import gradio as gr

    def run(file_obj, do_correct, repair_mode, sensitivity, progress=gr.Progress()):
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

    with gr.Blocks(title="mtdrop — tape dropout tool") as demo:
        gr.Markdown(
            """
# mtdrop — interface local / local listen UI

Pipeline: detect → **corrigido/corrected** (só azimuth + nível L/R) → **reparado/repaired** (resultado final: dropouts + estalos/clicks da fonte).

| Faixa / Player | PT | EN |
|---|---|---|
| **Original** | Arquivo de entrada (pode já ter estalos) | Source WAV (may already contain ticks/pops) |
| **Corrigido** | Só alinhamento (azimuth + ganho L/R) — **não** é o resultado final | Alignment only — **not** the final result |
| **Reparado** | **Resultado final** (alinhamento + dropouts + de-click) | **Final output** (alignment + dropout + de-click) |

**Reparo padrão = conservative (protege voz/letra).** Não substitui sílabas com fill inventado. Use `aggressive` só se o dropout for grave e você aceitar mais invasão.

O WAV de origem **nunca** é sobrescrito. Saídas mantêm taxa / bits / canais do input.
            """
        )
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
            run,
            inputs=[inp, do_correct, repair_mode, sensitivity],
            outputs=[status, log, aud_orig, aud_corr, aud_rep, downloads, summary_json],
        )

    return demo


def launch(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    demo = build_app()
    demo.queue().launch(server_name=host, server_port=port, share=share, show_error=True)
