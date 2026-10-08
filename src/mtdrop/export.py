from __future__ import annotations

import csv
import json
from pathlib import Path

from mtdrop.models import AnalysisReport


def write_report_bundle(report: AnalysisReport, out_dir: Path, stem: str | None = None) -> dict[str, Path]:
    """Write JSON + CSV + Audacity labels. Never touches the source WAV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = stem or Path(report.source).stem
    paths = {
        "json": out_dir / f"{stem}.dropouts.json",
        "csv": out_dir / f"{stem}.dropouts.csv",
        "txt": out_dir / f"{stem}.dropouts.txt",
    }
    write_json(report, paths["json"])
    write_csv(report, paths["csv"])
    write_audacity_labels(report, paths["txt"])
    return paths


def write_json(report: AnalysisReport, path: Path) -> None:
    write_json_obj(report.to_dict(), path)


def write_json_obj(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def write_csv(report: AnalysisReport, path: Path) -> None:
    fields = [
        "start_s",
        "end_s",
        "duration_s",
        "start_sample",
        "end_sample",
        "channel",
        "type",
        "severity",
        "confidence",
        "duration_class",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ev in report.events:
            row = ev.to_dict()
            writer.writerow({k: row[k] for k in fields})


def write_audacity_labels(report: AnalysisReport, path: Path) -> None:
    """Audacity label track: start\\tend\\tlabel per line."""
    lines: list[str] = []
    for ev in report.events:
        label = f"{ev.type}|ch={ev.channel}|sev={ev.severity:.2f}|{ev.duration_class}"
        lines.append(f"{ev.start_s:.6f}\t{ev.end_s:.6f}\t{label}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
