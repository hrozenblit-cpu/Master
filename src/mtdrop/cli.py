from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mtdrop import __version__
from mtdrop.align import AlignmentReport, apply_corrections, measure_azimuth, measure_level
from mtdrop.detect import DetectConfig, analyze
from mtdrop.export import write_json_obj, write_report_bundle
from mtdrop.fixture import synthesize_dropout_wav
from mtdrop.repair import plan_repairs
from mtdrop.wav_io import read_wav


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mtdrop",
        description=(
            "Master Tools — tape dropout detect/measure/correct. "
            "Never overwrites source masters; writes derived WAVs + reports under --out."
        ),
    )
    parser.add_argument("--version", action="version", version=f"mtdrop {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze_p = sub.add_parser(
        "analyze",
        help="Detect dropouts, measure azimuth/level, optional gated correction (Phase A+B)",
    )
    _add_analyze_args(analyze_p)

    detect_p = sub.add_parser(
        "detect",
        help="Alias of analyze without correction (Phase A markers only)",
    )
    _add_analyze_args(detect_p)

    fix_p = sub.add_parser("make-fixture", help="Write a synthetic WAV with injected dropouts + skew")
    fix_p.add_argument("--out", type=Path, required=True, help="Destination .wav path")
    fix_p.add_argument("--sr", type=int, default=48000, help="Sample rate (default 48000)")
    fix_p.add_argument("--mono", action="store_true", help="Mono instead of stereo")
    fix_p.add_argument("--duration", type=float, default=2.0, help="Duration in seconds")
    fix_p.add_argument("--azimuth-lag-samples", type=float, default=3.0, help="Inject R lag (stereo)")
    fix_p.add_argument("--level-db", type=float, default=-2.5, help="Inject R level offset dB (stereo)")

    args = parser.parse_args(argv)

    if args.command == "make-fixture":
        meta = synthesize_dropout_wav(
            args.out,
            sample_rate=args.sr,
            duration_s=args.duration,
            stereo=not args.mono,
            azimuth_lag_samples=0.0 if args.mono else args.azimuth_lag_samples,
            level_offset_db_r=0.0 if args.mono else args.level_db,
        )
        print(json.dumps(meta, indent=2))
        return 0

    if args.command in {"analyze", "detect"}:
        if args.command == "detect":
            # Force no correction / repair for detect alias unless user passed flags
            # (flags still honored if explicitly set on detect).
            pass
        return _cmd_analyze(args)

    parser.error(f"unknown command {args.command}")
    return 2


def _add_analyze_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="WAV file(s) and/or directories of WAVs (batch)",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for reports / derived WAVs (created if missing). Sources never modified.",
    )
    p.add_argument("--min-duration-ms", type=float, default=3.0, help="Minimum dropout event length (ms)")
    p.add_argument("--severity-threshold", type=float, default=0.15, help="Drop events below this severity")
    p.add_argument("--dip-ratio", type=float, default=0.35, help="RMS / baseline ratio for level dips")
    p.add_argument(
        "--correct",
        type=str,
        default="",
        help="Comma list: azimuth,level — write derived *.corrected.wav (Phase B, gated)",
    )
    p.add_argument(
        "--lag-samples",
        type=float,
        default=None,
        help="Override azimuth lag (samples) when applying correction",
    )
    p.add_argument(
        "--max-lag-ms",
        type=float,
        default=2.0,
        help="Max azimuth search window in milliseconds",
    )
    p.add_argument(
        "--repair",
        choices=["off", "conservative", "preview"],
        default="off",
        help="Phase C: plan dropout repairs (audio apply not yet implemented)",
    )
    p.add_argument("--quiet", action="store_true", help="Less console output")


def _cmd_analyze(args: argparse.Namespace) -> int:
    wavs = _collect_wavs(args.inputs)
    if not wavs:
        print("error: no WAV files found", file=sys.stderr)
        return 1

    correct = {c.strip().lower() for c in args.correct.split(",") if c.strip()}
    unknown = correct - {"azimuth", "level"}
    if unknown:
        print(f"error: unknown --correct values: {sorted(unknown)}", file=sys.stderr)
        return 2

    cfg = DetectConfig(
        min_duration_s=max(0.0, args.min_duration_ms / 1000.0),
        severity_threshold=args.severity_threshold,
        dip_ratio=args.dip_ratio,
    )
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    total_events = 0
    for wav_path in wavs:
        try:
            wav = read_wav(wav_path)
            report = analyze(wav, cfg)
            paths = write_report_bundle(report, out_dir)
            total_events += len(report.events)

            az = measure_azimuth(wav, max_lag_ms=args.max_lag_ms) if wav.channels == 2 else None
            lvl = measure_level(wav) if wav.channels == 2 else None

            applied: dict = {"azimuth": None, "level": None}
            out_wav_path = None
            if correct and wav.channels == 2:
                out_wav_path = out_dir / f"{wav_path.stem}.corrected.wav"
                applied = apply_corrections(
                    wav,
                    azimuth=az,
                    level=lvl,
                    correct=correct,
                    out_path=out_wav_path,
                    lag_override=args.lag_samples,
                )
            elif correct and wav.channels == 1:
                print(f"warning: {wav_path}: --correct ignored for mono", file=sys.stderr)

            align = AlignmentReport(
                source=str(wav.path),
                sample_rate=wav.sample_rate,
                channels=wav.channels,
                azimuth=az,
                level=lvl,
                stereo_relationship=report.stereo_relationship,
                channel_correlation=report.channel_correlation,
                applied=applied,
                output_wav=str(out_wav_path) if out_wav_path else None,
            )
            align_path = out_dir / f"{wav_path.stem}.alignment.json"
            write_json_obj(align.to_dict(), align_path)

            repair_info = None
            if args.repair != "off":
                plan = plan_repairs(report, mode=args.repair)
                repair_path = out_dir / f"{wav_path.stem}.repair-plan.json"
                write_json_obj(plan.to_dict(), repair_path)
                repair_info = repair_path.name

            if not args.quiet:
                az_s = f"lag={az.lag_samples:.2f}sa ({az.lag_microseconds:.1f}µs)" if az else "n/a"
                lv_s = f"L−R={lvl.lr_rms_diff_db:+.2f} dB" if lvl else "n/a"
                extra = f", corrected={out_wav_path.name}" if out_wav_path else ""
                if repair_info:
                    extra += f", repair_plan={repair_info}"
                print(
                    f"{wav_path}: {len(report.events)} dropout(s), azimuth[{az_s}], level[{lv_s}] "
                    f"-> {paths['json'].name}, {align_path.name}{extra}"
                )
        except Exception as exc:  # noqa: BLE001 — CLI boundary
            failures += 1
            print(f"error: {wav_path}: {exc}", file=sys.stderr)

    if not args.quiet:
        print(
            f"done: {len(wavs) - failures}/{len(wavs)} file(s), {total_events} dropout event(s), out={out_dir}"
        )
    return 1 if failures else 0


def _collect_wavs(inputs: list[Path]) -> list[Path]:
    found: list[Path] = []
    for item in inputs:
        if item.is_dir():
            found.extend(sorted(p for p in item.rglob("*") if p.suffix.lower() == ".wav"))
        elif item.is_file() and item.suffix.lower() == ".wav":
            found.append(item)
        else:
            print(f"warning: skipping non-WAV path {item}", file=sys.stderr)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)
    return unique


if __name__ == "__main__":
    raise SystemExit(main())
