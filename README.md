# Master Tools — `mtdrop`

Offline tool for **magnetic-tape audio transfers** (Ampex → Studer A80 and similar).

**End state is resolve/fix**, not detect-only: multi-class dropout detection, **azimuth** (L/R time) estimate + correction, **channel level** measure + correction, and a scaffolded path to dropout repair. Source masters are **never overwritten** — derived WAVs and reports go under `--out`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires libsndfile (`soundfile`). Debian/Ubuntu: `sudo apt-get install -y libsndfile1`.

## Quick start

```bash
# Synthetic fixture (dropouts + ~3-sample R lag + R −2.5 dB)
mtdrop make-fixture --out tests/fixtures/synth_dropouts_48k_stereo.wav

# Phase A — detect dropouts + measure azimuth/level
mtdrop analyze tests/fixtures/synth_dropouts_48k_stereo.wav --out ./reports

# Phase B — also write derived corrected WAV (gated)
mtdrop analyze tests/fixtures/synth_dropouts_48k_stereo.wav --out ./reports \
  --correct azimuth,level

# Batch a folder
mtdrop analyze /path/to/transfers/ --out ./reports --correct azimuth,level

# Phase C scaffold — repair plan JSON only (no audio heal yet)
mtdrop analyze input.wav --out ./reports --repair conservative
```

`mtdrop detect …` is an alias of `analyze` (same flags).

## Outputs (under `--out` only)

| File | Phase | Use |
|---|---|---|
| `*.dropouts.json` / `.csv` / `.txt` | A | Markers for RX / CEDAR / Audacity / Reaper |
| `*.alignment.json` | A/B | Azimuth lag, L/R level, applied corrections, provenance |
| `*.corrected.wav` | B | Derived WAV after `--correct` (optional) |
| `*.repair-plan.json` | C | Planned heal strategies (`--repair`; audio apply TBD) |

### Dropout event fields

`start`/`end` (seconds + samples), **per-channel** (`L`/`R`/`mono`) and joint (`both`) tags, type (`level_dip`, `hard_mute`, `hf_loss`, `channel_asymmetry`), severity, confidence, duration class. Stereo never assumes L==R; `stereo_relationship` is a correlation hint only (`dual_mono_like` / `true_stereo` / `mono`).

### Alignment fields

- **Azimuth:** `lag_samples`, `lag_microseconds` (positive ⇒ right lags left)
- **Level:** per-channel RMS/peak dB, L−R difference, suggested gains (match mid-RMS)

## Supported input (v0.1)

- WAV PCM, mono or stereo
- Up to 192 kHz, up to 24-bit PCM

## Tests

```bash
pytest -q
```

## Roadmap

| Phase | Status |
|---|---|
| A — detect + markers + azimuth/level measure | **Ships now** |
| B — gated `--correct` → derived WAV + provenance | **Ships now** |
| C — dropout repair apply (interp / spectral / cross-channel) | Scaffold + plan only; heal next |

See project plan: tape-dropout-tool-plan (Master Tools docs store).
