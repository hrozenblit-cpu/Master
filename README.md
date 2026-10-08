# Master Tools — `mtdrop`

Offline tool for **magnetic-tape audio transfers** (Ampex → Studer A80 and similar).

**Resolve/fix** path: multi-class dropout detection, **azimuth** + **L/R level** correction, and **dropout repair** onto derived WAVs. Source masters are **never overwritten**.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,ui]"
```

Requires libsndfile (`soundfile`). Debian/Ubuntu: `sudo apt-get install -y libsndfile1`.  
UI also needs Gradio (`[ui]` extra). Optional loudnorm MP3 previews need `ffmpeg`.

## Quick start

```bash
# Synthetic fixture (dropouts + ~3-sample R lag + R −2.5 dB)
mtdrop make-fixture --out tests/fixtures/synth_dropouts_48k_stereo.wav

# Detect + measure
mtdrop analyze tests/fixtures/synth_dropouts_48k_stereo.wav --out ./reports

# Correct azimuth/level + repair dropouts (derived WAVs only)
mtdrop analyze tests/fixtures/synth_dropouts_48k_stereo.wav --out ./reports \
  --correct azimuth,level --repair conservative

# Batch
mtdrop analyze /path/to/transfers/ --out ./reports --repair conservative
```

`preview` allows slightly longer repair durations than `conservative`.

Detection sensitivity presets:

```bash
mtdrop analyze input.wav --out ./out --sensitivity balanced   # default
mtdrop analyze input.wav --out ./out --sensitivity conservative
mtdrop analyze input.wav --out ./out --sensitivity aggressive  # mild dips; more FPs
```

### Local listen UI (full-file A/B)

```bash
pip install -e ".[ui]"
mtdrop ui
# open http://127.0.0.1:7860
# or: mtdrop ui --host 127.0.0.1 --port 7860
```

Upload a WAV → toggle correct / repair / sensitivity → play **full** original vs corrected vs repaired in the browser → download derived WAVs + JSON. Source file is never overwritten.

### Listen previews (≥2.5 s — do not ship sub-second clips)

```bash
# During analyze (top events → WAV + loudnorm MP3 under --out)
mtdrop analyze input.wav --out ./out --repair conservative --listen-previews

# Or standalone
mtdrop preview \
  --wav original=input.wav --wav repaired=out/input.repaired.wav \
  --events out/input.dropouts.json \
  --out ./out --stem input --min-duration 2.5
```

## Outputs (under `--out` only)

| File | Phase | Use |
|---|---|---|
| `*.dropouts.json` / `.csv` / `.txt` | A | Markers for RX / CEDAR / Audacity / Reaper |
| `*.alignment.json` | A/B | Azimuth lag, L/R level, applied corrections |
| `*.corrected.wav` | B | After `--correct` |
| `*.repaired.wav` | C | After `--repair` (actual audio heal) |
| `*.repair.json` | C | Provenance of strategies applied / deferred |
| `*.repaired.txt` | C | Audacity labels for repaired regions |

## Repair methods (Phase C)

- **Cross-channel borrow** when the other channel is clean (dual-mono or true stereo)
- **Mirror / edge interpolation** for short dips & mutes
- **STFT fill** for longer gaps
- **HF-band reconstruct** for tape-clog (keep LF, rebuild HF)

**Calibration note:** defaults tuned on the synthetic fixture **and** Helio `exemplo_1_18` (full-track mono → A80 two-track). `dual_mono_like` prefers cross-channel borrow. **Not production-ready** — more reels + listen sign-off needed.

## Supported input

- WAV PCM, mono or stereo, ≤192 kHz, ≤24-bit

## Tests

```bash
pytest -q
```

## Roadmap status

| Phase | Status |
|---|---|
| A — detect + markers + azimuth/level measure | Ships |
| B — gated `--correct` → derived WAV | Ships |
| C — gated `--repair` → derived repaired WAV | Ships (conservative) |
| Local Gradio listen UI (`mtdrop ui`) | Ships (single-file); batch-folder UI later |
| Calibration on Helio full-track→two-track clips | First pass done; more A80 reels needed |
