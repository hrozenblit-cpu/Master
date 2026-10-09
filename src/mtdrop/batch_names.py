"""Batch output naming helpers for the Gradio UI (Windows-safe filenames)."""

from __future__ import annotations

import re
from pathlib import Path

# Windows-illegal filename characters + control chars.
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WS = re.compile(r"\s+")


def sanitize_filename_part(text: str, *, fallback: str = "sem_titulo") -> str:
    """Strip illegal Windows chars; collapse whitespace; keep readable spacing."""
    s = (text or "").strip()
    s = _ILLEGAL.sub("", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = _WS.sub(" ", s).strip(" .")
    # Windows reserved device names
    if s.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        s = f"_{s}"
    return s or fallback


def title_from_filename(path: str | Path) -> str:
    """Prefill song title from WAV stem (drop common transfer suffixes when obvious)."""
    stem = Path(path).stem
    # Soft clean: collapse underscores/dashes used as spaces in transfers
    soft = stem.replace("_", " ")
    soft = _WS.sub(" ", soft).strip()
    return soft or stem or "sem_titulo"


def format_index(n: int | str) -> str:
    """Zero-pad track index to 2 digits (01, 02, …). Accepts override strings."""
    if isinstance(n, str):
        s = n.strip()
        if s.isdigit():
            return f"{int(s):02d}"
        # keep user override if non-numeric but sanitize
        return sanitize_filename_part(s, fallback="00")[:8] or "00"
    return f"{int(n):02d}"


def batch_output_filename(
    index: int | str,
    title: str,
    artist: str,
    *,
    suffix: str = "reparado",
    ext: str = ".wav",
) -> str:
    """``01_[Nome da Musica] - [Artista] - reparado.wav`` (exact spirit)."""
    num = format_index(index)
    tit = sanitize_filename_part(title, fallback="sem_titulo")
    art = sanitize_filename_part(artist, fallback="sem_artista")
    if not ext.startswith("."):
        ext = f".{ext}"
    # Pattern: 01_[Nome da Musica] - [Artista] - reparado.wav
    name = f"{num}_[{tit}] - [{art}] - {suffix}{ext}"
    # Final pass: no residual illegals; cap length for Windows MAX_PATH comfort
    name = _ILLEGAL.sub("", name)
    if len(name) > 180:
        # Truncate title/artist sections proportionally
        budget = 180 - len(f"{num}_[] - [] - {suffix}{ext}")
        t_budget = max(8, budget * 2 // 3)
        a_budget = max(8, budget - t_budget)
        tit = tit[:t_budget].rstrip(" .")
        art = art[:a_budget].rstrip(" .")
        name = f"{num}_[{tit}] - [{art}] - {suffix}{ext}"
    return name
