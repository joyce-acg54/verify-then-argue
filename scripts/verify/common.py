"""Shared helpers for the verification CLIs (deck selection, page
segmentation, hashing, paths). Not vendored — written for this project."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from dotenv import load_dotenv

VERIFY_DIR = Path(__file__).resolve().parent
REPO_ROOT = VERIFY_DIR.parents[1]
DOCS_DIR = REPO_ROOT / "data" / "documents"
CLAIMS_DIR = REPO_ROOT / "data" / "claims"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "verify"

# Filename prefix of email-export wrapper files (never decks). Set the env var
# to match your corpus's email-export naming convention.
EMAIL_WRAPPER_PREFIX = os.environ.get("VERIFY_EMAIL_WRAPPER_PREFIX",
                                      "EMAIL-WRAPPER")
MIN_DECK_CHARS = 4000

load_dotenv(REPO_ROOT / ".env")


def pick_deck(account_id: str) -> tuple[Path, str] | None:
    """Largest parsed .txt that is not an email wrapper and has >= 4000 chars.
    Returns (path, text) or None if the account has no usable deck."""
    parsed = DOCS_DIR / account_id / "parsed"
    if not parsed.is_dir():
        return None
    candidates = [p for p in parsed.glob("*.txt")
                  if not p.name.startswith(EMAIL_WRAPPER_PREFIX)]
    best: tuple[Path, str] | None = None
    for p in sorted(candidates, key=lambda p: p.stat().st_size, reverse=True):
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) >= MIN_DECK_CHARS:
            best = (p, text)
            break
    return best


def segment_pages(text: str, target_chars: int = 2800) -> list[str]:
    """Split parsed deck text into pseudo-pages.

    The deck extractor (internal ingestion tooling, not part of this release)
    joins PyMuPDF page texts with a
    single newline; each page's own text ends with a newline, so page
    boundaries show up as blank lines. We split on blank lines and greedily
    merge the resulting blocks up to ~target_chars so each unit stays within
    the L1 prompt windows.
    """
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    pages: list[str] = []
    current = ""
    for b in blocks:
        if current and len(current) + len(b) + 2 > target_chars:
            pages.append(current)
            current = b
        else:
            current = f"{current}\n\n{b}" if current else b
    if current:
        pages.append(current)
    return pages


def claim_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()[:16]


def claims_raw_path(account_id: str) -> Path:
    return CACHE_DIR / f"{account_id}_claims_raw.json"


def verify_jsonl_path(account_id: str) -> Path:
    return CACHE_DIR / f"{account_id}_verify.jsonl"
