"""Text normalisation helpers used for matching names and verifying quotes."""

from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_HONORIFICS = {
    "the",
    "hon",
    "honourable",
    "mr",
    "mrs",
    "ms",
    "miss",
    "dr",
    "prof",
    "professor",
    "sir",
    "dame",
    "senator",
    "mp",
    "mlc",
    "mla",
    "ao",
    "ac",
    "am",
    "oam",
    "qc",
    "kc",
    "sc",
}
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'‘“(])")
_QUOTE_FOLD = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", "\u00a0": " "})


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def fold_quote_text(text: str) -> str:
    """Normalise typography and whitespace so verbatim checks ignore curly quotes and line breaks."""
    return normalize_whitespace(unicodedata.normalize("NFKC", text).translate(_QUOTE_FOLD))


def is_verbatim(quote: str, passage_text: str) -> bool:
    folded = fold_quote_text(quote)
    return bool(folded) and folded in fold_quote_text(passage_text)


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def name_tokens(name: str) -> list[str]:
    """Lowercase, accent-free name tokens without honorifics, e.g. 'PLIBERSEK, the Hon. Tanya' -> ['plibersek', 'tanya']."""
    cleaned = strip_accents(name.translate(_QUOTE_FOLD)).lower()
    cleaned = re.sub(r"[^a-z' -]", " ", cleaned).replace("'", "")
    return [t for t in cleaned.replace("-", " ").split() if t and t not in _HONORIFICS]


def normalize_name(name: str) -> str:
    return " ".join(name_tokens(name))


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(normalize_whitespace(text)) if s.strip()]
