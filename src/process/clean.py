"""Text cleaning & filtering for the raw crawled documents.

Cleaning is source-aware: arXiv full-text PDFs need de-hyphenation and
reference-section stripping, while Wikipedia / OpenAlex text is already fairly
clean and only needs entity-unescaping and whitespace normalization.
"""
from __future__ import annotations

import html
import re
import unicodedata

# --- English detection via stopword ratio -----------------------------------
# English scientific prose sits well above this; Spanish/other-language text
# (e.g. Substack's translated posts) falls far below.
_EN_STOP = {
    "the", "of", "and", "to", "in", "a", "is", "that", "for", "was", "on",
    "are", "with", "as", "at", "by", "an", "be", "this", "which", "or", "from",
    "we", "it", "have", "has", "were", "these", "our", "not", "can", "but",
}
_WORD_RE = re.compile(r"[a-z]+")


def english_ratio(text: str) -> float:
    toks = _WORD_RE.findall(text.lower())
    if not toks:
        return 0.0
    return sum(t in _EN_STOP for t in toks) / len(toks)


# --- normalization ----------------------------------------------------------
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")   # keep \t \n \r
_MULTISPACE_RE = re.compile(r"[ \t]{2,}")
_MULTINL_RE = re.compile(r"\n{3,}")
_DEHYPHEN_RE = re.compile(r"(\w)-\n(\w)")                 # line-break hyphenation


def normalize(text: str) -> str:
    text = html.unescape(text)               # &#13; &amp; -> real chars
    text = unicodedata.normalize("NFKC", text)   # ligatures, full-width, etc.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_RE.sub("", text)
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINL_RE.sub("\n\n", text)
    # strip trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


# --- arXiv PDF specific -----------------------------------------------------
# Truncate at the reference/bibliography section (last matching heading).
_REF_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|acknowledge?ments)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def strip_pdf_cruft(text: str) -> str:
    text = _DEHYPHEN_RE.sub(r"\1\2", text)
    # Drop everything from the last "References"/"Bibliography" heading onward.
    matches = list(_REF_HEADING_RE.finditer(text))
    if matches:
        text = text[: matches[-1].start()]
    # Remove standalone page-number / very-short noise lines.
    kept = [
        ln for ln in text.split("\n")
        if not re.fullmatch(r"\s*\d{1,4}\s*", ln)
    ]
    return "\n".join(kept)


def clean_doc(source: str, text: str) -> str:
    if source == "arxiv":
        text = strip_pdf_cruft(text)
    return normalize(text)


# --- quality filter ---------------------------------------------------------
def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(c.isalpha() or c.isspace() for c in text)
    return letters / len(text)


def passes_filters(
    text: str,
    *,
    min_chars: int = 200,
    min_english: float = 0.05,
    min_alpha: float = 0.6,
) -> tuple[bool, str]:
    """Return (ok, reason). reason is '' when ok."""
    if len(text) < min_chars:
        return False, "too_short"
    if alpha_ratio(text) < min_alpha:
        return False, "low_alpha"        # garbled PDF / tables / math dumps
    if english_ratio(text) < min_english:
        return False, "non_english"
    return True, ""
