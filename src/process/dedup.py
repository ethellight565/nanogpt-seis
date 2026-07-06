"""Deduplication: exact (normalized-hash) + near-duplicate (MinHash LSH)."""
from __future__ import annotations

import hashlib
import re

from datasketch import MinHash, MinHashLSH

_TOKEN_RE = re.compile(r"\w+")


def exact_key(text: str) -> str:
    """Hash of the whitespace-collapsed lowercased text for exact dedup."""
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _minhash(text: str, num_perm: int, k: int) -> MinHash:
    """MinHash over word k-shingles."""
    toks = _TOKEN_RE.findall(text.lower())
    m = MinHash(num_perm=num_perm)
    if len(toks) < k:
        shingles = {" ".join(toks)} if toks else set()
    else:
        shingles = {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def dedup(
    docs: list[dict],
    *,
    threshold: float = 0.7,
    num_perm: int = 128,
    shingle_k: int = 5,
) -> tuple[list[dict], dict]:
    """Return (kept_docs, stats).

    Docs are processed longest-first so that when near-duplicates exist we keep
    the most complete version. Exact duplicates are removed first (cheap).
    """
    stats = {"input": len(docs), "exact_dups": 0, "near_dups": 0}

    # Exact dedup.
    seen_exact: set[str] = set()
    unique: list[dict] = []
    for d in docs:
        k = exact_key(d["text"])
        if k in seen_exact:
            stats["exact_dups"] += 1
            continue
        seen_exact.add(k)
        unique.append(d)

    # Near-dup via LSH, longest first.
    unique.sort(key=lambda d: len(d["text"]), reverse=True)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    for i, d in enumerate(unique):
        m = _minhash(d["text"], num_perm, shingle_k)
        if lsh.query(m):
            stats["near_dups"] += 1
            continue
        lsh.insert(str(i), m)
        kept.append(d)

    stats["kept"] = len(kept)
    return kept, stats
