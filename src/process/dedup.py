"""Deduplication: exact (normalized-hash) + near-duplicate (MinHash LSH).

The MinHash signature of a document is independent of every other document, so
that step — the expensive one — is computed across a process pool. The LSH
insert/query is stateful and stays serial, but it is cheap once signatures exist.
"""
from __future__ import annotations

import hashlib
import os
import re
from multiprocessing import Pool

from datasketch import MinHash, MinHashLSH

_TOKEN_RE = re.compile(r"\w+")

# Set in the parent before the pool forks; worker processes inherit these via
# copy-on-write (Linux fork) and read them by index — so the ~GB of document
# text is never pickled across the process boundary.
_TEXTS: list[str] = []
_NUM_PERM = 128
_SHINGLE_K = 5


def exact_key(text: str) -> str:
    """Hash of the whitespace-collapsed lowercased text for exact dedup."""
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _minhash(text: str, num_perm: int, k: int) -> MinHash:
    """MinHash over word k-shingles (5-grams)."""
    toks = _TOKEN_RE.findall(text.lower())
    m = MinHash(num_perm=num_perm)
    if len(toks) < k:
        shingles = {" ".join(toks)} if toks else set()
    else:
        shingles = {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def _sig(i: int):
    """Worker: return just the MinHash hashvalues for document index i."""
    return _minhash(_TEXTS[i], _NUM_PERM, _SHINGLE_K).hashvalues


def dedup(
    docs: list[dict],
    *,
    threshold: float = 0.7,
    num_perm: int = 128,
    shingle_k: int = 5,
    workers: int | None = None,
) -> tuple[list[dict], dict]:
    """Return (kept_docs, stats).

    Docs are processed longest-first so that when near-duplicates exist we keep
    the most complete version. Exact duplicates are removed first (cheap).
    """
    global _TEXTS, _NUM_PERM, _SHINGLE_K
    stats = {"input": len(docs), "exact_dups": 0, "near_dups": 0}

    # 1) exact dedup (cheap, serial).
    seen_exact: set[str] = set()
    unique: list[dict] = []
    for d in docs:
        k = exact_key(d["text"])
        if k in seen_exact:
            stats["exact_dups"] += 1
            continue
        seen_exact.add(k)
        unique.append(d)

    # 2) MinHash signatures — parallel. Sort longest-first first so signature
    #    order matches the insertion order below.
    unique.sort(key=lambda d: len(d["text"]), reverse=True)
    _TEXTS = [d["text"] for d in unique]
    _NUM_PERM, _SHINGLE_K = num_perm, shingle_k
    n = len(unique)
    workers = workers or min(32, (os.cpu_count() or 2))
    if workers > 1 and n >= 2000:
        with Pool(workers) as pool:                      # forks: inherits _TEXTS
            sigs = pool.map(_sig, range(n), chunksize=256)
    else:
        sigs = [_sig(i) for i in range(n)]
    _TEXTS = []                                          # release the text copy

    # 3) LSH insert/query — serial, but fast (dict ops on precomputed signatures).
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept: list[dict] = []
    for i, d in enumerate(unique):
        m = MinHash(num_perm=num_perm, hashvalues=sigs[i])
        if lsh.query(m):
            stats["near_dups"] += 1
            continue
        lsh.insert(str(i), m)
        kept.append(d)

    stats["kept"] = len(kept)
    return kept, stats
