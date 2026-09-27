"""I/O helpers — TSV streaming, country partitioning, deterministic writes.

Designed to keep memory bounded: we scan each source once per split, holding only
the current country's rows in RAM.
"""
from __future__ import annotations
import csv
import os
import sys
from typing import Iterator, Iterable
from collections import defaultdict

from .normalize import country_key

csv.field_size_limit(10**9)

# Windows compatibility for resource module
if sys.platform == 'win32':
    import psutil
    _process = psutil.Process(os.getpid())
else:
    import resource


def read_tsv(path: str) -> Iterator[dict]:
    """Yield rows as dicts. First row is the header. No pandas — memory-lean."""
    with open(path, encoding='utf-8', newline='') as f:
        r = csv.reader(f, delimiter='\t')
        header = next(r)
        for row in r:
            if len(row) != len(header):
                continue  # malformed guard; EDA says 0 malformed, but be safe.
            yield dict(zip(header, row))


def partition_by_country(path: str) -> dict[str, list[dict]]:
    """Read a TSV fully and bucket by country_key(country). Returns dict[key]->rows.

    Memory: proportional to file size. Called separately per source, and only for
    the sources needed for the current phase. For a full run we do NOT hold both
    S2 and S3 for a country at the same time in the caller — see blocker_v1.py.
    """
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in read_tsv(path):
        buckets[country_key(row.get('country', ''))].append(row)
    return buckets


def stream_country_rows(path: str, target_country_key: str) -> Iterator[dict]:
    """Streaming variant: yield only rows matching the target country_key.
    Two full scans of a source cost less than holding all buckets in RAM."""
    for row in read_tsv(path):
        if country_key(row.get('country', '')) == target_country_key:
            yield row


def country_row_counts(path: str) -> dict[str, int]:
    """Cheap pass: how many rows per country_key. Used to order partitions."""
    counts: dict[str, int] = defaultdict(int)
    for row in read_tsv(path):
        counts[country_key(row.get('country', ''))] += 1
    return dict(counts)


class CandidateWriter:
    """Writes candidate_pairs.tsv deterministically.

    Schema (tab-separated):
        s1_entity_id  candidate_entity_id  source  score

    - `source` is a string of P-tags that produced it, joined by '+', e.g. 'P2+P3'.
    - `score` is the combined rescored score (float, 6 decimals).
    - Rows for a given s1 are written contiguously, sorted by score desc then
      candidate_entity_id asc for stable output.
    """
    def __init__(self, path: str, write_header: bool = True, append: bool = False):
        self.path = path
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        mode = 'a' if append else 'w'
        self._f = open(path, mode, encoding='utf-8', newline='')
        self._w = csv.writer(self._f, delimiter='\t', lineterminator='\n')
        if write_header and not append:
            self._w.writerow(['s1_entity_id', 'candidate_entity_id', 'source', 'score'])
        self.rows_written = 0

    def write_s1(self, s1_id: str, candidates: Iterable[tuple[str, str, float]]) -> None:
        # candidates: iterable of (cand_id, source_tag, score). Sort stably.
        rows = sorted(candidates, key=lambda t: (-t[2], t[0]))
        for cid, src, sc in rows:
            self._w.writerow([s1_id, cid, src, f'{sc:.6f}'])
            self.rows_written += 1

    def flush(self) -> None:
        self._f.flush()

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def peak_rss_mb() -> float:
    """Peak resident set size for this process in MiB."""
    if sys.platform == 'win32':
        # psutil returns bytes on Windows
        return _process.memory_info().rss / 1024 / 1024
    else:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is in KiB on Linux, bytes on macOS.
        kb = ru.ru_maxrss
        return kb / 1024.0  # MiB on Linux
