"""blocker_v1 — country-partitioned union blocker producing candidate_pairs.tsv.

Pipeline (per country partition):
    P1  normalized-exact on name core                       (~22%, near-zero cost)
    P2  char-3gram TF-IDF cosine, top-K on name             (primary engine)
    P3  address-keyed word 1-2 TF-IDF cosine, top-K         (~15% w/ no name overlap)
    P4  transliterated Devanagari name index                (4.17% cross-script)
    UNION -> rescore(w_name*cos_name + w_addr*cos_addr) -> top-K -> write

Ground truth is NEVER consulted here. See measure.py for evaluation.

Usage:
    python -m blocking.blocker_v1                     # full run, writes candidate_pairs.tsv
    python -m blocking.blocker_v1 --only ireland      # single-partition dry run
    python -m blocking.blocker_v1 --dry-run           # smallest partition, print stats only
"""
from __future__ import annotations
import argparse
import time
import sys
from dataclasses import replace
from collections import defaultdict
from typing import Iterable
import numpy as np

from .config import BlockerConfig
from .io_utils import (
    read_tsv, country_row_counts, stream_country_rows,
    CandidateWriter, peak_rss_mb,
)
from .normalize import (
    base_normalize, name_core, name_tokens, addr_tokens,
    has_devanagari, transliterate_deva, country_key,
)
from .index import build_tfidf_index, transform_queries, topk_search, cosine_pairs


# ---------------------------------------------------------------------------
# Row containers — flat lists, indexed positionally to align with TF-IDF matrices.
# ---------------------------------------------------------------------------
class PartitionData:
    """Rows for one country partition, already normalized. Held in memory as
    parallel lists of primitives — no pandas."""
    def __init__(self, label: str, source_tag: str):
        self.label = label
        self.source_tag = source_tag        # 'S1' | 'S2' | 'S3'
        self.ids: list[str] = []
        self.name_raw: list[str] = []
        self.addr_raw: list[str] = []
        self.name_doc: list[str] = []       # base_normalize(name)
        self.addr_doc: list[str] = []       # base_normalize(addr), '' if empty
        self.name_core: list[str] = []      # sorted-token canonical string for P1
        self.name_translit: list[str] = []  # unidecode name; '' if not Devanagari
        # for P3 memory: track which docs have non-empty address.
        self.addr_nonempty_mask: list[bool] = []

    def add(self, row: dict, keep_raw: bool = True) -> None:
        eid = row['entity_id']
        name = row.get('business_name', '') or ''
        addr = row.get('business_address', '') or ''
        self.ids.append(eid)
        self.name_raw.append(name if keep_raw else '')
        self.addr_raw.append(addr if keep_raw else '')
        self.name_doc.append(base_normalize(name))
        addr_n = base_normalize(addr)
        self.addr_doc.append(addr_n)
        self.addr_nonempty_mask.append(bool(addr_n))
        self.name_core.append(name_core(name))
        if has_devanagari(name):
            self.name_translit.append(base_normalize(transliterate_deva(name)))
        else:
            self.name_translit.append('')

    @property
    def n(self) -> int:
        return len(self.ids)


def load_partition(path: str, target_ck: str, source_tag: str, label: str,
                   keep_raw: bool = True) -> PartitionData:
    p = PartitionData(label=label, source_tag=source_tag)
    for row in stream_country_rows(path, target_ck):
        p.add(row, keep_raw=keep_raw)
    return p


# ---------------------------------------------------------------------------
# Blockers — each returns dict[s1_row_idx] -> list[(cand_source_tag, cand_id, score, blocker_tag)]
# ---------------------------------------------------------------------------
def block_p1_exact(
    s1: PartitionData, targets: list[PartitionData]
) -> dict[int, list[tuple[str, str, float, str]]]:
    """Exact name_core match. Score = 1.0.
    For P1 we also add transliterated-Latin names as extra keys so a Devanagari
    S2/S3 whose transliteration matches an ASCII S1 will be caught here."""
    # Build inverse index: name_core -> [(source_tag, cand_id)]
    inv: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for t in targets:
        for i, nc in enumerate(t.name_core):
            if nc:
                inv[nc].append((t.source_tag, t.ids[i]))
            # transliterated name gets its own name_core-style key
            nct = ' '.join(sorted(name_tokens(t.name_translit[i], drop_suffix=True))) \
                if t.name_translit[i] else ''
            if nct and nct != nc:
                inv[nct].append((t.source_tag, t.ids[i]))
    out: dict[int, list[tuple[str, str, float, str]]] = defaultdict(list)
    for i, nc in enumerate(s1.name_core):
        if not nc:
            continue
        for (src, cid) in inv.get(nc, ()):
            out[i].append((src, cid, 1.0, 'P1'))
    return out


def _build_name_index(target: PartitionData, cfg: BlockerConfig):
    """Build the name TF-IDF index with Devanagari transliteration aliases folded in.
    Returns (index_or_None, idx_ids, translit_row_mask) where translit_row_mask[i]
    is True iff row i in the index is a P4 transliteration alias (not a native doc)."""
    idx_ids: list[str] = []
    idx_docs: list[str] = []
    translit_mask: list[bool] = []
    for i, doc in enumerate(target.name_doc):
        if doc:
            idx_ids.append(target.ids[i]); idx_docs.append(doc); translit_mask.append(False)
        if cfg.p4_enabled and target.name_translit[i]:
            idx_ids.append(target.ids[i]); idx_docs.append(target.name_translit[i])
            translit_mask.append(True)
    if not idx_docs:
        return None, [], []
    try:
        index = build_tfidf_index(
            idx_ids, idx_docs,
            field_label=f'name[{target.source_tag}]',
            analyzer=cfg.p2_analyzer,
            ngram_range=(cfg.p2_ngram_lo, cfg.p2_ngram_hi),
            min_df=cfg.p2_min_df, max_df=cfg.p2_max_df,
            sublinear_tf=cfg.p2_sublinear_tf,
        )
    except ValueError:
        return None, [], []
    return index, idx_ids, translit_mask


def _build_addr_index(target: PartitionData, cfg: BlockerConfig):
    ids: list[str] = []; docs: list[str] = []
    for i, d in enumerate(target.addr_doc):
        if d:
            ids.append(target.ids[i]); docs.append(d)
    if not docs:
        return None, []
    try:
        index = build_tfidf_index(
            ids, docs,
            field_label=f'addr[{target.source_tag}]',
            analyzer=cfg.p3_analyzer,
            ngram_range=(cfg.p3_ngram_lo, cfg.p3_ngram_hi),
            min_df=cfg.p3_min_df, max_df=cfg.p3_max_df,
            sublinear_tf=cfg.p3_sublinear_tf,
        )
    except ValueError:
        return None, []
    return index, ids


def block_p2_name_tfidf(
    s1: PartitionData, target: PartitionData, cfg: BlockerConfig,
    prebuilt: tuple | None = None,
) -> dict[int, list[tuple[str, str, float, str]]]:
    """char-3gram TF-IDF cosine, top-K over one target source.

    Devanagari records in target contribute TWO documents: the Devanagari string
    itself AND its transliteration — this is P4 folded in efficiently (one shared
    vocabulary). The union tracks both aliases back to the same id.
    """
    if prebuilt is None:
        index, idx_ids, translit_mask = _build_name_index(target, cfg)
    else:
        index, idx_ids, translit_mask = prebuilt
    if index is None:
        return {}
    # Query docs: S1 rows (skip empties). Track a query_positions -> s1_idx map.
    q_docs: list[str] = []
    q_pos_to_s1: list[int] = []
    for i, doc in enumerate(s1.name_doc):
        if doc:
            q_docs.append(doc)
            q_pos_to_s1.append(i)
    if not q_docs:
        return {}
    Q = transform_queries(index, q_docs)
    top_idx, top_scr = topk_search(
        Q, index.matrix, cfg.p2_top_k,
        chunk_rows=cfg.p2_chunk_rows, dtype=cfg.p2_score_dtype,
        backend=cfg.topk_backend, n_threads=cfg.topk_threads,
    )
    out: dict[int, list[tuple[str, str, float, str]]] = defaultdict(list)
    for qi, s1_idx in enumerate(q_pos_to_s1):
        for j in range(top_idx.shape[1]):
            ii = int(top_idx[qi, j])
            if ii < 0:
                break
            sc = float(top_scr[qi, j])
            if not np.isfinite(sc) or sc <= 0.0:
                continue
            cid = idx_ids[ii]
            tag = 'P4' if (translit_mask and translit_mask[ii]) else 'P2'
            out[s1_idx].append((target.source_tag, cid, sc, tag))
    return out


def block_p3_addr_tfidf(
    s1: PartitionData, target: PartitionData, cfg: BlockerConfig,
    prebuilt: tuple | None = None,
) -> dict[int, list[tuple[str, str, float, str]]]:
    """word 1-2 gram TF-IDF cosine on addresses, top-K. Recovers zero-name-overlap
    matches. Rows with empty address are excluded from the index AND queries."""
    if prebuilt is None:
        index, idx_ids = _build_addr_index(target, cfg)
    else:
        index, idx_ids = prebuilt
    if index is None:
        return {}
    q_docs: list[str] = []
    q_pos_to_s1: list[int] = []
    for i, doc in enumerate(s1.addr_doc):
        if doc:
            q_docs.append(doc)
            q_pos_to_s1.append(i)
    if not q_docs:
        return {}
    Q = transform_queries(index, q_docs)
    top_idx, top_scr = topk_search(
        Q, index.matrix, cfg.p3_top_k,
        chunk_rows=cfg.p3_chunk_rows, dtype=cfg.p3_score_dtype,
        backend=cfg.topk_backend, n_threads=cfg.topk_threads,
    )
    out: dict[int, list[tuple[str, str, float, str]]] = defaultdict(list)
    for qi, s1_idx in enumerate(q_pos_to_s1):
        for j in range(top_idx.shape[1]):
            ii = int(top_idx[qi, j])
            if ii < 0:
                break
            sc = float(top_scr[qi, j])
            if not np.isfinite(sc) or sc <= 0.0:
                continue
            out[s1_idx].append((target.source_tag, idx_ids[ii], sc, 'P3'))
    return out


# ---------------------------------------------------------------------------
# Rescore + union
# ---------------------------------------------------------------------------
def rescore_and_union(
    s1: PartitionData,
    per_source_hits: list[dict[int, list[tuple[str, str, float, str]]]],
    s2_by_id: dict[str, tuple[str, str]],   # cand_id -> (name_doc, addr_doc)
    s3_by_id: dict[str, tuple[str, str]],
    cfg: BlockerConfig,
    # pre-built per-source TF-IDF indexes so we can vectorize the rescore.
    name_index_s2, name_index_s3,
    addr_index_s2, addr_index_s3,
) -> dict[int, list[tuple[str, str, float, str]]]:
    """Union all P1..P4 hits per S1, then compute a proper cos_name+cos_addr score.

    We rebuild query and target vectors for the specific pairs, so we get a real
    combined-score number even when the pair came in only from the name blocker
    or only from the address blocker. Vectorized: no Python per-pair loop over
    pairwise metrics."""
    # 1) union
    merged: dict[int, dict[str, dict]] = defaultdict(dict)
    for hits in per_source_hits:
        for i, lst in hits.items():
            for (src, cid, sc, tag) in lst:
                key = f'{src}:{cid}'
                slot = merged[i].get(key)
                if slot is None:
                    merged[i][key] = {'src': src, 'cid': cid, 'tags': {tag}, 'raw_max': sc}
                else:
                    slot['tags'].add(tag)
                    if sc > slot['raw_max']:
                        slot['raw_max'] = sc

    if not merged:
        return {}

    # 2) collect pair batch for a vectorised rescore.
    pair_s1_idx: list[int] = []
    pair_src: list[str] = []
    pair_cid: list[str] = []
    pair_tags: list[str] = []
    pair_raw: list[float] = []
    for i, slots in merged.items():
        for slot in slots.values():
            pair_s1_idx.append(i)
            pair_src.append(slot['src'])
            pair_cid.append(slot['cid'])
            pair_tags.append('+'.join(sorted(slot['tags'])))
            pair_raw.append(slot['raw_max'])
    if not pair_s1_idx:
        return {}
    pair_s1_idx = np.asarray(pair_s1_idx, dtype=np.int64)
    pair_raw = np.asarray(pair_raw, dtype=np.float32)

    # 3) name-cos: build query and doc row-batches per source.
    def _cos_for(field_index, cand_ids: list[str], s1_docs_for_field: list[str]):
        if field_index is None or field_index.n_docs == 0:
            return np.zeros(len(cand_ids), dtype=np.float32)
        # queries: s1_docs_for_field aligned with cand_ids (one query per pair).
        q = transform_queries(field_index, s1_docs_for_field)
        # target rows: look up each cand_id -> row index in field_index.
        cid_to_row = {cid: i for i, cid in enumerate(field_index.ids)}
        rows = np.fromiter((cid_to_row.get(c, -1) for c in cand_ids), dtype=np.int64,
                           count=len(cand_ids))
        keep = rows >= 0
        scores = np.zeros(len(cand_ids), dtype=np.float32)
        if keep.any():
            sub_q = q[np.where(keep)[0]]
            sub_d = field_index.matrix[rows[keep]]
            scores[keep] = cosine_pairs(sub_q, sub_d)
        return scores

    # Partition pairs by source.
    mask_s2 = np.array([s == 'S2' for s in pair_src])
    mask_s3 = ~mask_s2

    # S1 name/addr docs aligned with the pair.
    pair_s1_name_docs = [s1.name_doc[i] for i in pair_s1_idx]
    pair_s1_addr_docs = [s1.addr_doc[i] for i in pair_s1_idx]

    cos_name = np.zeros(len(pair_s1_idx), dtype=np.float32)
    cos_addr = np.zeros(len(pair_s1_idx), dtype=np.float32)

    if mask_s2.any():
        s2_cids = [pair_cid[i] for i in np.where(mask_s2)[0]]
        s2_q_name = [pair_s1_name_docs[i] for i in np.where(mask_s2)[0]]
        s2_q_addr = [pair_s1_addr_docs[i] for i in np.where(mask_s2)[0]]
        cos_name[mask_s2] = _cos_for(name_index_s2, s2_cids, s2_q_name)
        cos_addr[mask_s2] = _cos_for(addr_index_s2, s2_cids, s2_q_addr)
    if mask_s3.any():
        s3_cids = [pair_cid[i] for i in np.where(mask_s3)[0]]
        s3_q_name = [pair_s1_name_docs[i] for i in np.where(mask_s3)[0]]
        s3_q_addr = [pair_s1_addr_docs[i] for i in np.where(mask_s3)[0]]
        cos_name[mask_s3] = _cos_for(name_index_s3, s3_cids, s3_q_name)
        cos_addr[mask_s3] = _cos_for(addr_index_s3, s3_cids, s3_q_addr)

    combined = cfg.rescore_w_name * cos_name + cfg.rescore_w_addr * cos_addr
    # Weighted raw-score floor. Prevents the combined score from ranking a pair
    # strictly below either raw signal after the weights are applied.
    #
    # This is imperfect: on the India smoke test the final top-K recall
    # (94.30% @ K=50) lags the P1+P2+P3 union recall (98.27%) by ~4pp. The
    # ranking is where true pairs get displaced by name-strong distractors, not
    # the union.
    #
    # DO NOT "fix" this by dropping the weight multiplication (using
    # np.maximum(pair_raw, cos_addr) unscaled). It was tested and measured
    # WORSE — India smoke final top-K dropped to 93.30% (−1 pp), cross-script
    # to 73.68% (−2.7 pp). Cause: real cos_name values run 0.5–0.95, real
    # cos_addr values run 0.10–0.35 (TF-IDF cosine, not Jaccard). Unweighted
    # max promotes the field with higher dynamic range — always names — so
    # name-strong distractors (per FINDING 4, generic names are abundant)
    # get bigger score boosts than address-only true pairs. Net: fewer true
    # pairs survive the top-K trim.
    #
    # The correct long-term fix is score calibration / rank-fusion (RRF); the
    # short-term mitigation is raising final_top_k so the trim discards less.
    raw_floor = np.maximum(pair_raw * cfg.rescore_w_name,
                           cos_addr * cfg.rescore_w_addr)
    combined = np.maximum(combined, raw_floor)

    # 4) group by s1, keep final_top_k.
    out: dict[int, list[tuple[str, str, float, str]]] = defaultdict(list)
    for i, sc in enumerate(combined):
        if sc < cfg.min_output_score:
            continue
        out[int(pair_s1_idx[i])].append(
            (pair_src[i], pair_cid[i], float(sc), pair_tags[i])
        )
    # top-K per s1
    trimmed: dict[int, list[tuple[str, str, float, str]]] = {}
    for i, lst in out.items():
        lst.sort(key=lambda t: (-t[2], t[1]))
        trimmed[i] = lst[: cfg.final_top_k]
    return trimmed


def rescore_and_union_batched(
    s1: PartitionData,
    per_source_hits: list[dict[int, list[tuple[str, str, float, str]]]],
    s2_by_id: dict[str, tuple[str, str]],
    s3_by_id: dict[str, tuple[str, str]],
    cfg: BlockerConfig,
    name_index_s2, name_index_s3,
    addr_index_s2, addr_index_s3,
    s1_batch: int = 20000,
) -> dict[int, list[tuple[str, str, float, str]]]:
    """Batch rescore_and_union over S1-index ranges so the pair-level
    vectorization never holds all pairs at once (test-scale OOM fix).
    Identical output to a single rescore_and_union call: per-S1 top-K trim
    is independent across S1 entities."""
    out: dict[int, list[tuple[str, str, float, str]]] = {}
    n = s1.n
    for start in range(0, n, s1_batch):
        end = min(start + s1_batch, n)
        sub = PartitionData(label=s1.label, source_tag=s1.source_tag)
        sub.ids = s1.ids[start:end]
        sub.name_doc = s1.name_doc[start:end]
        sub.addr_doc = s1.addr_doc[start:end]
        sub.name_core = s1.name_core[start:end]
        sub.name_translit = s1.name_translit[start:end]
        sub.addr_nonempty_mask = s1.addr_nonempty_mask[start:end]
        remapped = []
        for hits in per_source_hits:
            d: dict[int, list[tuple[str, str, float, str]]] = {}
            for i, lst in hits.items():
                if start <= i < end:
                    d[i - start] = lst
            remapped.append(d)
        part = rescore_and_union(
            sub, remapped, s2_by_id=s2_by_id, s3_by_id=s3_by_id, cfg=cfg,
            name_index_s2=name_index_s2, name_index_s3=name_index_s3,
            addr_index_s2=addr_index_s2, addr_index_s3=addr_index_s3,
        )
        for i, lst in part.items():
            out[i + start] = lst
    return out


# ---------------------------------------------------------------------------
# Per-country driver
# ---------------------------------------------------------------------------
def _build_partition_indexes(target: PartitionData, cfg: BlockerConfig):
    """Return (name_index, name_ids, addr_index, addr_ids) for one target source.
    Kept as a convenience wrapper so measure.py can build in one call."""
    name_index, name_ids, _mask = _build_name_index(target, cfg)
    addr_index, addr_ids = _build_addr_index(target, cfg)
    return name_index, addr_index


def run_partition(
    ck: str, cfg: BlockerConfig, writer: CandidateWriter | None,
) -> dict:
    """Process one country partition end-to-end. Returns per-partition stats."""
    t0 = time.time()
    d = cfg.data_root + ('train/' if cfg.split == 'train' else 'test/')
    s1_path = d + f'{cfg.split}_source1.tsv'
    s2_path = d + f'{cfg.split}_source2.tsv'
    s3_path = d + f'{cfg.split}_source3.tsv'

    s1 = load_partition(s1_path, ck, 'S1', label=f'S1[{ck}]', keep_raw=False)
    s2 = load_partition(s2_path, ck, 'S2', label=f'S2[{ck}]', keep_raw=False)
    s3 = load_partition(s3_path, ck, 'S3', label=f'S3[{ck}]', keep_raw=False)
    t_load = time.time() - t0

    stats = {
        'country': ck, 'n_s1': s1.n, 'n_s2': s2.n, 'n_s3': s3.n,
        'load_sec': round(t_load, 2),
    }
    if s1.n == 0 or (s2.n == 0 and s3.n == 0):
        stats['skipped'] = True
        return stats

    # Build indexes ONCE per source and reuse them for both blocking and rescoring.
    t = time.time()
    name_s2 = _build_name_index(s2, cfg) if (cfg.p2_enabled and s2.n) else (None, [], [])
    name_s3 = _build_name_index(s3, cfg) if (cfg.p2_enabled and s3.n) else (None, [], [])
    addr_s2 = _build_addr_index(s2, cfg) if (cfg.p3_enabled and s2.n) else (None, [])
    addr_s3 = _build_addr_index(s3, cfg) if (cfg.p3_enabled and s3.n) else (None, [])
    stats['index_build_sec'] = round(time.time() - t, 2)

    # P1
    t = time.time()
    hits_p1 = block_p1_exact(s1, [s2, s3]) if cfg.p1_enabled else {}
    stats['p1_sec'] = round(time.time() - t, 2)

    # P2 per source (reuse prebuilt indexes)
    t = time.time()
    hits_p2_s2 = block_p2_name_tfidf(s1, s2, cfg, prebuilt=name_s2) if name_s2[0] else {}
    hits_p2_s3 = block_p2_name_tfidf(s1, s3, cfg, prebuilt=name_s3) if name_s3[0] else {}
    stats['p2_sec'] = round(time.time() - t, 2)

    # P3 per source (reuse prebuilt indexes)
    t = time.time()
    hits_p3_s2 = block_p3_addr_tfidf(s1, s2, cfg, prebuilt=addr_s2) if addr_s2[0] else {}
    hits_p3_s3 = block_p3_addr_tfidf(s1, s3, cfg, prebuilt=addr_s3) if addr_s3[0] else {}
    stats['p3_sec'] = round(time.time() - t, 2)

    # Rescore + union — reuse the same indexes.
    t = time.time()
    per_source_hits = [hits_p1, hits_p2_s2, hits_p2_s3, hits_p3_s2, hits_p3_s3]
    final = rescore_and_union_batched(
        s1, per_source_hits,
        s2_by_id={}, s3_by_id={},
        cfg=cfg,
        name_index_s2=name_s2[0], name_index_s3=name_s3[0],
        addr_index_s2=addr_s2[0], addr_index_s3=addr_s3[0],
        s1_batch=20000,
    )
    stats['rescore_sec'] = round(time.time() - t, 2)
    # Explicitly drop indexes before writing (frees memory before the next partition).
    del name_s2, name_s3, addr_s2, addr_s3

    # Write.
    n_pairs = 0
    if writer is not None:
        for i in range(s1.n):
            hits = final.get(i, [])
            if not hits:
                continue
            writer.write_s1(s1.ids[i], hits)
            n_pairs += len(hits)
    else:
        n_pairs = sum(len(v) for v in final.values())
    stats['pairs_written'] = n_pairs
    stats['peak_rss_mib'] = round(peak_rss_mb(), 1)
    stats['total_sec'] = round(time.time() - t0, 2)
    stats['candidates_per_s1'] = round(n_pairs / max(s1.n, 1), 2)
    return stats


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def _partition_order(counts_s1: dict[str, int], mode: str) -> list[str]:
    items = [(k, v) for k, v in counts_s1.items() if v > 0]
    if mode == 'ascending_size':
        items.sort(key=lambda kv: (kv[1], kv[0]))
    elif mode == 'descending_size':
        items.sort(key=lambda kv: (-kv[1], kv[0]))
    else:
        items.sort(key=lambda kv: kv[0])
    return [k for k, _ in items]


def run(cfg: BlockerConfig, dry_run: bool = False,
        only_partition: str | None = None) -> list[dict]:
    d = cfg.data_root + ('train/' if cfg.split == 'train' else 'test/')
    s1_path = d + f'{cfg.split}_source1.tsv'

    print(f'[blocker_v1] scanning S1 for country counts...', flush=True)
    counts = country_row_counts(s1_path)
    print(f'  {len(counts)} distinct country_keys in S1 (top 10 by size):')
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f'    {v:>10,}  {k}')

    order = _partition_order(counts, cfg.partition_order)
    if only_partition is not None:
        order = [only_partition] if only_partition in counts else []
        if not order:
            print(f'[blocker_v1] partition {only_partition!r} not found. '
                  f'Available: {list(counts.keys())[:20]}', file=sys.stderr)
            return []
    if dry_run:
        order = order[:1]
        print(f'[blocker_v1] DRY RUN — smallest partition only: {order[0]!r}')

    writer = None
    if not dry_run:
        writer = CandidateWriter(cfg.output_path, write_header=cfg.write_header)
        print(f'[blocker_v1] writing to {cfg.output_path}')

    all_stats: list[dict] = []
    try:
        for ck in order:
            print(f'\n[blocker_v1] === partition {ck!r} (S1 rows = {counts[ck]:,}) ===', flush=True)
            stats = run_partition(ck, cfg, writer)
            all_stats.append(stats)
            for k, v in stats.items():
                print(f'    {k}: {v}')
    finally:
        if writer is not None:
            writer.close()
            print(f'\n[blocker_v1] done. rows written = {writer.rows_written:,}')

    print(f'\n[blocker_v1] peak RSS = {peak_rss_mb():.1f} MiB')
    return all_stats


# ---------------------------------------------------------------------------
def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=('train', 'test'), default='train')
    ap.add_argument('--only', dest='only_partition', default=None,
                    help='Run one country_key only (e.g. "ireland").')
    ap.add_argument('--dry-run', action='store_true',
                    help='Smallest partition only; do not write output.')
    ap.add_argument('--output', default=None,
                    help='Output TSV path (default: blocking/candidate_pairs.tsv)')
    ap.add_argument('--measure', action='store_true',
                    help='Delegate to blocking.measure (evaluates against ground truth).')
    ap.add_argument('--measure-sample', type=int, default=None)
    return ap.parse_args()


def main():
    args = _parse_args()
    cfg = BlockerConfig(split=args.split)
    if args.output is not None:
        cfg = replace(cfg, output_path=args.output)

    if args.measure:
        # Delegate to measure.py so this file stays generation-only.
        from .measure import run_measurement
        run_measurement(cfg,
                        sample_s1=args.measure_sample or cfg.measure_sample_s1,
                        only_partition=args.only_partition,
                        dry_run=args.dry_run)
        return

    run(cfg, dry_run=args.dry_run, only_partition=args.only_partition)


if __name__ == '__main__':
    main()
