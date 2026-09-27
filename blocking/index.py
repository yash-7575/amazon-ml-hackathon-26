"""Index construction + top-K search for the candidate-generation stage.

Split from `blocker_v1.py` so it's independently testable.

Design choices:
- TF-IDF via sklearn.feature_extraction.text.TfidfVectorizer. Standard, well-tested,
  produces l2-normalized CSR matrices so cosine similarity == sparse matmul.
- Query = chunked S1 chunk × index.T. Sparse-times-sparse in scipy returns sparse;
  we materialise dense per chunk for argpartition. Chunk size is bounded by config.
- No pairwise fuzzy metric anywhere in this file (blocking rule).
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TfidfIndex:
    """A single TF-IDF index over one document field for one country's rows."""
    ids: list[str]                    # aligned with matrix rows
    matrix: csr_matrix                # (n_docs, n_features), l2-normalised
    vectorizer: TfidfVectorizer
    field_label: str                  # 'name' | 'addr' | 'name_translit' — for logs

    @property
    def n_docs(self) -> int:
        return len(self.ids)

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    def memory_mb(self) -> float:
        m = self.matrix
        return (m.data.nbytes + m.indices.nbytes + m.indptr.nbytes) / (1024 * 1024)


def build_tfidf_index(
    ids: list[str],
    documents: list[str],
    *,
    field_label: str,
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int,
    max_df: float,
    sublinear_tf: bool,
) -> TfidfIndex:
    """Fit + transform in one pass over the corpus."""
    vec = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        lowercase=False,             # we already normalised.
        norm='l2',                   # so matmul == cosine.
        dtype=np.float32,
    )
    matrix = vec.fit_transform(documents)
    matrix.sort_indices()
    return TfidfIndex(
        ids=list(ids), matrix=matrix, vectorizer=vec, field_label=field_label,
    )


def transform_queries(index: TfidfIndex, documents: list[str]) -> csr_matrix:
    """Project query documents into the same TF-IDF space as the index."""
    q = index.vectorizer.transform(documents)
    q.sort_indices()
    return q


# The transpose of a multi-million-row index costs seconds and is IDENTICAL on every
# call. topk_search used to rebuild it per invocation, so a run that queries in N batches
# paid for N transposes of the same matrix. Cache it on identity and pre-warm it once
# before threads start.
_T_CACHE: dict = {}


def transposed_index(m: csr_matrix) -> csr_matrix:
    key = id(m)
    hit = _T_CACHE.get(key)
    if hit is not None and hit[0] is m:
        return hit[1]
    t = m.T.tocsr()
    _T_CACHE[key] = (m, t)
    return t


def topk_search(
    queries: csr_matrix,
    index_matrix: csr_matrix,
    k: int,
    *,
    chunk_rows: int,
    dtype: str = 'float32',
    max_temp_bytes: int = 512 * 1024 * 1024,   # 512 MiB cap on chunk-score buffer
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices[n_queries, k], scores[n_queries, k]) via chunked sparse matmul.

    Auto-shrinks chunk_rows if `chunk_rows * n_index_rows * dtype_bytes` would
    exceed max_temp_bytes. This is what keeps us inside 14GB RAM when the index
    has millions of documents. Padding: if index has fewer than k rows, missing
    slots are filled with index=-1 and score=-inf.
    """
    n_q = queries.shape[0]
    n_i = index_matrix.shape[0]
    # Memory-aware auto chunk sizing.
    dtype_bytes = 4 if dtype == 'float32' else 8
    max_rows = max(1, max_temp_bytes // max(n_i * dtype_bytes, 1))
    chunk_rows = max(1, min(chunk_rows, int(max_rows)))
    k_eff = min(k, n_i)
    if k_eff == 0:
        return (
            np.full((n_q, k), -1, dtype=np.int64),
            np.full((n_q, k), -np.inf, dtype=np.float32),
        )

    idx_T = transposed_index(index_matrix)
    out_idx = np.full((n_q, k), -1, dtype=np.int64)
    out_scr = np.full((n_q, k), -np.inf, dtype=np.float32)

    for start in range(0, n_q, chunk_rows):
        end = min(start + chunk_rows, n_q)
        # Sparse * sparse -> sparse. Densify only this chunk.
        chunk_scores = (queries[start:end] @ idx_T).toarray().astype(dtype, copy=False)
        # argpartition for top-k per row (unsorted top-k), then sort just those k.
        if k_eff < n_i:
            part = np.argpartition(-chunk_scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        else:
            part = np.tile(np.arange(n_i), (end - start, 1))
        # Gather scores for the partitioned indices.
        row_ix = np.arange(end - start)[:, None]
        gathered_scores = chunk_scores[row_ix, part]
        # Sort each row's k slice by score descending, ties by index ascending.
        order = np.lexsort((part, -gathered_scores), axis=1)
        sorted_idx = np.take_along_axis(part, order, axis=1)
        sorted_scr = np.take_along_axis(gathered_scores, order, axis=1)
        out_idx[start:end, :k_eff] = sorted_idx
        out_scr[start:end, :k_eff] = sorted_scr

    return out_idx, out_scr


def cosine_pairs(a: csr_matrix, b: csr_matrix) -> np.ndarray:
    """Row-wise cosine similarity for two aligned l2-normalised CSR matrices.
    Both must have shape (n, features). Returns shape (n,)."""
    assert a.shape == b.shape, f'{a.shape} vs {b.shape}'
    # Element-wise multiply then sum by row: equivalent to per-row dot product
    # because a and b are l2-normalised.
    prod = a.multiply(b)
    return np.asarray(prod.sum(axis=1)).ravel().astype(np.float32)
