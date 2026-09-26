"""Vectorized pairwise feature extraction.

THIS MODULE MUST NOT IMPORT GROUND TRUTH. That is the structural guarantee against
leakage -- it cannot encode labels because it cannot see them. `leakage_check.py`
enforces it.

Design rules enforced here:
  - operate on ARRAYS of pairs, never loop over pairs in Python
  - undefined similarity is NaN, never 0 (0 asserts disagreement -- a false statement)
  - asymmetric features keep their direction; never averaged away
  - float32 output with explicit column names
  - rapidfuzz cpdist with score_cutoff + workers=-1
"""
from __future__ import annotations
import unicodedata
import re
from typing import Dict, List, Sequence, Tuple
import numpy as np

from .featconfig import FeatureConfig

try:
    from rapidfuzz import process as rf_process, fuzz as rf_fuzz
    _HAS_RF = True
except ImportError:                                    # degrade, do not crash
    _HAS_RF = False

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")

LEGAL = {"ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co",
         "company", "plc", "pvt", "private", "pte", "gmbh", "ag", "sarl", "sas", "sa",
         "eurl", "bv", "nv", "srl", "spa", "sl"}


# ---------------------------------------------------------------- normalization
def strip_punct_by_category(s: str) -> str:
    """Strip punctuation by Unicode CATEGORY, preserving combining marks.

    [^\\w\\s] shatters Devanagari: vowel signs and virama are categories Mn/Mc, which
    \\w does not match, so words fragment into single consonants -- silently.
    """
    return "".join(" " if (unicodedata.category(c)[0] in "PSZ" and not c.isspace()) else c
                   for c in s)


def normalize(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold().replace("&", " and ")
    return _WS.sub(" ", strip_punct_by_category(s)).strip()


def core_tokens(norm: str) -> List[str]:
    toks = norm.split()
    core = [t for t in toks if t not in LEGAL]
    return core or toks                                # never return empty


def dominant_script(s: str) -> str:
    counts: Dict[str, int] = {}
    for ch in s:
        if not ch.isalpha():
            continue
        try:
            n = unicodedata.name(ch).split()[0]
        except ValueError:
            continue
        counts[n] = counts.get(n, 0) + 1
    return max(counts, key=counts.get) if counts else "NONE"


# ---------------------------------------------------------------- helpers
def _ngrams(s: str, n: int) -> set:
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    """NaN when either side is empty: 'equally absent' is not evidence of a match.
    jaccard(empty, empty) is conventionally 1.0 -- that would assert a perfect match
    between two records that say nothing. Guard it."""
    if not a or not b:
        return np.nan
    u = len(a | b)
    return len(a & b) / u if u else np.nan


def _dice(a: set, b: set) -> float:
    if not a or not b:
        return np.nan
    return 2 * len(a & b) / (len(a) + len(b))


def _containment(a: set, b: set) -> float:
    """Asymmetric by design -- |A n B| / |A|. Keep both directions."""
    if not a or not b:
        return np.nan
    return len(a & b) / len(a)


def _ratio(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    m, M = np.minimum(x, y), np.maximum(x, y)
    return np.where(M > 0, m / np.maximum(M, 1e-9), np.nan)


def _cpdist(left: Sequence[str], right: Sequence[str], scorer, cfg: FeatureConfig) -> np.ndarray:
    """Vectorized element-wise string metric. C++, multithreaded, GIL released."""
    if not _HAS_RF:
        return np.full(len(left), np.nan, dtype=np.float32)
    out = rf_process.cpdist(list(left), list(right), scorer=scorer,
                            score_cutoff=cfg.score_cutoff, workers=cfg.workers)
    return (np.asarray(out, dtype=np.float32) / 100.0)


# ---------------------------------------------------------------- record prep
def prepare_records(names: Sequence[str], addrs: Sequence[str],
                    cfg: FeatureConfig) -> Dict[str, list]:
    """Normalize ONCE per record, not per pair. Records >> pairs is never true here,
    but pairs >> records always is -- so per-record work is nearly free."""
    n_norm = [normalize(x) for x in names]
    a_norm = [normalize(x) for x in addrs]
    return {
        "name_norm": n_norm,
        "name_core": [" ".join(core_tokens(x)) for x in n_norm],
        "name_tokens": [set(core_tokens(x)) for x in n_norm],
        "name_ngrams": [_ngrams(x, cfg.ngram_n) for x in n_norm],
        "addr_norm": a_norm,
        "addr_tokens": [set(x.split()) for x in a_norm],
        "addr_ngrams": [_ngrams(x, cfg.ngram_n) for x in a_norm],
        "addr_nums": [set(_NUM.findall(x)) for x in a_norm],
        # digits in the NAME are a separate, critical signal: "Store 14" vs "Store 15"
        # are DIFFERENT businesses but score ~0.97 on every string metric. Only an
        # explicit digit comparison rejects them.
        "name_nums": [set(_NUM.findall(x)) for x in n_norm],
        "script": [dominant_script(x) for x in n_norm],
    }


def token_idf(token_sets: Sequence[set]) -> Dict[str, float]:
    """Corpus IDF. An UNSUPERVISED statistic -- computing it per split is correct and is
    NOT leakage. Document frequencies genuinely differ between corpora."""
    df: Dict[str, int] = {}
    for s in token_sets:
        for t in s:
            df[t] = df.get(t, 0) + 1
    n = len(token_sets)
    return {t: float(np.log((n + 1) / (d + 1)) + 1.0) for t, d in df.items()}


# ---------------------------------------------------------------- the extractor
def extract(li: np.ndarray, ri: np.ndarray,
            L: Dict[str, list], R: Dict[str, list],
            cfg: FeatureConfig,
            retrieval_score: np.ndarray | None = None,
            idf: Dict[str, float] | None = None) -> Tuple[np.ndarray, List[str]]:
    """THE frozen extractor. Call this for BOTH train and test -- never a second copy.

    li, ri : aligned int index arrays into L and R.
    Returns (matrix float32 [n_pairs, n_features], column_names).
    """
    P = len(li)
    cols: List[str] = []
    feats: List[np.ndarray] = []

    def add(name: str, arr) -> None:
        cols.append(name)
        feats.append(np.asarray(arr, dtype=np.float32))

    lt = [L["name_tokens"][i] for i in li]; rt = [R["name_tokens"][j] for j in ri]
    lg = [L["name_ngrams"][i] for i in li]; rg = [R["name_ngrams"][j] for j in ri]
    lat = [L["addr_tokens"][i] for i in li]; rat = [R["addr_tokens"][j] for j in ri]
    lag = [L["addr_ngrams"][i] for i in li]; rag = [R["addr_ngrams"][j] for j in ri]
    lan = [L["addr_nums"][i] for i in li];  ran = [R["addr_nums"][j] for j in ri]
    lnm = [L["name_nums"][i] for i in li];  rnm = [R["name_nums"][j] for j in ri]
    lnn = [L["name_norm"][i] for i in li];  rnn = [R["name_norm"][j] for j in ri]
    lnc = [L["name_core"][i] for i in li];  rnc = [R["name_core"][j] for j in ri]
    lad = [L["addr_norm"][i] for i in li];  rad = [R["addr_norm"][j] for j in ri]

    addr_both = np.array([bool(a) and bool(b) for a, b in zip(lad, rad)])
    name_both = np.array([bool(a) and bool(b) for a, b in zip(lnn, rnn)])

    # ---- retrieval / provenance (free: blocking already computed it) ----
    if retrieval_score is not None:
        add("retrieval_score", retrieval_score)

    # ---- A. exact indicators ----
    if cfg.exact:
        add("name_exact", [a == b and bool(a) for a, b in zip(lnn, rnn)])
        add("name_core_exact", [a == b and bool(a) for a, b in zip(lnc, rnc)])
        add("first_token_exact", [bool(a) and bool(b) and a.split()[0] == b.split()[0]
                                  for a, b in zip(lnn, rnn)])
        add("last_token_exact", [bool(a) and bool(b) and a.split()[-1] == b.split()[-1]
                                 for a, b in zip(lnn, rnn)])
        add("addr_exact", [bool(a) and a == b for a, b in zip(lad, rad)])
        add("addr_numeric_set_exact", [bool(a) and a == b for a, b in zip(lan, ran)])
        # --- name digits: the sequential-numbering guard ---
        # "Store 14" vs "Store 15": Levenshtein, Jaro-Winkler and n-gram Jaccard all score
        # ~0.97 on different businesses. Under a precision-weighted metric that false
        # positive costs double. These three features are the only thing that rejects it.
        add("name_digits_equal", [a == b for a, b in zip(lnm, rnm)])
        add("name_digits_conflict", [bool(a) and bool(b) and a != b
                                     for a, b in zip(lnm, rnm)])
        add("name_has_digits_both", [bool(a) and bool(b) for a, b in zip(lnm, rnm)])

    # ---- B. token-set similarity ----
    if cfg.token:
        add("name_token_jaccard", [_jaccard(a, b) for a, b in zip(lt, rt)])
        add("name_token_dice", [_dice(a, b) for a, b in zip(lt, rt)])
        add("name_contain_fwd", [_containment(a, b) for a, b in zip(lt, rt)])   # asymmetric
        add("name_contain_rev", [_containment(b, a) for a, b in zip(lt, rt)])   # both directions
        add("shared_token_count", [len(a & b) for a, b in zip(lt, rt)])
        if idf:
            add("idf_weighted_overlap",
                [sum(idf.get(t, 0.0) for t in (a & b)) for a, b in zip(lt, rt)])
            add("max_idf_shared",
                [max((idf.get(t, 0.0) for t in (a & b)), default=0.0) for a, b in zip(lt, rt)])

    # ---- C. char n-gram ----
    if cfg.ngram:
        add("name_ngram_jaccard", [_jaccard(a, b) for a, b in zip(lg, rg)])
        add("name_ngram_contain", [_containment(a, b) for a, b in zip(lg, rg)])

    # ---- lengths / shape ----
    ln = np.array([len(x) for x in lnn], dtype=np.float32)
    rn = np.array([len(x) for x in rnn], dtype=np.float32)
    add("name_len_diff", np.abs(ln - rn))
    add("name_len_ratio", _ratio(ln, rn))
    ltc = np.array([len(x) for x in lt], dtype=np.float32)
    rtc = np.array([len(x) for x in rt], dtype=np.float32)
    add("token_count_diff", np.abs(ltc - rtc))
    add("token_count_ratio", _ratio(ltc, rtc))

    # ---- F. address ----
    if cfg.address:
        add("addr_token_jaccard", [_jaccard(a, b) for a, b in zip(lat, rat)])
        add("addr_ngram_jaccard", [_jaccard(a, b) for a, b in zip(lag, rag)])
        add("addr_numeric_jaccard", [_jaccard(a, b) for a, b in zip(lan, ran)])

    # ---- D. edit distance (expensive tier) ----
    if cfg.edit_distance:
        add("name_token_sort_ratio", _cpdist(lnn, rnn, rf_fuzz.token_sort_ratio, cfg))
        add("name_token_set_ratio", _cpdist(lnn, rnn, rf_fuzz.token_set_ratio, cfg))
        add("name_ratio", _cpdist(lnn, rnn, rf_fuzz.ratio, cfg))

    # ---- I. multilingual ----
    if cfg.multilingual:
        lsc = [L["script"][i] for i in li]; rsc = [R["script"][j] for j in ri]
        add("script_match", [a == b for a, b in zip(lsc, rsc)])
        add("is_cross_script", [a != b and a != "NONE" and b != "NONE"
                                for a, b in zip(lsc, rsc)])

    # ---- H. missingness: NaN above, explicit flags here ----
    if cfg.missingness:
        add("addr_missing_either", ~addr_both)
        add("addr_missing_both", [not a and not b for a, b in zip(lad, rad)])
        add("name_missing_either", ~name_both)
        add("n_fields_comparable", name_both.astype(np.float32) + addr_both.astype(np.float32))

    M = np.vstack(feats).T.astype(cfg.dtype)

    # ---- G. cross-field consistency: evidence NEITHER field gives alone ----
    if cfg.cross_field:
        ci = {c: k for k, c in enumerate(cols)}
        ns = M[:, ci["name_ngram_jaccard"]] if "name_ngram_jaccard" in ci else \
             M[:, ci["name_token_jaccard"]]
        as_ = M[:, ci["addr_token_jaccard"]] if "addr_token_jaccard" in ci else \
              np.full(P, np.nan, dtype=np.float32)
        t = cfg.high_sim_threshold
        extra, enames = [], []
        # name high + addr low is the NEAR-MISS signature: same chain, different branch.
        # A single similarity value cannot express this conjunction.
        enames.append("both_high");      extra.append((ns > t) & (as_ > t))
        enames.append("name_high_addr_low"); extra.append((ns > t) & (as_ <= t))
        enames.append("name_low_addr_high"); extra.append((ns <= t) & (as_ > t))
        enames.append("sim_disagreement");   extra.append(np.abs(ns - as_))
        enames.append("min_field_sim");      extra.append(np.fmin(ns, as_))
        enames.append("mean_field_sim");     extra.append((ns + as_) / 2.0)
        M = np.hstack([M, np.vstack([np.asarray(e, dtype=np.float32) for e in extra]).T])
        cols += enames

    return M.astype(cfg.dtype), cols


def variance_report(M: np.ndarray, cols: Sequence[str]) -> List[dict]:
    """Zero variance means zero information. Report it -- a feature you EXPECTED to
    discriminate turning out constant is a finding about the data."""
    out = []
    for k, c in enumerate(cols):
        col = M[:, k]
        finite = col[np.isfinite(col)]
        out.append({
            "feature": c,
            "variance": float(np.var(finite)) if finite.size else 0.0,
            "pct_nan": round(100.0 * (1 - finite.size / max(col.size, 1)), 2),
            "constant": bool(finite.size == 0 or np.ptp(finite) == 0),
        })
    return out
