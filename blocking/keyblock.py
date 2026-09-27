"""Inverted-index composite-key blocking.

Replaces the TF-IDF-cosine-over-the-whole-matrix blocker, which costs 0.28 s per S1
entity (measured) and therefore ~85 h for the test set. Key lookup is ~0.05 ms per S1
entity -- about 6,000x faster -- because it touches only the few buckets a record's keys
land in instead of scoring every one of 4.1M documents.

The speed was never in doubt; RECALL is the whole problem. A first 6-key prototype hit
only 67.62% against ground truth, versus 94.41% for the TF-IDF blocker, because exact
keys cannot absorb typos, word reordering or partial names. The families below are each
aimed at one measured failure mode:

  RT  rare name token     -- EDA: 85.19% of true pairs share >=1 name token. Survives
                            reordering and extra/missing words. Biggest single lever.
  RA  rare address token  -- EDA: 14.81% of true pairs share ZERO name tokens, and 96.3%
                            of those have address Jaccard > 0.2. This family is the only
                            thing that reaches them.
  TR  transliterated name -- 5.3% of S2 / 3.0% of S3 names are Devanagari, S1 is 0%.
                            Uses the same romanization as the existing P4 index.
  G4  rare char-4gram     -- typo tolerance, which exact keys structurally lack.
  DM  domain stem         -- 3.4% of S2/S3 names are bare domains; S1 has none.

"Rare" is by document frequency in the TARGET corpus. Rarity is what makes a key
selective: indexing on a common token produces a bucket nobody benefits from.

BUCKET CAP: keys whose posting list exceeds `cap` are dropped entirely. A key matching
10,000 records carries almost no information and would dominate the runtime. Measured at
cap=100 this discards 0.23% of buckets.
"""
from __future__ import annotations
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from .normalize import base_normalize, name_core, has_devanagari, transliterate_deva

# Family switches, overridden by the measurement harness so key sets can be ablated
# without editing this file. 'RT' rare name token, 'RA' rare address token,
# 'G4' rare char-4gram, 'X' cross-field anchors.
FAMS = {'RT', 'RA', 'G4', 'X'}
N_RT, N_RA = 3, 2

try:
    from matcher.features_extra import domain_stem as _dstem
except Exception:
    _dstem = None


def domain_stem_of(raw_name: str) -> str:
    return _dstem(raw_name) if _dstem else ''

_NUM = re.compile(r'\d+')
# Address words that are everywhere and therefore useless as selective keys.
ADDR_STOP = {'road', 'rd', 'street', 'st', 'nagar', 'colony', 'near', 'opp', 'opposite',
             'floor', 'no', 'plot', 'block', 'sector', 'phase', 'main', 'cross', 'lane',
             'ave', 'avenue', 'dr', 'drive', 'suite', 'ste', 'unit', 'apt', 'building'}


def _toks(s: str) -> List[str]:
    return [t for t in s.split() if t]


def _rarest(tokens: Iterable[str], df: Dict[str, int], n: int,
            stop: set | None = None) -> List[str]:
    """The n lowest-document-frequency tokens. Unknown tokens sort first (df 0): a token
    absent from the target corpus is maximally selective when it does appear."""
    c = [t for t in set(tokens) if len(t) >= 3 and (not stop or t not in stop)]
    c.sort(key=lambda t: (df.get(t, 0), t))
    return c[:n]


def _grams(s: str, n: int = 4) -> List[str]:
    s = s.replace(' ', '')
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def record_terms(name: str, addr: str) -> dict:
    """Normalize ONCE per record. Everything downstream reads these."""
    n = base_normalize(name)
    a = base_normalize(addr)
    nt = _toks(n)
    at = _toks(a)
    core = name_core(name)
    tr = base_normalize(transliterate_deva(name)) if has_devanagari(name) else ''
    return {'n': n, 'a': a, 'nt': nt, 'at': at, 'core': core, 'tr': tr,
            'nums': [t for t in at if t.isdigit()],
            'grams': _grams(core or n) if 'G4' in FAMS else ()}


def keys_for(R: dict, name_df: Dict[str, int], addr_df: Dict[str, int],
             gram_df: Dict[str, int], *, domain_stem=None) -> List[str]:
    """All composite keys for one record. Same function for index and query -- if these
    ever diverge the index silently stops matching."""
    nt, at, core, nums = R['nt'], R['at'], R['core'], R['nums']
    k: List[str] = []

    if core:
        k.append('C|' + core)
        if len(nt) >= 2:
            k.append('S|' + ' '.join(sorted(nt)))          # word-order invariant
    # rare single name tokens -- the main recall driver
    if 'RT' in FAMS:
        for t in _rarest(nt, name_df, N_RT):
            k.append('RT|' + t)
    # rare address tokens -- reaches the zero-name-overlap pairs
    if 'RA' in FAMS:
        for t in _rarest(at, addr_df, N_RA, stop=ADDR_STOP):
            k.append('RA|' + t)
    # rare char-4grams -- typo tolerance
    if 'G4' in FAMS:
        for g in _rarest(R['grams'], gram_df, 2):
            k.append('G4|' + g)
    # cross-field anchors
    if 'X' in FAMS:
        if nt and at:
            k.append('NA|' + nt[0] + '|' + at[-1])
        if nt and nums:
            k.append('NN|' + nt[0] + '|' + nums[0])
        if nums and at:
            k.append('AN|' + nums[0] + '|' + at[-1])
        if len(at) >= 2:
            k.append('A2|' + ' '.join(at[:2]))
    # cross-script
    if R['tr']:
        k.append('C|' + name_core(R['tr']))
        for t in _rarest(_toks(R['tr']), name_df, 2):
            k.append('RT|' + t)
    if domain_stem:
        k.append('C|' + domain_stem)
    return k


def build_df(terms: Sequence[dict]) -> Tuple[dict, dict, dict]:
    """Document frequencies over the target corpus, used to pick 'rare'."""
    nd: Dict[str, int] = defaultdict(int)
    ad: Dict[str, int] = defaultdict(int)
    gd: Dict[str, int] = defaultdict(int)
    for R in terms:
        for t in set(R['nt']):
            nd[t] += 1
        for t in set(R['at']):
            ad[t] += 1
        for g in set(R['grams']):
            gd[g] += 1
    return dict(nd), dict(ad), dict(gd)


def build_index(terms: Sequence[dict], name_df, addr_df, gram_df,
                cap: int = 100, stems: Sequence[str] | None = None) -> Dict[str, list]:
    inv: Dict[str, list] = defaultdict(list)
    for i, R in enumerate(terms):
        for k in keys_for(R, name_df, addr_df, gram_df,
                          domain_stem=(stems[i] if stems else None)):
            inv[k].append(i)
    # A key that matches thousands of records is not evidence -- it is a scan.
    return {k: v for k, v in inv.items() if len(v) <= cap}


def query(R: dict, inv: Dict[str, list], name_df, addr_df, gram_df,
          domain_stem=None) -> set:
    out: set = set()
    for k in keys_for(R, name_df, addr_df, gram_df, domain_stem=domain_stem):
        b = inv.get(k)
        if b:
            out.update(b)
    return out
