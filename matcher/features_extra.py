"""Feature families the vendored extractor does not cover.

Each one is here because a MEASUREMENT on this dataset says it matters, not because
it is a standard idea. Like `features.py`, this module must never see a label --
`leakage_check` enforces the same import boundary.

1. DOMAIN STEMS. 3.38% of S2 names and 3.36% of S3 names are bare web domains
   ("wilfordhancock.com"); S1 has 0%. For those pairs every string metric on the raw
   name is comparing a domain against a business name, so they all score near zero and
   the true match is invisible. Stripping the TLD and splitting on '.' / '-' recovers
   the tokens.

2. TRANSLITERATION COMPARISON. The vendored `multilingual` family only has
   `script_match` / `is_cross_script` FLAGS -- it never actually compares across
   scripts. 5.3% of S2 and 3.0% of S3 names are Devanagari while S1 is 0%, and 4.17% of
   true pairs are cross-script. We romanize with the SAME `transliterate_deva` the
   blocker used to build its P4 index, so the feature agrees with retrieval.

3. ADDRESS DEPTH. Addresses carry the 14.81% of true pairs with zero name-token
   overlap, and 96.3% of those have address Jaccard > 0.2. Three address features is
   not enough weight on the field that decides that slice.

4. SOURCE INDICATOR. S2 and S3 have measurably different noise profiles (Devanagari
   5.3% vs 3.0%). One free column.
"""
from __future__ import annotations
import re
from typing import Dict, List, Sequence, Tuple
import numpy as np

from .featconfig import FeatureConfig
from .features import (
    _cpdist, _jaccard, _containment, _dice, _ratio, normalize, core_tokens, _NUM,
)
from blocking.normalize import transliterate_deva, has_devanagari

try:
    from rapidfuzz import fuzz as rf_fuzz
    _HAS_RF = True
except ImportError:
    _HAS_RF = False

# A bare domain as a business name: no spaces, at least one dot, a plausible TLD.
_DOMAIN = re.compile(r'^[a-z0-9][a-z0-9.\-]*\.'
                     r'(com|net|org|in|co|io|biz|info|us|uk|fr|shop|online|site|store)$')
_TLD_TAIL = re.compile(r'\.(com|net|org|in|co|io|biz|info|us|uk|fr|shop|online|site|store)$')
_SPLIT = re.compile(r'[.\-_]+')
# Common subdomain noise that carries no business identity.
_SUBDOM = {'www', 'web', 'shop', 'store', 'online', 'my', 'the'}


# Second-level labels that appear in two-part TLDs ('.co.in', '.co.uk', '.ac.in').
# Left in place they leak a meaningless 'co' token into the stem.
_TLDISH = {'com', 'net', 'org', 'in', 'co', 'io', 'biz', 'info', 'us', 'uk', 'fr',
           'shop', 'online', 'site', 'store', 'ac', 'gov', 'edu'}


def domain_stem(raw_name: str) -> str:
    """'' when the name is not a bare domain. Otherwise its identity-bearing tokens.

    Detected on the RAW string, deliberately, NOT on the normalized one. `normalize()`
    turns '.' into a space, so after it 'wilfordhancock.com' and 'Wilford Hancock Com'
    are indistinguishable -- and a legitimate name ending in 'Co' would be misread as a
    '.co' domain. A bare domain has no spaces; that is the reliable signal, and it only
    exists before normalization.
    """
    s = (raw_name or '').strip().lower()
    if not s or ' ' in s or '.' not in s:
        return ''
    if not _DOMAIN.match(s):
        return ''
    toks = [t for t in _SPLIT.split(s) if t]
    while toks and toks[-1] in _TLDISH:          # strip '.co.in' one label at a time
        toks.pop()
    while toks and toks[0] in _SUBDOM:           # strip 'www.', 'shop.'
        toks.pop(0)
    return ' '.join(toks)


def prepare_extra(names: Sequence[str], addrs: Sequence[str],
                  cfg: FeatureConfig) -> Dict[str, list]:
    """Per-record work for the families above. Records << pairs, so this is cheap."""
    n_norm = [normalize(x) for x in names]
    a_norm = [normalize(x) for x in addrs]

    stems = [domain_stem(x) for x in names]
    # Effective name: the domain stem when the name IS a domain, else the core tokens.
    # This is the string a cross-source comparison should actually use.
    eff = [s if s else ' '.join(core_tokens(x)) for s, x in zip(stems, n_norm)]

    # Romanize only what needs it; transliterate_deva is only correct on Devanagari.
    tr = [normalize(transliterate_deva(raw)) if has_devanagari(raw) else nn
          for raw, nn in zip(names, n_norm)]

    a_toks = [set(x.split()) for x in a_norm]
    return {
        'dom_stem': stems,
        'is_domain': [bool(s) for s in stems],
        'eff_name': eff,
        'eff_tokens': [set(x.split()) for x in eff],
        'translit': tr,
        'translit_tokens': [set(core_tokens(x)) for x in tr],
        'addr_norm': a_norm,
        'addr_tokens': a_toks,
        # last address token is usually the state / region: a cheap coarse-geo check
        'addr_last': [x.split()[-1] if x.split() else '' for x in a_norm],
        'addr_first_num': [(_NUM.search(x).group(0) if _NUM.search(x) else '')
                           for x in a_norm],
    }


def addr_idf(token_sets: Sequence[set]) -> Dict[str, float]:
    """Address-token IDF. Unsupervised corpus statistic, not leakage."""
    df: Dict[str, int] = {}
    for s in token_sets:
        for t in s:
            df[t] = df.get(t, 0) + 1
    n = max(len(token_sets), 1)
    return {t: float(np.log((n + 1) / (d + 1)) + 1.0) for t, d in df.items()}


def extract_extra(li: np.ndarray, ri: np.ndarray,
                  L: Dict[str, list], R: Dict[str, list],
                  cfg: FeatureConfig,
                  source_tag: Sequence[str] | None = None,
                  a_idf: Dict[str, float] | None = None) -> Tuple[np.ndarray, List[str]]:
    cols: List[str] = []
    feats: List[np.ndarray] = []

    def add(name: str, arr) -> None:
        cols.append(name)
        feats.append(np.asarray(arr, dtype=np.float32))

    l_eff = [L['eff_name'][i] for i in li]; r_eff = [R['eff_name'][j] for j in ri]
    l_et = [L['eff_tokens'][i] for i in li]; r_et = [R['eff_tokens'][j] for j in ri]
    l_dom = [L['is_domain'][i] for i in li]; r_dom = [R['is_domain'][j] for j in ri]
    l_tr = [L['translit'][i] for i in li];   r_tr = [R['translit'][j] for j in ri]
    l_tt = [L['translit_tokens'][i] for i in li]; r_tt = [R['translit_tokens'][j] for j in ri]
    l_at = [L['addr_tokens'][i] for i in li]; r_at = [R['addr_tokens'][j] for j in ri]
    l_al = [L['addr_last'][i] for i in li];  r_al = [R['addr_last'][j] for j in ri]
    l_an = [L['addr_first_num'][i] for i in li]; r_an = [R['addr_first_num'][j] for j in ri]

    # ---- 1. domain-aware name comparison ----
    add('is_domain_l', l_dom)
    add('is_domain_r', r_dom)
    add('is_domain_either', [a or b for a, b in zip(l_dom, r_dom)])
    if _HAS_RF:
        add('eff_name_ratio', _cpdist(l_eff, r_eff, rf_fuzz.ratio, cfg))
        add('eff_name_token_set_ratio', _cpdist(l_eff, r_eff, rf_fuzz.token_set_ratio, cfg))
    add('eff_token_jaccard', [_jaccard(a, b) for a, b in zip(l_et, r_et)])
    add('eff_token_contain_fwd', [_containment(a, b) for a, b in zip(l_et, r_et)])
    add('eff_token_contain_rev', [_containment(b, a) for a, b in zip(l_et, r_et)])

    # ---- 2. transliterated comparison ----
    if _HAS_RF:
        add('translit_ratio', _cpdist(l_tr, r_tr, rf_fuzz.ratio, cfg))
        add('translit_token_set_ratio', _cpdist(l_tr, r_tr, rf_fuzz.token_set_ratio, cfg))
    add('translit_token_jaccard', [_jaccard(a, b) for a, b in zip(l_tt, r_tt)])
    add('translit_token_contain', [_containment(a, b) for a, b in zip(l_tt, r_tt)])

    # ---- 3. address depth ----
    add('addr_contain_fwd', [_containment(a, b) for a, b in zip(l_at, r_at)])
    add('addr_contain_rev', [_containment(b, a) for a, b in zip(l_at, r_at)])
    add('addr_dice', [_dice(a, b) for a, b in zip(l_at, r_at)])
    add('addr_last_exact', [bool(a) and a == b for a, b in zip(l_al, r_al)])
    add('addr_first_num_equal', [bool(a) and a == b for a, b in zip(l_an, r_an)])
    add('addr_first_num_conflict',
        [bool(a) and bool(b) and a != b for a, b in zip(l_an, r_an)])
    la = np.array([len(x) for x in l_at], dtype=np.float32)
    ra = np.array([len(x) for x in r_at], dtype=np.float32)
    add('addr_token_count_ratio', _ratio(la, ra))
    if a_idf:
        add('addr_idf_overlap',
            [sum(a_idf.get(t, 0.0) for t in (a & b)) for a, b in zip(l_at, r_at)])
        add('addr_max_idf_shared',
            [max((a_idf.get(t, 0.0) for t in (a & b)), default=0.0)
             for a, b in zip(l_at, r_at)])

    # ---- 4. which source the candidate came from ----
    if source_tag is not None:
        add('cand_is_S2', [t == 'S2' for t in source_tag])

    M = np.vstack(feats).T.astype(cfg.dtype)
    return M, cols


def _test() -> None:
    cfg = FeatureConfig()
    assert domain_stem('wilfordhancock.com') == 'wilfordhancock'
    assert domain_stem('www.ram-trading.co.in') == 'ram trading', \
        'two-part TLD must not leak a "co" token'
    assert domain_stem('Ram Trading Private Limited') == '', \
        'a normal business name must not be treated as a domain'
    # the case that made raw-string detection necessary: a real name ending in 'Co'
    assert domain_stem('Shree Stores Co') == '', \
        'a spaced name ending in Co is not a .co domain'
    assert domain_stem('') == ''

    names_l = ['Wilford Hancock', 'Ram Trading Private Limited', 'Shree Stores']
    addrs_l = ['12 Mack Rd, Haltom City, Texas', 'KH NO 570 New Delhi, Delhi', '']
    names_r = ['wilfordhancock.com', 'राम ट्रेडिंग प्राइवेट लिमिटेड', 'Shree Stores']
    addrs_r = ['Mack Rd, Haltom City, Texas', 'KH NO 570, NEW DELHI, Delhi', '']

    L = prepare_extra(names_l, addrs_l, cfg)
    R = prepare_extra(names_r, addrs_r, cfg)
    li = np.array([0, 1, 2]); ri = np.array([0, 1, 2])
    M, cols = extract_extra(li, ri, L, R, cfg, source_tag=['S2', 'S3', 'S2'],
                            a_idf=addr_idf(L['addr_tokens'] + R['addr_tokens']))
    c = {name: k for k, name in enumerate(cols)}

    # pair 0: a domain on the right. The raw names share nothing a metric can use;
    # the stem makes them near-identical.
    assert M[0, c['is_domain_r']] == 1 and M[0, c['is_domain_l']] == 0
    assert M[0, c['eff_name_ratio']] > 0.9, \
        f"domain stem must recover the name match, got {M[0, c['eff_name_ratio']]}"

    # pair 1: cross-script. Romanization is what makes it comparable at all.
    assert M[1, c['translit_ratio']] > 0.5, \
        f"transliteration must make the cross-script pair comparable, got {M[1, c['translit_ratio']]}"

    # pair 2: both addresses empty -> undefined, must be NaN and never 0.0 or 1.0
    assert np.isnan(M[2, c['addr_contain_fwd']]), 'empty-vs-empty address must be NaN'
    assert np.isnan(M[2, c['addr_dice']])
    assert M[2, c['addr_last_exact']] == 0

    # address state token agrees on the first two pairs
    assert M[0, c['addr_last_exact']] == 1 and M[1, c['addr_last_exact']] == 1
    assert M[0, c['cand_is_S2']] == 1 and M[1, c['cand_is_S2']] == 0

    print(f'features_extra self-test: PASS  ({len(cols)} extra features)')


if __name__ == '__main__':
    _test()
