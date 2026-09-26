"""Normalization helpers for candidate generation.

Design constraints from PROJECT_CONTEXT / FINDINGS:
- Devanagari appears in S2 (5.35%) and S3 (2.99%). `[^\\w\\s]` shatters it because
  Devanagari vowel signs / virama are Unicode categories Mc/Mn which `\\w` does not
  match. Strip punctuation by Unicode category instead (P/S/Z, excluding whitespace),
  keeping marks. Self-tested at the bottom of this file.
- Legal-suffix vocabulary spans US + FR + IN + DE; we drop them for the "name core".
- Normalization is ADDITIVE: we always keep the original string too. Callers may need it.
"""
from __future__ import annotations
import re
import unicodedata
from typing import Iterable

# Devanagari script range for cross-script detection.
DEVA_RE = re.compile(r'[ऀ-ॿ]')

# Legal / stop suffixes, deliberately multi-country.
SUFFIX: frozenset[str] = frozenset({
    # English / US
    'inc', 'incorporated', 'llc', 'ltd', 'limited', 'corp', 'corporation',
    'co', 'company', 'llp', 'lp', 'plc', 'holdings', 'group',
    # France
    'sarl', 'sarlu', 'sas', 'sasu', 'sa', 'eurl', 'snc', 'scop', 'scp',
    # India
    'pvt', 'private', 'pvtltd',
    # Germany / EU
    'gmbh', 'ag', 'kg', 'ohg', 'bv', 'nv', 'ab', 'oy', 'as', 'aps',
    # Filler
    'and', 'the', 'of', 'et',
})

# ASCII address-token stopwords (kept small — address recall matters).
ADDR_STOPWORDS: frozenset[str] = frozenset({
    'st', 'street', 'rd', 'road', 'ave', 'avenue', 'blvd', 'boulevard',
    'ln', 'lane', 'dr', 'drive', 'ct', 'court', 'pl', 'place',
    'suite', 'ste', 'apt', 'unit', 'floor', 'fl',
    'n', 's', 'e', 'w', 'north', 'south', 'east', 'west',
})


def strip_punct_by_category(s: str) -> str:
    """Replace every Unicode Punctuation / Symbol / Separator character with a
    single space, but KEEP Mark characters (Mn/Mc/Me) so Devanagari matra + virama
    survive. Whitespace is left alone.

    This is the fix for FINDING 4 in eda/FINDINGS.md. Do not replace with `\\w`.
    """
    out = []
    for c in s:
        if c.isspace():
            out.append(c)
            continue
        cat = unicodedata.category(c)  # e.g. 'Lo', 'Mn', 'Po', 'Sc', 'Zs'
        top = cat[0]
        if top in ('P', 'S', 'Z'):
            out.append(' ')
        else:
            out.append(c)
    return ''.join(out)


def base_normalize(s: str) -> str:
    """NFKC + casefold + ampersand + category-aware punctuation strip + collapse whitespace.

    Preserves script (Devanagari stays Devanagari). Deterministic and idempotent.
    """
    if not s:
        return ''
    s = unicodedata.normalize('NFKC', s).casefold().replace('&', ' and ')
    s = strip_punct_by_category(s)
    # collapse whitespace deterministically
    return ' '.join(s.split())


def name_tokens(s: str, drop_suffix: bool = True) -> list[str]:
    """Whitespace tokens after base_normalize. Optional suffix drop.
    If suffix drop empties the list, fall back to raw tokens (never return [])."""
    toks = base_normalize(s).split()
    if not drop_suffix:
        return toks
    kept = [t for t in toks if t not in SUFFIX]
    return kept if kept else toks


def name_core(s: str) -> str:
    """Canonical string used for the P1 exact-match index. Sorted tokens (order-invariant).

    Sorting removes the "acme foods" vs "foods acme" false-negative without any
    fuzzy step. Deterministic.
    """
    return ' '.join(sorted(name_tokens(s, drop_suffix=True)))


def addr_tokens(s: str, drop_stopwords: bool = True) -> list[str]:
    """Address tokens: same normalization; optional street-word stopword drop.
    Preserves numerals (they are the highest-signal address tokens)."""
    toks = base_normalize(s).split()
    if not drop_stopwords:
        return toks
    kept = [t for t in toks if t not in ADDR_STOPWORDS]
    return kept if kept else toks


def has_devanagari(s: str) -> bool:
    return bool(DEVA_RE.search(s))


def transliterate_deva(s: str) -> str:
    """Character-level romanization for the cross-script blocking key.

    Tries `indic_transliteration` (better quality) first, falls back to `unidecode`.
    Called ONLY on strings that contain Devanagari.
    """
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate as _t
        return _t(s, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    except Exception:
        try:
            from unidecode import unidecode
            return unidecode(s).lower()
        except Exception:
            return s  # last-resort: leave as-is; still safe


def country_key(c: str) -> str:
    """Country partition key. Deterministic, case-insensitive, whitespace-collapsed.
    NO fuzzy matching — findings say true pairs never cross country."""
    if not c:
        return ''
    return ' '.join(unicodedata.normalize('NFKC', c).casefold().split())


# ---------------------------------------------------------------------------
# Self-test — run `python normalize.py` to verify Devanagari + suffix behavior.
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # 1. Devanagari must survive as multi-char tokens.
    s = 'राम मार्केटिंग प्राइवेट लिमिटेड'
    toks = name_tokens(s, drop_suffix=False)
    assert len(toks) == 4, f'expected 4 Devanagari tokens, got {toks}'
    assert all(len(t) >= 3 for t in toks), f'Devanagari shattered: {toks}'
    # 2. Suffix stripping.
    assert name_core('Acme Foods Inc') == name_core('Foods Acme LLC'), \
        f'{name_core("Acme Foods Inc")} vs {name_core("Foods Acme LLC")}'
    # 3. Ampersand mapped.
    assert 'and' in name_tokens('AT & T', drop_suffix=False)
    # 4. Category-based strip vs \\w.
    naive = re.sub(r'[^\w\s]', ' ', s).split()
    smart = name_tokens(s, drop_suffix=False)
    assert len(smart) < len(naive), \
        f'category-based strip should produce FEWER tokens than \\w for Devanagari'
    # 5. Idempotence.
    assert base_normalize(base_normalize(s)) == base_normalize(s)
    # 6. Empty inputs.
    assert base_normalize('') == ''
    assert name_core('') == ''
    assert country_key('') == ''
    # 7. Country key.
    assert country_key(' UNITED  states ') == country_key('United States')
    # 8. Devanagari detection.
    assert has_devanagari(s) and not has_devanagari('Acme Foods Inc')
    print('normalize.py self-test OK')
    print(f'  Devanagari tokens: {toks}')
    print(f'  name_core("Acme Foods Inc") = {name_core("Acme Foods Inc")!r}')
    print(f'  transliterate({s!r}) = {transliterate_deva(s)!r}')


if __name__ == '__main__':
    _selftest()
