"""Urdu text normalization for ASR training and evaluation.

Why this exists: Urdu is written with the Perso-Arabic script, but real-world
transcripts mix Arabic/Urdu codepoints (e.g. Arabic kaf U+0643 vs Urdu kaf
U+06A9, alef variants, stray diacritics, kashida). Without normalization,
word error rate (WER) punishes the model for orthographic noise instead of
real transcription errors.

The rules below follow common Urdu NLP practice:
  - strip tashkeel / diacritics (harakat)
  - collapse alef variants to bare alef
  - map Arabic yeh/kaf to their Urdu forms
  - remove kashida (tatweel)
  - normalize digits and punctuation (optional)
"""

import re
import unicodedata

# Arabic diacritics (tashkeel): FATHATAN..SUPERSCRIPT ALEF
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")

# Tatweel / kashida
_KASHIDA_RE = re.compile(r"\u0640")

# Whitespace runs
_WS_RE = re.compile(r"\s+")

# Alef variants -> bare alef (U+0627).
# NOTE: U+0622 (alef-madda, آ) is intentionally NOT mapped — it is a distinct
# Urdu letter (word-initial /aː/), not orthographic noise.
_ALEF_MAP = {
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW
    "\u0671": "\u0627",  # ALEF WASLA
}

# Yeh / kaf variants -> Urdu forms
_YEH_KAF_MAP = {
    "\u064a": "\u06cc",  # ARABIC YEH -> FARSI YEH (Urdu yeh)
    "\u0649": "\u06cc",  # ALEF MAKSURA -> FARSI YEH
    "\u0643": "\u06a9",  # ARABIC KAF -> KEHEH (Urdu kaf)
}

# Heh variants -> Urdu heh (U+06C1). Arabic heh (U+0647) and teh marbuta
# (U+0629) render identically to Urdu heh but are different codepoints, so
# unnormalized text double-counts the same word in WER.
# NOTE: do-chashmi he (U+06BE, as in بھ) is a DISTINCT Urdu letter and is
# left untouched.
_HEH_MAP = {
    "\u0647": "\u06c1",  # ARABIC HEH -> HEH GOAL (Urdu heh)
    "\u0629": "\u06c1",  # TEH MARBUTA -> HEH GOAL
}

# Eastern Arabic-Indic digits (Urdu) -> ASCII digits
_DIGIT_MAP = {chr(0x06F0 + i): str(i) for i in range(10)}
# Arabic-Indic digits (sometimes mixed in) -> ASCII digits
_DIGIT_MAP.update({chr(0x0660 + i): str(i) for i in range(10)})

# Punctuation stripped for WER-style normalization.
_PUNCT_RE = re.compile(
    r"["
    r"\u060c"  # ARABIC COMMA ،
    r"\u061b"  # ARABIC SEMICOLON ؛
    r"\u061f"  # ARABIC QUESTION MARK ؟
    r"\u06d4"  # ARABIC FULL STOP ۔
    r"\u0021-\u002f"  # ASCII !"#$%&'()*+,-./
    r"\u003a-\u0040"  # ASCII :;<=>?@
    r"\u005b-\u0060"  # ASCII [\]^_`
    r"\u007b-\u007e"  # ASCII {|}~
    r"\u2018-\u201f"  # curly quotes
    r"\u2026"  # ellipsis
    r"]+"
)


def normalize_urdu(
    text: str,
    remove_diacritics: bool = True,
    normalize_digits: bool = True,
    remove_punctuation: bool = True,
) -> str:
    """Normalize Urdu text to a canonical orthographic form.

    Args:
        text: Raw Urdu (or mixed Arabic/Urdu) text.
        remove_diacritics: Strip tashkeel/harakat.
        normalize_digits: Map Eastern Arabic-Indic digits to ASCII.
        remove_punctuation: Strip punctuation (recommended for WER scoring).

    Returns:
        Normalized string with collapsed whitespace.
    """
    if not text:
        return ""

    # Unicode NFC first so composed/decomposed forms agree.
    text = unicodedata.normalize("NFC", text)

    if remove_diacritics:
        text = _DIACRITICS_RE.sub("", text)

    # Kashida (elongation character used for justification).
    text = _KASHIDA_RE.sub("", text)

    # Character-level canonical mappings.
    text = "".join(_ALEF_MAP.get(ch, ch) for ch in text)
    text = "".join(_YEH_KAF_MAP.get(ch, ch) for ch in text)
    text = "".join(_HEH_MAP.get(ch, ch) for ch in text)

    if normalize_digits:
        text = "".join(_DIGIT_MAP.get(ch, ch) for ch in text)

    if remove_punctuation:
        text = _PUNCT_RE.sub(" ", text)

    # Collapse whitespace and strip.
    text = _WS_RE.sub(" ", text).strip()
    return text


if __name__ == "__main__":
    # Small self-test: run `python src/normalize.py` from the repo root.
    cases = [
        # (raw, expected)
        ("مَیں نے کِتاب پڑھی۔", "میں نے کتاب پڑھی"),          # diacritics + full stop
        ("آپ کیسے ہیں؟", "آپ کیسے ہیں"),                    # alef-madda -> alef
        ("كيا حال هے", "کیا حال ہے"),                        # arabic kaf/yeh -> urdu forms
        ("خوبــصورت", "خوبصورت"),                             # kashida removal
        ("میں ۱۲۳ روپے", "میں 123 روپے"),                    # urdu digits -> ascii
        ("اُردُو، زبان!", "اردو زبان"),                        # punctuation strip
        ("بھائی", "بھائی"),                                   # do-chashmi he untouched
    ]
    failed = 0
    for raw, expected in cases:
        got = normalize_urdu(raw)
        status = "OK  " if got == expected else "FAIL"
        if got != expected:
            failed += 1
        print(f"{status} {raw!r} -> {got!r} (expected {expected!r})")
    raise SystemExit(1 if failed else 0)
