"""Synthetic adversarial-input generator for the category router.

Used by the v4 training pipeline (Plan C) to expose the softmax router to
inputs that resemble nothing in the 7 real categories — short gibberish,
repeated symbols, number runs, fake-brand syllables. Without these, the
classifier confidently misroutes such strings to whichever class their
embedding happens to land near (typically ``cosmetics``, whose training
brands are themselves short invented syllables).

Sampling families
-----------------
``rand_letters``  : 3–14 Latin/Cyrillic letters, e.g. ``qwer asdf zxcv``.
``rand_cvcv``     : alternating consonant–vowel pseudo-words, e.g. ``bopulime``.
``rand_digits``   : digit runs with optional units, e.g. ``12345`` / ``99 87 11``.
``rand_symbols``  : punctuation/special-char noise, e.g. ``??? @@@ !!!``.
``repeated_char`` : single char repeated 5–14 times, e.g. ``aaaaaa``.
``mixed_noise``   : letters mixed with symbols/digits, e.g. ``ssad@@12``.

The mix is deliberately broad: real-world bad partner data includes empty
strings, copy-paste artefacts, placeholder tokens, and accidental ASCII
noise. Numbers / units alone (``500g``, ``250ml``) are *not* adversarial —
those are legitimate ``quantity`` fragments; we leave them as is.
"""
from __future__ import annotations

import random
import string

import pandas as pd

from src.pipeline.category_router.constants import ROUTER_INPUT_FIELDS

_LATIN_CONS = "bcdfghjklmnpqrstvwxz"
_LATIN_VOWS = "aeiouy"
_CYR_CONS = "бвгджзйклмнпрстфхцчшщ"
_CYR_VOWS = "аеёиоуыэюя"
_SYMBOLS = "?!@#$%^&*<>~_+=/"


def _rand_letters(rng: random.Random) -> str:
    """Многословный мусор из случайных букв (имитация длины реал-продуктов)."""
    n_words = rng.randint(2, 5)
    out = []
    alphabet = string.ascii_lowercase if rng.random() < 0.7 else "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
    for _ in range(n_words):
        out.append("".join(rng.choice(alphabet) for _ in range(rng.randint(4, 8))))
    return " ".join(out)


def _rand_cvcv(rng: random.Random) -> str:
    """Псевдо-слова CVCV, минимум 3 слова чтобы не путать с короткими брендами."""
    if rng.random() < 0.7:
        cons, vows = _LATIN_CONS, _LATIN_VOWS
    else:
        cons, vows = _CYR_CONS, _CYR_VOWS
    n_words = rng.randint(3, 5)
    words = []
    for _ in range(n_words):
        n_syll = rng.randint(2, 4)
        words.append("".join(rng.choice(cons) + rng.choice(vows) for _ in range(n_syll)))
    return " ".join(words)


def _rand_digits(rng: random.Random) -> str:
    """Цифровая помойка — много групп, никаких единиц измерения."""
    n_groups = rng.randint(3, 6)
    groups = []
    for _ in range(n_groups):
        length = rng.randint(3, 7)
        groups.append("".join(rng.choice(string.digits) for _ in range(length)))
    return " ".join(groups)


def _rand_symbols(rng: random.Random) -> str:
    """Длинные шумы из спецсимволов."""
    n = rng.randint(8, 18)
    return " ".join(rng.choice(_SYMBOLS) * rng.randint(2, 4) for _ in range(n // 3 or 1))


def _repeated_char(rng: random.Random) -> str:
    """Повторённые символы — гарантированно нечеловеческое."""
    ch = rng.choice(string.ascii_lowercase + string.digits)
    base = ch * rng.randint(10, 24)
    if rng.random() < 0.5:
        base = base + " " + ch * rng.randint(5, 12)
    return base


def _mixed_noise(rng: random.Random) -> str:
    """Длинное месиво букв/цифр/символов."""
    parts = []
    for _ in range(rng.randint(3, 6)):
        kind = rng.random()
        if kind < 0.4:
            parts.append("".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 8))))
        elif kind < 0.7:
            parts.append("".join(rng.choice(string.digits) for _ in range(rng.randint(2, 6))))
        else:
            parts.append(rng.choice(_SYMBOLS) * rng.randint(2, 4))
    return "".join(parts) if rng.random() < 0.5 else " ".join(parts)


GENERATORS = {
    "rand_letters": _rand_letters,
    "rand_cvcv": _rand_cvcv,
    "rand_digits": _rand_digits,
    "rand_symbols": _rand_symbols,
    "repeated_char": _repeated_char,
    "mixed_noise": _mixed_noise,
}


def sample_adversarial(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate ``n`` synthetic adversarial product_name strings.

    The returned DataFrame mirrors the schema used by
    :func:`sample_ood` so callers can ``pd.concat`` it directly into the
    router training table.

    Parameters
    ----------
    n    : number of adversarial samples to draw.
    seed : RNG seed for reproducibility.

    Returns
    -------
    DataFrame with columns
        product_name, brands, ingredients_text, quantity,
        category_label="garbage", brand="adv_<family>".
    """
    rng = random.Random(seed)
    families = list(GENERATORS.keys())
    rows = []
    for _ in range(n):
        family = rng.choice(families)
        name = GENERATORS[family](rng)
        rows.append({
            "product_name": name,
            "brands": "",
            "ingredients_text": "",
            "quantity": "",
            "category_label": "garbage",
            "brand": f"adv_{family}",
        })
    df = pd.DataFrame(rows)
    return df[list(ROUTER_INPUT_FIELDS) + ["category_label", "brand"]]
