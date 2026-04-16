from __future__ import annotations

import random
from pathlib import Path


DEFAULT_CORPUS = [
    "Qazaq tili",
    "Jana kun",
    "Alem men adam",
    "Uly dala",
    "Bilim jane enbek",
    "Taza paraq",
    "Korkem metin",
    "San men sozdik",
    "Mektep pen gylym",
    "Qala jane auyl",
    "Aqyl men jurek",
    "Sapa joǵary bolsyn",
    "Qazir oqyp otyrmyn",
    "Mınau synaq joly",
    "Qazaq latin qarpi",
    "Adebiet jane tarih",
    "Jaqsy natije kerek",
    "Óner men madeniet",
    "Ülgili jobalar",
    "Ádemi jazu ulgisi",
]


def load_corpus(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_CORPUS
    content = Path(path).read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in content if line.strip()]
    return lines or DEFAULT_CORPUS


def sample_text(corpus: list[str], min_words: int = 1, max_words: int = 5) -> str:
    if not corpus:
        return "Qazaq tili"
    if random.random() < 0.45:
        return random.choice(corpus)
    words = []
    target = random.randint(min_words, max_words)
    while len(words) < target:
        words.extend(random.choice(corpus).split())
    return " ".join(words[:target])
