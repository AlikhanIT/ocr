from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


WORD_RE = re.compile(r"[0-9A-Za-zÁÄÇÉĞİIıÑŊÓÖŞÚÜÝáäçéğiñŋóöşúüý'-]+", re.UNICODE)

ASCII_TO_KAZ = str.maketrans(
    {
        "c": "ç",
        "g": "ğ",
        "n": "ñ",
        "o": "ö",
        "u": "ü",
        "y": "ý",
    }
)

FOLD_MAP = str.maketrans(
    {
        "á": "a",
        "ä": "a",
        "ç": "c",
        "é": "e",
        "ğ": "g",
        "ı": "i",
        "İ": "I",
        "ñ": "n",
        "ŋ": "n",
        "ó": "o",
        "ö": "o",
        "ş": "s",
        "ú": "u",
        "ü": "u",
        "ý": "y",
        "Á": "A",
        "Ä": "A",
        "Ç": "C",
        "É": "E",
        "Ğ": "G",
        "Ñ": "N",
        "Ŋ": "N",
        "Ó": "O",
        "Ö": "O",
        "Ş": "S",
        "Ú": "U",
        "Ü": "U",
        "Ý": "Y",
    }
)

COMMON_CONFUSIONS = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "6": "g",
        "8": "b",
        "|": "i",
    }
)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def normalize_token(token: str) -> str:
    token = token.strip()
    token = token.translate(COMMON_CONFUSIONS)
    token = token.replace("’", "'").replace("`", "'")
    return token


def fold_token(token: str) -> str:
    return normalize_token(token).translate(FOLD_MAP).lower()


def restore_case(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


@dataclass
class CorrectionResult:
    raw_text: str
    corrected_text: str
    changed_tokens: list[tuple[str, str]]
    score: float = 0.0


class KazakhWordCorrector:
    def __init__(self, lexicon_path: str | None = None) -> None:
        default_path = Path(__file__).parent / "resources" / "kazakh_lexicon.txt"
        path = Path(lexicon_path) if lexicon_path else default_path
        self.words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.folded = {word: fold_token(word) for word in self.words}
        self.folded_values = set(self.folded.values())

    def looks_kazakhish(self, token: str) -> str:
        lowered = token.lower()
        if lowered in {"zh", "sh", "ch", "ng"}:
            return lowered.translate(ASCII_TO_KAZ)
        return token

    def best_match(self, token: str) -> str:
        cleaned = normalize_token(token)
        folded = fold_token(cleaned)
        if not folded or len(folded) <= 1 or any(ch.isdigit() for ch in folded):
            return cleaned
        if folded in self.folded_values:
            for word, folded_word in self.folded.items():
                if folded_word == folded:
                    return restore_case(token, word)

        best_word = cleaned
        best_score = 999
        for word, folded_word in self.folded.items():
            if abs(len(folded_word) - len(folded)) > 2:
                continue
            score = _edit_distance(folded, folded_word)
            if word[:1].lower() == cleaned[:1].lower():
                score -= 0.25
            if score < best_score:
                best_score = score
                best_word = word
        if best_score <= max(1.5, len(folded) * 0.34):
            return restore_case(token, best_word)
        return restore_case(token, self.looks_kazakhish(cleaned))

    def correct_text(self, text: str) -> CorrectionResult:
        changed_tokens: list[tuple[str, str]] = []

        def repl(match: re.Match[str]) -> str:
            raw = match.group(0)
            corrected = self.best_match(raw)
            if raw != corrected:
                changed_tokens.append((raw, corrected))
            return corrected

        corrected_text = WORD_RE.sub(repl, text)
        return CorrectionResult(
            raw_text=text,
            corrected_text=corrected_text,
            changed_tokens=changed_tokens,
            score=self.score_text(corrected_text),
        )

    def score_text(self, text: str) -> float:
        tokens = [token for token in WORD_RE.findall(text) if token.strip()]
        if not tokens:
            return -1.0
        lexicon_hits = 0
        alpha_chars = 0
        non_ascii_bonus = 0
        weird_penalty = 0
        total_len = 0
        for token in tokens:
            folded = fold_token(token)
            total_len += len(token)
            alpha_chars += sum(ch.isalpha() for ch in token)
            non_ascii_bonus += sum(ch in "áäçéğıñŋóöşúüýÁÄÇÉĞİÑŊÓÖŞÚÜÝ" for ch in token)
            if folded in self.folded_values:
                lexicon_hits += 1
            weird_penalty += sum(ch in "|[]{}_=+<>~" for ch in token)
        hit_ratio = lexicon_hits / max(1, len(tokens))
        alpha_ratio = alpha_chars / max(1, total_len)
        return hit_ratio * 4.0 + alpha_ratio + non_ascii_bonus * 0.03 - weird_penalty * 0.25
