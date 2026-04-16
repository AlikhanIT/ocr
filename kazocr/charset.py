from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CHARSET = (
    " "
    "0123456789"
    ".,!?;:-_()[]{}\"'/\\|@#$%&*+=<>"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "ÁáÄäÇçÉéĞğİiIıÑñŊŋÓóÖöŞşÚúÜüÝý"
)


@dataclass
class KazakhLatinCharset:
    alphabet: str = DEFAULT_CHARSET
    blank_token: str = "<blank>"

    def __post_init__(self) -> None:
        deduped = []
        seen = set()
        for ch in self.alphabet:
            if ch not in seen:
                deduped.append(ch)
                seen.add(ch)
        self.alphabet = "".join(deduped)
        self.tokens = [self.blank_token, *self.alphabet]
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.tokens)}
        self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[ch] for ch in text if ch in self.char_to_idx]

    def decode_ctc(self, indices: list[int]) -> str:
        result: list[str] = []
        prev = None
        for idx in indices:
            if idx != self.blank_id and idx != prev:
                result.append(self.idx_to_char[idx])
            prev = idx
        return "".join(result)
