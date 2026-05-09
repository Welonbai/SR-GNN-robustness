from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class SuffixLengthBucket:
    name: str
    min_len: int
    max_len: int | None = None

    def matches(self, length: int) -> bool:
        normalized = int(length)
        if normalized < int(self.min_len):
            return False
        if self.max_len is None:
            return True
        return normalized <= int(self.max_len)


def default_suffix_length_buckets() -> tuple[SuffixLengthBucket, ...]:
    return (
        SuffixLengthBucket(name="suffix_1", min_len=1, max_len=1),
        SuffixLengthBucket(name="suffix_2", min_len=2, max_len=2),
        SuffixLengthBucket(name="suffix_3plus", min_len=3, max_len=None),
    )


def assign_suffix_length_group(
    length: int,
    buckets: Sequence[SuffixLengthBucket],
) -> str:
    normalized = int(length)
    if normalized < 1:
        raise ValueError("Residual suffix length must be >= 1.")
    matches = [bucket for bucket in buckets if bucket.matches(normalized)]
    if not matches:
        raise ValueError(
            f"No residual suffix length bucket matches length {normalized}."
        )
    if len(matches) > 1:
        names = ", ".join(bucket.name for bucket in matches)
        raise ValueError(
            f"Multiple residual suffix length buckets match length {normalized}: "
            f"{names}."
        )
    return str(matches[0].name)


__all__ = [
    "SuffixLengthBucket",
    "assign_suffix_length_group",
    "default_suffix_length_buckets",
]
