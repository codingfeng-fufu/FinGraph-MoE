from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    expert: str
    text: str
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
