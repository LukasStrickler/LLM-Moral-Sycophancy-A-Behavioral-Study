from __future__ import annotations

from typing import Literal

Affinity = Literal["pos", "neu", "neg"]


def quality_to_affinity(q: str | None) -> Affinity:
    if q is None:
        return "neu"
    q_norm = q.lower()
    if q_norm in {"good", "positive", "+"}:
        return "pos"
    if q_norm in {"poor", "bad", "-"}:
        return "neg"
    return "neu"


def compute_relationship(landlord_affinity: Affinity, tenant_affinity: Affinity) -> tuple[str, int]:
    table = {
        ("pos", "pos"): ("good", 2),
        ("neg", "neg"): ("bad", -2),
        ("neu", "neu"): ("neutral", 0),
        ("pos", "neu"): ("leaning_good", 1),
        ("neu", "pos"): ("leaning_good", 1),
        ("neg", "neu"): ("leaning_bad", -1),
        ("neu", "neg"): ("leaning_bad", -1),
        ("pos", "neg"): ("one_sided_positive", 0),
        ("neg", "pos"): ("one_sided_negative", 0),
    }
    return table.get((landlord_affinity, tenant_affinity), ("neutral", 0))
