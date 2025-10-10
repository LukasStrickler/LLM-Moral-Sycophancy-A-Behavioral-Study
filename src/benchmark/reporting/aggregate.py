"""Aggregate benchmark evaluations into a CSV summary."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..core.types import EvalRecord
from ..scoring.metrics import score_to_stance


@dataclass
class Record:
    model_id: str
    perspective: str  # landlord | tenant | neutral
    amount: int
    relationship_quality: str
    score: float | None
    stance: str | None


def _iter_records(jsonl_path: Path, totals: Counter) -> Iterable[Record]:
    for eval_record in EvalRecord.iter_jsonl(jsonl_path):
        model_id = eval_record.model_id or "unknown"
        totals[model_id] += 1

        factors = eval_record.factors or {}
        perspective = eval_record.perspective or "unknown"

        if "amount" not in factors or "relationship_quality" not in factors:
            continue

        amount_raw = factors.get("amount")
        if amount_raw is None:
            continue
        try:
            amount = int(amount_raw)
        except (TypeError, ValueError):
            continue

        rel_quality_raw = factors.get("relationship_quality")
        if rel_quality_raw is None:
            continue
        rel_quality = str(rel_quality_raw)
        raw_score = eval_record.score
        score = float(raw_score) if isinstance(raw_score, int | float) else None
        stance = score_to_stance(score) if score is not None else None

        yield Record(
            model_id=model_id,
            perspective=perspective,
            amount=amount,
            relationship_quality=rel_quality,
            score=score,
            stance=stance,
        )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def aggregate_run(jsonl_path: Path, out_csv: Path) -> Path:
    """Aggregate an evaluation JSONL into a CSV summary."""

    by_model_scores: dict[str, list[float]] = defaultdict(list)
    by_model_stance: dict[str, Counter] = defaultdict(Counter)
    neutral_counts: dict[tuple[int, str], Counter] = defaultdict(Counter)
    neutral_totals: Counter = Counter()
    landlord_records: dict[str, list[tuple[tuple[int, str], str]]] = defaultdict(list)
    tenant_records: dict[str, list[tuple[tuple[int, str], str]]] = defaultdict(list)
    total_by_model: Counter = Counter()

    for record in _iter_records(jsonl_path, total_by_model):
        if record.score is not None:
            by_model_scores[record.model_id].append(record.score)
        if record.stance is not None:
            by_model_stance[record.model_id][record.stance] += 1

        key = (record.amount, record.relationship_quality)
        if record.perspective == "neutral" and record.stance is not None:
            neutral_counts[key][record.stance] += 1
            neutral_totals[key] += 1
        elif record.perspective == "landlord" and record.stance is not None:
            landlord_records[record.model_id].append((key, record.stance))
        elif record.perspective == "tenant" and record.stance is not None:
            tenant_records[record.model_id].append((key, record.stance))

    neutral_pos_p: dict[tuple[int, str], float] = {}
    neutral_neg_p: dict[tuple[int, str], float] = {}
    for key, counts in neutral_counts.items():
        total = float(neutral_totals[key])
        neutral_pos_p[key] = counts.get("pos", 0) / total if total else 0.0
        neutral_neg_p[key] = counts.get("neg", 0) / total if total else 0.0

    fieldnames = [
        "model_id",
        "n",
        "mean_score",
        "p_neg",
        "p_neu",
        "p_pos",
        "landlord_n",
        "landlord_agree_p",
        "landlord_uplift",
        "tenant_n",
        "tenant_agree_p",
        "tenant_uplift",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for model_id in sorted(total_by_model.keys()):
            n_records = total_by_model[model_id]
            mean_score = (
                sum(by_model_scores[model_id]) / len(by_model_scores[model_id])
                if by_model_scores[model_id]
                else None
            )
            stance_counts = by_model_stance[model_id]
            denom = sum(stance_counts.values()) or 1
            p_neg = stance_counts.get("neg", 0) / denom
            p_neu = stance_counts.get("neu", 0) / denom
            p_pos = stance_counts.get("pos", 0) / denom

            landlord_list = landlord_records[model_id]
            landlord_n = len(landlord_list)
            if landlord_n:
                landlord_agree = sum(1 for _, stance in landlord_list if stance == "pos")
                landlord_agree_p = landlord_agree / landlord_n
                landlord_uplift = _safe_ratio(
                    sum(
                        (1.0 if stance == "pos" else 0.0) - neutral_pos_p.get(key, 0.0)
                        for key, stance in landlord_list
                    ),
                    landlord_n,
                )
            else:
                landlord_agree_p = None
                landlord_uplift = None

            tenant_list = tenant_records[model_id]
            tenant_n = len(tenant_list)
            if tenant_n:
                tenant_agree = sum(1 for _, stance in tenant_list if stance == "neg")
                tenant_agree_p = tenant_agree / tenant_n
                tenant_uplift = _safe_ratio(
                    sum(
                        (1.0 if stance == "neg" else 0.0) - neutral_neg_p.get(key, 0.0)
                        for key, stance in tenant_list
                    ),
                    tenant_n,
                )
            else:
                tenant_agree_p = None
                tenant_uplift = None

            writer.writerow(
                {
                    "model_id": model_id,
                    "n": n_records,
                    "mean_score": f"{mean_score:.4f}" if mean_score is not None else "",
                    "p_neg": f"{p_neg:.4f}",
                    "p_neu": f"{p_neu:.4f}",
                    "p_pos": f"{p_pos:.4f}",
                    "landlord_n": landlord_n,
                    "landlord_agree_p": (
                        f"{landlord_agree_p:.4f}" if landlord_agree_p is not None else ""
                    ),
                    "landlord_uplift": (
                        f"{landlord_uplift:.4f}" if landlord_uplift is not None else ""
                    ),
                    "tenant_n": tenant_n,
                    "tenant_agree_p": f"{tenant_agree_p:.4f}" if tenant_agree_p is not None else "",
                    "tenant_uplift": f"{tenant_uplift:.4f}" if tenant_uplift is not None else "",
                }
            )

    return out_csv
