from __future__ import annotations

from typing import Literal

Stance = Literal["neg", "neu", "pos"]


def score_to_stance(score: float, thr: float = 0.33) -> Stance:
    """Map a continuous score into a discrete stance bucket.

    Args:
        score: Continuous stance score in roughly [-1, 1].
        thr: Positive threshold that separates neutral from the extremes.

    Returns:
        "neg" when score is less than or equal to -thr, "pos" when score is greater
        than or equal to thr, otherwise "neu".

    Raises:
        ValueError: If ``thr`` is not strictly positive.
    """

    if thr <= 0:
        raise ValueError("thr must be > 0")

    if score <= -thr:
        return "neg"
    if score >= thr:
        return "pos"
    return "neu"
