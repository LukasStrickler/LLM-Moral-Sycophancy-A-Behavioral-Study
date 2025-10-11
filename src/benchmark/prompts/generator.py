from __future__ import annotations

from ..core.types import Factors
from .triplets import generate_triplets


def generate_factor_grid(include_neutral: bool = True) -> list[Factors]:
    """Return a flattened, triplet-matched grid.

    Each logical scenario yields: landlord + tenant (+ neutral) prompts that
    share amount, justification category, and a computed relationship label.
    """
    return generate_triplets(include_neutral=include_neutral)
