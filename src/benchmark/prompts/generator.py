from __future__ import annotations

from ..core.types import Factors
from .triplets import generate_triplets


def generate_factor_grid() -> list[Factors]:
    """Return a flattened, matched grid.

    Each logical scenario yields: landlord + tenant prompts that
    share amount, justification category, and a computed relationship label.
    """
    return generate_triplets()
