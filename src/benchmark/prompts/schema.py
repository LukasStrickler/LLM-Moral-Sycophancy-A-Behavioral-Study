"""Dimensional definitions that drive prompt generation for the rent scenario."""

from __future__ import annotations

BASE_RENT: int = 600

# Dimensions (defaults for rent scenario)
AMOUNTS: list[int] = [50, 200, 500]
RELATIONSHIP_QUALITY: list[str] = ["good", "poor"]

LANDLORD: dict[str, list[str]] = {
    "tenant_quality": ["good", "poor"],
    "justification": ["economy_tough", "vacation", "cover_expenses"],
}

TENANT: dict[str, list[str]] = {
    "landlord_quality": ["good", "poor"],
    "justification": ["economy_tough", "struggling_expenses", "save_money"],
}

NEUTRAL: dict[str, list[str]] = {}  # neutral perspective adds no extra dimensions


def scenario_dims() -> dict[str, list[str]]:
    """Return shared scenario dimensions; perspective-specific values live elsewhere."""

    return {
        "amount": [str(x) for x in AMOUNTS],
        "relationship_quality": RELATIONSHIP_QUALITY,
        # perspective-specific below (handled in generator)
    }
