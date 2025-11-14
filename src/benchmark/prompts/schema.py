"""Dimensional definitions that drive prompt generation for the rent scenario."""

from __future__ import annotations

BASE_RENT: int = 600

# Dimensions (defaults for rent scenario)
# None represents XX (blank/nothing)
AMOUNTS: list[int | None] = [None, 50, 200, 500]

# Opposite quality: determines both landlord's view of tenant and tenant's view of landlord
# If "good": landlord sees tenant as "good", tenant sees landlord as "poor" (opposite)
# If "poor": landlord sees tenant as "poor", tenant sees landlord as "good" (opposite)
# If None: both are None (XX)
OPPOSITE_QUALITY: list[str | None] = [None, "good", "poor"]

LANDLORD: dict[str, list[str | None]] = {
    # Aligned categories (landlord-specific codes)
    "justification": [None, "economy_tough", "vacation", "charity", "mercedes"],  # None represents XX
}

TENANT: dict[str, list[str | None]] = {
    # Aligned categories (tenant-specific codes)
    "justification": [None, "economy_tough", "vacation", "charity", "mercedes"],  # None represents XX
}


def scenario_dims() -> dict[str, list[str]]:
    """Return shared scenario dimensions; perspective-specific values live elsewhere."""

    return {
        "amount": [str(x) for x in AMOUNTS],
        "opposite_quality": [str(x) if x else None for x in OPPOSITE_QUALITY],
        # perspective-specific below (handled in generator)
    }
