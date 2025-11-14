from __future__ import annotations

from itertools import product
from typing import Literal

from ..core.types import Factors
from .relationship import compute_relationship, quality_to_affinity
from .schema import AMOUNTS, BASE_RENT, LANDLORD, TENANT, OPPOSITE_QUALITY

JustCat = Literal["economy", "vacation", "charity", "mercedes"]


def _justify_category_for_landlord(code: str) -> JustCat:
    mapping = {
        "economy_tough": "economy",
        "vacation": "vacation",
        "charity": "charity",
        "mercedes": "mercedes",
    }
    return mapping.get(code, "economy")  # type: ignore[return-value]


def _justify_category_for_tenant(code: str) -> JustCat:
    mapping = {
        "economy_tough": "economy",
        "vacation": "vacation",
        "charity": "charity",
        "mercedes": "mercedes",
    }
    return mapping.get(code, "economy")  # type: ignore[return-value]


def _landlord_code_for_category(cat: JustCat) -> str:
    return {
        "economy": "economy_tough",
        "vacation": "vacation",
        "charity": "charity",
        "mercedes": "mercedes",
    }[cat]


def _tenant_code_for_category(cat: JustCat) -> str:
    return {
        "economy": "economy_tough",
        "vacation": "vacation",
        "charity": "charity",
        "mercedes": "mercedes",
    }[cat]


def _sort_key(f: Factors) -> tuple:
    """Sort key for factors: perspective, amount (None first), opposite_quality, justification."""
    # Perspective: landlord < tenant (alphabetically)
    perspective_order = {"landlord": 0, "tenant": 1}
    
    # Amount: None first (as lowest), then ascending numeric values
    amount_value = (-1 if f.amount is None else f.amount)
    
    # Opposite quality: None < "good" < "poor"
    quality_order = {None: 0, "good": 1, "poor": 2}
    
    # Justification: None first, then alphabetically
    justification_value = ("" if f.justification is None else f.justification)
    
    return (
        perspective_order.get(f.perspective, 999),
        amount_value,
        quality_order.get(f.opposite_quality, 999),
        justification_value,
    )


def generate_triplets() -> list[Factors]:
    """Create matched landlord/tenant prompts per dimension set.

    Dimensions:
    - amount: AMOUNTS (can be None for XX)
    - justification categories: {economy, vacation, charity, mercedes} or None for XX
    - opposite_quality: single dimension that sets opposite views
      - If "good": landlord sees tenant as "good", tenant sees landlord as "poor"
      - If "poor": landlord sees tenant as "poor", tenant sees landlord as "good"
      - If None: both are None (XX)
    
    Returns factors sorted by: perspective (landlord, tenant), amount (None first, then ascending),
    opposite_quality (None, good, poor), justification (None first, then alphabetically).
    """

    cats: list[JustCat | None] = [None, "economy", "vacation", "charity", "mercedes"]

    factors: list[Factors] = []
    for amount, cat, opposite_qual in product(AMOUNTS, cats, OPPOSITE_QUALITY):
        # Map category to justification code, handling None
        landlord_justification = None if cat is None else _landlord_code_for_category(cat)
        tenant_justification = None if cat is None else _tenant_code_for_category(cat)

        # landlord perspective
        factors.append(
            Factors(
                perspective="landlord",
                base_rent=BASE_RENT,
                amount=amount,  # Can be None for XX
                opposite_quality=opposite_qual,  # Single dimension for both perspectives
                justification=landlord_justification,
            )
        )

        # tenant perspective
        factors.append(
            Factors(
                perspective="tenant",
                base_rent=BASE_RENT,
                amount=amount,  # Can be None for XX
                opposite_quality=opposite_qual,  # Same opposite_quality for matched pair
                justification=tenant_justification,
            )
        )

    # Sort factors: perspective, amount (None first), opposite_quality, justification
    factors.sort(key=_sort_key)
    
    return factors
