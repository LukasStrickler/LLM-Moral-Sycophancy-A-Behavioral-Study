from __future__ import annotations

from itertools import product
from typing import Literal

from ..core.types import Factors
from .relationship import compute_relationship, quality_to_affinity
from .schema import AMOUNTS, BASE_RENT, LANDLORD, TENANT

JustCat = Literal["economy", "costs", "timing"]


def _justify_category_for_landlord(code: str) -> JustCat:
    mapping = {
        "economy_tough": "economy",
        "rising_operational_costs": "costs",
        "budget_timing": "timing",
    }
    return mapping.get(code, "economy")


def _justify_category_for_tenant(code: str) -> JustCat:
    mapping = {
        "economy_tough": "economy",
        "rising_living_costs": "costs",
        "budget_timing": "timing",
    }
    return mapping.get(code, "economy")


def _landlord_code_for_category(cat: JustCat) -> str:
    return {
        "economy": "economy_tough",
        "costs": "rising_operational_costs",
        "timing": "budget_timing",
    }[cat]


def _tenant_code_for_category(cat: JustCat) -> str:
    return {
        "economy": "economy_tough",
        "costs": "rising_living_costs",
        "timing": "budget_timing",
    }[cat]


def generate_triplets(include_neutral: bool = True) -> list[Factors]:
    """Create matched landlord/tenant/neutral prompts per dimension set.

    Dimensions:
    - amount: AMOUNTS
    - justification categories: {economy, expenses, personal}
    - pair of qualities: landlord view of tenant x tenant view of landlord

    Relationship label is computed from both affinities and attached as
    the single `relationship_quality` value for all three perspectives.
    """

    cats: list[JustCat] = ["economy", "costs", "timing"]
    tenant_qualities = TENANT["landlord_quality"]  # good/poor
    landlord_qualities = LANDLORD["tenant_quality"]  # good/poor

    factors: list[Factors] = []
    for amount, cat, lq, tq in product(AMOUNTS, cats, tenant_qualities, landlord_qualities):
        rel_label, _ = compute_relationship(quality_to_affinity(lq), quality_to_affinity(tq))

        # landlord perspective
        factors.append(
            Factors(
                perspective="landlord",
                base_rent=BASE_RENT,
                amount=amount,
                relationship_quality=rel_label,
                tenant_quality=tq,
                justification=_landlord_code_for_category(cat),
            )
        )

        # tenant perspective
        factors.append(
            Factors(
                perspective="tenant",
                base_rent=BASE_RENT,
                amount=amount,
                relationship_quality=rel_label,
                landlord_quality=lq,
                justification=_tenant_code_for_category(cat),
            )
        )

        if include_neutral:
            factors.append(
                Factors(
                    perspective="neutral",
                    base_rent=BASE_RENT,
                    amount=amount,
                    relationship_quality=rel_label,
                    justification=None,
                )
            )

    return factors
