from __future__ import annotations

from collections.abc import Iterable
from itertools import product

from ..core.types import Factors
from .schema import (
    AMOUNTS,
    BASE_RENT,
    LANDLORD,
    RELATIONSHIP_QUALITY,
    TENANT,
)


def _landlord_factors() -> Iterable[Factors]:
    for amount, rq, tq, just in product(
        AMOUNTS, RELATIONSHIP_QUALITY, LANDLORD["tenant_quality"], LANDLORD["justification"]
    ):
        yield Factors(
            perspective="landlord",
            base_rent=BASE_RENT,
            amount=amount,
            relationship_quality=rq,
            tenant_quality=tq,
            justification=just,
        )


def _tenant_factors() -> Iterable[Factors]:
    for amount, rq, lq, just in product(
        AMOUNTS, RELATIONSHIP_QUALITY, TENANT["landlord_quality"], TENANT["justification"]
    ):
        yield Factors(
            perspective="tenant",
            base_rent=BASE_RENT,
            amount=amount,
            relationship_quality=rq,
            landlord_quality=lq,
            justification=just,
        )


def _neutral_factors() -> Iterable[Factors]:
    for amount, rq in product(AMOUNTS, RELATIONSHIP_QUALITY):
        yield Factors(
            perspective="neutral",
            base_rent=BASE_RENT,
            amount=amount,
            relationship_quality=rq,
            justification=None,
        )


def generate_factor_grid(include_neutral: bool = True) -> list[Factors]:
    items: list[Factors] = []
    items.extend(_landlord_factors())
    items.extend(_tenant_factors())
    if include_neutral:
        items.extend(_neutral_factors())
    return items
