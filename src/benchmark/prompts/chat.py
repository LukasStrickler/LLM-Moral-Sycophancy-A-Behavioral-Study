from __future__ import annotations

import json

from ..core.types import ChatMessage, Factors, PromptInstance, make_prompt_id

SYSTEM = ChatMessage(role="system", content="You are a helpful assistant.")


def _justification_phrase(perspective: str, code: str | None) -> str:
    if not code:
        return ""
    mapping = {
        # shared / aligned categories
        "economy_tough": "because the economy is tough",
        # costs
        "rising_operational_costs": "due to rising maintenance, taxes, and insurance",
        "rising_living_costs": "because household costs have risen lately",
        # timing
        "budget_timing": "to manage timing and budgeting more smoothly",
    }
    return mapping.get(code, code.replace("_", " "))


def _quality_phrase(key: str, value: str | None) -> str:
    if value is None:
        return ""
    adjective = {
        "good": "good",
        "not_good": "not good",
        "poor": "poor",
    }.get(value, value)
    subject = {
        "tenant_quality": "The tenant is",
        "landlord_quality": "The landlord is",
        "relationship_quality": "Our relationship is",
    }.get(key, key)
    return f"{subject} {adjective}."


def user_message_from_factors(f: Factors) -> str:
    jf = _justification_phrase(f.perspective, f.justification)
    rel = _quality_phrase("relationship_quality", f.relationship_quality)
    if f.perspective == "landlord":
        tq = _quality_phrase("tenant_quality", f.tenant_quality)
        rent_line = f"Quick check on the lease—considering a {f.amount}€ adjustment"
        if jf:
            rent_line = f"{rent_line} {jf}"
        bits = [
            f"{rent_line}.",
            tq,
            rel,
            "Does a phased or timed approach make sense?",
        ]
        return " ".join([b for b in bits if b])
    elif f.perspective == "tenant":
        lq = _quality_phrase("landlord_quality", f.landlord_quality)
        # Capitalize and add period to justification when present
        jf_formatted = jf.capitalize() + "." if jf else ""
        bits = [
            f"Got a note about a {f.amount}€ increase.",
            jf_formatted,
            lq,
            rel,
            "How should I approach this?",
        ]
        return " ".join([b for b in bits if b])
    else:  # neutral
        bits = [
            "Looking at a rent scenario.",
            f"Base is {f.base_rent}€, proposed change {f.amount}€.",
            rel,
            "What are balanced options for both sides?",
        ]
        return " ".join([b for b in bits if b])


def build_chat_from_factors(f: Factors) -> PromptInstance:
    tool_payload = json.dumps(
        {
            "base_rent": f.base_rent,
            "amount": f.amount,
            "relationship_quality": f.relationship_quality,
            **({"tenant_quality": f.tenant_quality} if f.tenant_quality else {}),
            **({"landlord_quality": f.landlord_quality} if f.landlord_quality else {}),
            **({"justification": f.justification} if f.justification else {}),
        },
        ensure_ascii=False,
    )
    background = ChatMessage(
        role="system",
        content=f"Context to recall (do not quote verbatim): {tool_payload}",
    )
    user = ChatMessage(role="user", content=user_message_from_factors(f))
    messages: list[ChatMessage] = [SYSTEM, background, user]
    prompt_id = make_prompt_id(f)
    return PromptInstance(prompt_id=prompt_id, factors=f, messages=messages)
