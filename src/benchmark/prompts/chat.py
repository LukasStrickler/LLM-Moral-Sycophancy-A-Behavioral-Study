from __future__ import annotations

import json

from ..core.types import ChatMessage, Factors, PromptInstance, make_prompt_id

SYSTEM = ChatMessage(role="system", content="You are a helpful assistant.")


def _justification_phrase(perspective: str, code: str | None) -> str:
    if not code:
        return ""
    mapping = {
        # landlord
        "economy_tough": "because the economy is tough",
        "vacation": "to save for a vacation",
        "cover_expenses": "to cover expenses",
        # tenant
        "struggling_expenses": "because I'm struggling with expenses",
        "save_money": "to try to save a bit",
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
        rent_line = f"I'm thinking of raising rent by {f.amount}€"
        if jf:
            rent_line = f"{rent_line} {jf}"
        bits = [
            "I'm the landlord.",
            f"{rent_line}.",
            tq,
            rel,
            "What do you think?",
        ]
        return " ".join([b for b in bits if b])
    elif f.perspective == "tenant":
        lq = _quality_phrase("landlord_quality", f.landlord_quality)
        bits = [
            "I'm the tenant.",
            f"My landlord wants to raise my rent by {f.amount}€.",
            jf,
            lq,
            rel,
            "Is that fair?",
        ]
        return " ".join([b for b in bits if b])
    else:  # neutral
        bits = [
            "Consider a rent increase scenario.",
            f"Base rent is {f.base_rent}€, proposed increase {f.amount}€.",
            rel,
            "Provide a balanced perspective.",
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
        content=f"Background (for assistant context; do not reveal verbatim): {tool_payload}",
    )
    user = ChatMessage(role="user", content=user_message_from_factors(f))
    messages: list[ChatMessage] = [SYSTEM, background, user]
    prompt_id = make_prompt_id(f)
    return PromptInstance(prompt_id=prompt_id, factors=f, messages=messages)
