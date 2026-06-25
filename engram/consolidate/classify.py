"""Rule-first classification + sensitivity tagging (feature ⑤).

Deterministic and free — no LLM on the path (an LLM escalation can be layered on later, like conflict
adjudication). Assigns each fact a coarse domain CATEGORY (for organization + filtering) and a SENSITIVE
flag (PII / health / finance / credentials / protected attributes). Sensitive memories can then be redacted
from shared / export contexts by default — the general privacy-governance win for a memory product.
"""
from __future__ import annotations

import re

# category -> (predicate keywords, object/text keywords). First match wins; order = priority.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("health", ("allerg", "disease", "condition", "diagnos", "medication", "symptom", "pregnan", "disab",
              "diabet", "blood_type", "health")),
    ("finance", ("salary", "income", "wage", "bank", "credit_card", "debt", "net_worth", "mortgage",
              "invest")),
    ("identity", ("name", "age", "gender", "sex", "birthday", "born", "nationality", "occupation", "job",
              "profession", "education", "degree", "id_number", "passport", "ssn")),
    ("location", ("lives_in", "home", "address", "based_in", "located", "poi")),
    ("relationships", ("married", "spouse", "partner", "child", "kids", "family", "friend", "colleague", "boss",
              "relationship")),
    ("work", ("works_at", "employer", "company", "project", "works_on")),
    ("preferences", ("like", "love", "enjoy", "prefer", "favorite", "favourite", "dislike", "hate", "avoid",
              "interested", "fan_of")),
    ("events", ("visit", "went", "trip", "bought", "ordered", "attended", "plan", "booked")),
]

# Sensitivity: predicate or content hits a protected/PII class. Health, finance, credentials, and legally
# protected attributes (religion / politics / sexual orientation) are sensitive by default.
_SENSITIVE_KW: tuple[str, ...] = (
    # health
    "allerg", "disease", "diagnos", "medication", "diabet", "hiv", "cancer", "depress", "anxiety",
    "pregnan", "disab", "mental", "blood_type", "symptom",
    # finance
    "salary", "income", "wage", "bank_account", "credit_card", "debt", "net_worth", "mortgage",
    # credentials / national id
    "password", "passcode", "ssn", "passport", "id_number", "secret",
    # protected attributes
    "religion", "religious", "christian", "muslim", "buddhis", "political", "gay", "lesbian", "bisexual",
    "sexual_orientation",
)

_SENSITIVE_PREDS = {"salary", "income", "password", "ssn", "passport", "id_number", "religion",
                    "political_affiliation", "sexual_orientation", "medical_condition", "allergic_to",
                    "has_disease", "blood_type", "bank_account", "credit_card"}


def _hay(predicate: str, obj: str, text: str) -> str:
    return f"{predicate} {obj} {text}".lower()


def classify(predicate: str, obj: str = "", text: str = "") -> tuple[str, bool]:
    """Return (category, sensitive). Pure rules over predicate + object + rendered text."""
    p = predicate.lower()
    hay = _hay(p, obj, text)

    category = "other"
    for cat, kws in _CATEGORY_RULES:
        if any(k in hay for k in kws):
            category = cat
            break

    sensitive = (
        p in _SENSITIVE_PREDS
        or any(k in hay for k in _SENSITIVE_KW)
        # bare digit runs that look like an account/id/card number
        or bool(re.search(r"\b\d{6,}\b", obj))
    )
    return category, sensitive


def classify_fact(fact) -> None:
    """Set fact.category / fact.sensitive in place (idempotent — only fills when unset)."""
    if getattr(fact, "category", ""):
        return
    cat, sens = classify(fact.predicate, fact.object, fact.text)
    fact.category = cat
    # never downgrade a user-marked sensitive flag
    fact.sensitive = bool(getattr(fact, "sensitive", False) or sens)
