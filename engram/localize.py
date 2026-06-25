"""Display rendering — render a Fact for display WITHOUT touching the canonical English predicate (the
slot key that makes cross-language conflict resolution work). Data model = canonical; presentation =
rendered. This is purely a display layer: retrieval, dedup, and conflict logic are unchanged.

The user works in Korean / Japanese / English (never Chinese), and the canonical `text` is already
English, so display rendering simply returns the readable English statement — we never force-translate
someone's data.
"""
from __future__ import annotations

_USER_SUBJECTS = {"user", "i", "me", "myself"}


def render_display(subject: str, predicate: str, obj: str, text: str = "", lang: str = "auto") -> str:
    """Readable display string for a fact. Returns the canonical English `text` when present, otherwise a
    simple English rendering of (subject, predicate, object)."""
    if text:
        return text
    body = f"{predicate.replace('_', ' ')} {obj}".strip()
    if subject.lower() in _USER_SUBJECTS:
        return body
    return f"{subject} {body}".strip()


def display_of(fact, lang: str = "auto") -> str:
    # Prefer the extractor's one-line phrasing (recorded for ANY predicate); fall back to rendering the
    # canonical predicate for old facts / offline extraction.
    native = (getattr(fact, "display", "") or "").strip()
    if native:
        return native
    return render_display(fact.subject, fact.predicate, fact.object, getattr(fact, "text", ""), lang)
