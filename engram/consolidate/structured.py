"""L2 structured profile — a typed, grouped VIEW over the live facts (CLAUDE.md §3, the structured layer).

Design constraints (the "reasonable version" we agreed on):
  * DERIVED, not a separate source of truth. Built read-only from the live fact set, so it inherits
    provenance and can never drift from the bi-temporal store.
  * DISPLAY-ONLY tiering. We split items into `confirmed` vs `tentative` purely for presentation
    (so a shaky one-off guess like "might like jazz" isn't shown as part of the canonical profile).
    This NEVER filters retrieval — search()/lean_context() still see every fact, so recall is untouched.
  * NO invented weights. Each item carries an HONEST evidence descriptor (you set it / mentioned N times),
    not a made-up 0.0–1.0 score. Confidence the user can actually reason about.

Promotion to `confirmed` happens when the evidence is real: the user asserted it, it's an explicit
favorite/allergy, it was reinforced on access, or it was stated across ≥2 independent sessions.
"""
from __future__ import annotations

from ..types import Fact

# predicate -> (canonical field key, human label). Identity / basic-info slots (single-valued).
_BASIC: dict[str, tuple[str, str]] = {
    "name": ("name", "Name"),
    "age": ("age", "Age"), "age_range": ("age", "Age"),
    "gender": ("gender", "Gender"), "sex": ("gender", "Gender"),
    "occupation": ("occupation", "Occupation"), "job": ("occupation", "Occupation"),
    "profession": ("occupation", "Occupation"), "works_as": ("occupation", "Occupation"),
    "works_at": ("employer", "Employer"), "employer": ("employer", "Employer"),
    "company": ("employer", "Employer"),
    "lives_in": ("home", "Home"), "home": ("home", "Home"),
    "based_in": ("home", "Home"), "lives_at": ("home", "Home"),
    "born_in": ("birthplace", "Birthplace"), "birthplace": ("birthplace", "Birthplace"),
    "birthday": ("birthday", "Birthday"), "born_on": ("birthday", "Birthday"), "birth_date": ("birthday", "Birthday"),
    "married_to": ("spouse", "Spouse"), "spouse": ("spouse", "Spouse"),
    "has_child": ("children", "Children"), "has_children": ("children", "Children"),
    "children": ("children", "Children"), "kids": ("children", "Children"),
    "nationality": ("nationality", "Nationality"),
    "speaks": ("language", "Language"), "language": ("language", "Language"),
    "education": ("education", "Education"), "studied_at": ("education", "Education"),
    "graduated_from": ("education", "Education"), "degree": ("education", "Education"),
}

_POSITIVE = {"likes", "like", "loves", "love", "enjoys", "enjoy", "prefers", "prefer", "favorite",
             "favourite", "interested_in", "fan_of", "into", "fond_of", "wants", "wishes_for"}
_NEGATIVE = {"dislikes", "dislike", "hates", "hate", "avoids", "avoid", "allergic_to", "allergic",
             "cannot_eat", "cant_eat", "not_into", "disinterested_in"}
_HABIT = {"usually", "often", "routine", "regularly", "habit", "tends_to", "commutes", "frequently",
          "daily", "weekly", "every_day", "every_week", "every_morning"}

# coarse category buckets for grouping preferences (keyword match on predicate + object)
# keyword -> category. First match wins, so ORDER matters: a phrase mentioning both a hotel and a pool
# stays under travel because travel is listed before sports.
_CATEGORY_KW: list[tuple[str, tuple[str, ...]]] = [
    ("health_restrictions", ("allergic", "allergy", "intoleran")),
    ("food", ("food", "eat", "cuisine", "dish", "restaurant", "drink", "coffee", "tea", "spicy",
              "seafood", "meal", "cook", "snack", "fruit")),
    ("travel", ("travel", "trip", "destination", "hotel", "route", "drive", "flight", "scenery", "poi")),
    ("music", ("music", "song", "artist", "singer", "band", "genre", "playlist")),
    ("film_tv", ("movie", "film", "show", "tv", "series", "video", "cinema", "drama", "director")),
    ("sports", ("sport", "exercise", "gym", "run", "fitness", "yoga", "basketball", "football", "hike")),
    ("reading", ("book", "read", "author", "novel", "podcast", "news")),
    ("hobbies", ("dance", "photography", "selfie", "baking", "painting", "crafts")),
    ("environment", ("quiet", "atmosphere", "ambience", "noisy", "air_conditioning", "temperature", "seating")),
]


_POS_PREFIX = ("favorite", "favourite", "likes_", "loves_", "enjoys_", "prefers_", "prefer_",
               "interested", "fond_", "fan_")
_NEG_PREFIX = ("dislikes_", "dislike_", "hates_", "avoids_", "avoid_", "allergic", "cannot_",
               "doesn't_", "does_not_", "do_not_", "not_", "no_")


def _polarity(pred: str) -> str | None:
    """like / dislike polarity of a preference predicate — handles compound predicates the LLM emits
    (likes_quiet_environments, doesn't_drive_on_elevated_roads, ...) not just the canonical verbs."""
    p = pred.lower()
    if p in _POSITIVE or p.startswith(_POS_PREFIX):
        return "like"
    if p in _NEGATIVE or p.startswith(_NEG_PREFIX):
        return "dislike"
    return None


def _is_habit(pred: str) -> bool:
    p = pred.lower()
    return p in _HABIT or any(p.startswith(h + "_") for h in _HABIT)


def _category(pred: str, obj: str) -> str:
    p = pred.lower()
    if p.startswith("favorite_") or p.startswith("favourite_"):
        suffix = p.split("_", 1)[1]
        for cat, kws in _CATEGORY_KW:
            if any(k in suffix for k in kws):
                return cat
    hay = f"{p} {obj.lower()}"
    for cat, kws in _CATEGORY_KW:
        if any(k in hay for k in kws):
            return cat
    return "other"


def _evidence(f: Fact) -> dict:
    """An HONEST, user-legible confidence signal — not a fabricated numeric weight."""
    if f.source == "user":
        return {"kind": "user", "count": 1}
    n = len(set(f.provenance)) or 1
    if f.access_count > 0 and n < 2:
        return {"kind": "reinforced", "count": f.access_count}
    return {"kind": "mentions", "count": n}


def _confirmed(f: Fact) -> bool:
    """Whether a preference shows in the canonical profile vs sits as a tentative candidate (display-only,
    never gates retrieval). An EXPLICITLY STATED preference is confirmed — the user said it, it isn't a
    shaky inference (a user-asserted write is treated as confidence 1.0). Tentative is reserved for
    genuinely weak signals (e.g. a future passive-inference path emitting sub-1.0 confidence)."""
    if f.source == "user":
        return True
    if _polarity(f.predicate) is not None:  # an explicitly stated like / dislike / favorite / allergy
        return True
    if len(set(f.provenance)) >= 2 or f.access_count >= 1:  # corroborated across sessions / reinforced
        return True
    return False


# Universal first-person / user references (EN + KO + JA coreference). These denote THE USER — not a
# special case but the basic identity-resolution every memory system needs. Facts about other people
# (son/grandmother/friend ...) simply aren't in this set, so they're excluded without any kinship list.
USER_REFS = {"user", "i", "me", "myself", "나", "내", "제", "私", "僕", "俺", "わたし", "ぼく"}


def user_aliases(subject: str, user_id: str) -> set[str]:
    """Subjects that denote the user: the resolved self-name + the user_id + USER_REFS.
    Relies on extraction having normalized first-person subjects to the self-name (see LLMExtractor)."""
    return {subject.lower(), user_id.lower()} | USER_REFS


def build_structured_profile(facts: list[Fact], subject: str, user_id: str = "default") -> dict:
    """Group the user's live facts into basic info / weighted-free preferences / habits, split into
    confirmed vs tentative for display. `facts` should already be the live set."""
    who = user_aliases(subject, user_id)
    mine = [f for f in facts if f.subject.lower() in who]

    basic: dict[str, dict] = {}
    prefs: dict[str, list] = {}
    habits: list[dict] = []
    tentative: list[dict] = []

    for f in mine:
        pred = f.predicate.lower()
        # 1) basic identity slots (single-valued: keep the best-evidenced per field)
        if pred in _BASIC:
            field, label = _BASIC[pred]
            cand = {"field": field, "label": label, "value": f.object,
                    "evidence": _evidence(f), "source": f.source, "fact_id": f.id}
            cur = basic.get(field)
            if cur is None or _rank(f) > cur["_rank"]:
                cand["_rank"] = _rank(f)
                basic[field] = cand
            continue
        # 2) preferences (polarity-tagged), grouped by coarse category
        pol = _polarity(pred)
        if pol is not None:
            item = {"item": f.object, "polarity": pol, "category": _category(pred, f.object),
                    "evidence": _evidence(f), "source": f.source, "fact_id": f.id,
                    "subject": f.subject, "predicate": f.predicate, "object": f.object}
            if _confirmed(f):
                prefs.setdefault(item["category"], []).append(item)
            else:
                tentative.append(item)
            continue
        # 3) habits / routines (light)
        if _is_habit(pred):
            habits.append({"text": f.text, "evidence": _evidence(f), "fact_id": f.id})

    # The user's name is resolved into the identity (self-name) rather than stored as a fact, so surface
    # it in basic info explicitly when we have it (and no name fact already provided one).
    if "name" not in basic and subject and subject.lower() not in (USER_REFS | {user_id.lower()}):
        basic["name"] = {"field": "name", "label": "Name", "value": subject,
                         "evidence": {"kind": "mentions", "count": 1}, "source": "extracted", "fact_id": ""}
    for b in basic.values():
        b.pop("_rank", None)
    return {
        "basic": list(basic.values()),
        "preferences": prefs,
        "habits": habits,
        "tentative": tentative,
        "counts": {
            "basic": len(basic),
            "preferences": sum(len(v) for v in prefs.values()),
            "tentative": len(tentative),
            "habits": len(habits),
        },
    }


def _rank(f: Fact) -> tuple:
    """Pick the most trustworthy fact for a single-valued slot: user-asserted first, then most
    corroborated, then most recently valid."""
    return (1 if f.source == "user" else 0, len(set(f.provenance)), f.valid_at)
