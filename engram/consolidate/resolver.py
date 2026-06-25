"""Per-subject slot canonicalization — the reliability fix for cross-predicate updates (A1).

The conflict resolver (conflict.py) supersedes by SLOT = (user, subject, predicate). A single-valued
attribute updates correctly only when both facts land on the SAME predicate. But the LLM extractor emits
free-form predicates: "I'm on the Max plan" -> `subscription`/Max, later "switched to Pro" -> `plan`/Pro.
Different predicates -> different slots -> exact-slot misses -> BOTH stay live -> the answerer sees a
contradiction. The semantic (embedding) path is meant to catch this but is unreliable on short facts.

Fix: before reconciliation, map a new single-valued fact's predicate onto an EXISTING slot of the same
subject when they name the same attribute, so the deterministic exact-slot path fires. The live fact set
IS the registry — every live single-valued fact already defines a canonical slot, so no separate store is
needed. One LLM call is spent ONLY in the ambiguous zone (a candidate slot exists with a different value);
the decision is biased to NEW so a missed merge (safe — leaves both, same as today) is preferred over a
wrong merge (destroys a true value).
"""
from __future__ import annotations

from typing import Optional

from ..llm import LLM
from ..types import Fact
from .conflict import is_single_valued

_RESOLVE_SYSTEM = (
    "You curate a personal memory as attribute SLOTS about a subject. A NEW single-valued fact arrived. "
    "Decide whether it UPDATES one of the subject's existing slots (same attribute, new current value) or "
    "is a NEW attribute. Reply with ONLY the integer index of the slot it updates, or the word NEW. "
    "Single-valued = the subject can have one current value at a time (residence, employer, plan, "
    "marital_status). BIAS TO NEW: unless you are confident it is the SAME attribute as a listed slot, "
    "answer NEW — wrongly merging two different attributes destroys a true value, while a missed merge is "
    "harmless. Reply with the index or NEW, nothing else."
)


class SlotResolver:
    def __init__(self, llm: Optional[LLM] = None, gate: float = 0.5) -> None:
        self.llm = llm
        self.gate = gate  # min embedding similarity for a candidate slot to be worth adjudicating
        # (user, subject_lc, predicate_lc) -> canonical predicate. In-process memo so a repeated free-form
        # predicate isn't re-adjudicated every time; rebuildable from facts, so it need not persist.
        self.memo: dict[tuple[str, str, str], str] = {}

    def canonical(self, fact: Fact, live: list[Fact]) -> str:
        """Return the predicate `fact` should use so a same-attribute update lands on the existing slot.
        Returns fact.predicate unchanged for multi-valued logs, offline (no llm), or no candidate slot."""
        if self.llm is None or not is_single_valued(fact.predicate):
            return fact.predicate
        subj = fact.subject.lower()
        pred = fact.predicate.lower()
        # candidate = other single-valued slots of the same subject with a value (different predicate)
        cands: list[Fact] = []
        seen: set[str] = set()
        for f in live:
            if f.subject.lower() != subj or not is_single_valued(f.predicate):
                continue
            p = f.predicate.lower()
            if p == pred:
                return fact.predicate  # predicate already matches a live slot -> exact-slot handles it
            if p in seen:
                continue
            seen.add(p)
            cands.append(f)
        if not cands:
            return fact.predicate
        # only adjudicate candidates whose value embeds near the new fact (same attribute is the only case
        # worth an LLM call); without embeddings, fall back to adjudicating all candidates.
        if fact.embedding:
            from ..util import cosine

            near = [c for c in cands if c.embedding and cosine(fact.embedding, c.embedding) >= self.gate]
            cands = near
        if not cands:
            return fact.predicate
        cand_preds = {c.predicate.lower() for c in cands} | {pred}
        memo_key = (fact.user_id, subj, pred)
        cached = self.memo.get(memo_key)
        if cached is not None and cached.lower() in cand_preds:
            return cached
        chosen = self._adjudicate(fact, cands)
        self.memo[memo_key] = chosen
        return chosen

    def _adjudicate(self, fact: Fact, cands: list[Fact]) -> str:
        slots = "\n".join(f"  {i}. {c.predicate} = {c.object}" for i, c in enumerate(cands))
        prompt = (
            f"SUBJECT: {fact.subject}\n"
            f"EXISTING SLOTS:\n{slots}\n\n"
            f"NEW FACT: {fact.predicate} = {fact.object}\n\n"
            f"Index of the slot it updates, or NEW:"
        )
        try:
            ans = (self.llm.complete(prompt, system=_RESOLVE_SYSTEM) or "").strip()
        except Exception:  # noqa: BLE001 -- never let resolution break consolidation
            return fact.predicate
        # take the first integer token if present; anything else (incl. "NEW") -> keep its own predicate
        for tok in ans.replace(".", " ").replace(":", " ").split():
            if tok.isdigit():
                idx = int(tok)
                if 0 <= idx < len(cands):
                    return cands[idx].predicate
                break
        return fact.predicate


def demo() -> None:
    """Offline self-check: a Max->Pro update phrased with DIFFERENT predicates supersedes once the resolver
    canonicalizes onto the existing slot. Fake LLM returns '0' (the subscription slot) — no network."""
    from ..types import Fact as F
    from .conflict import ConflictResolver

    class FakeLLM:
        def complete(self, prompt, system=None, **k):
            return "0"  # the one existing slot

    res = SlotResolver(FakeLLM())
    old = F(subject="user", predicate="subscription", object="Max", user_id="u", valid_at=1.0, created_at=1.0)
    new = F(subject="user", predicate="plan", object="Pro", user_id="u", valid_at=2.0, created_at=2.0)

    # without canonicalization the slots differ -> no supersede (the A1 bug)
    cr = ConflictResolver(embedder=None)
    action, inv = cr.reconcile(new, [old])
    assert action == "add" and not inv, "precondition: free-form predicates should NOT supersede unaided"
    assert old.is_live(), "precondition"

    # with the resolver: predicate canonicalizes onto the existing slot -> exact-slot supersedes
    old2 = F(subject="user", predicate="subscription", object="Max", user_id="u", valid_at=1.0, created_at=1.0)
    new2 = F(subject="user", predicate="plan", object="Pro", user_id="u", valid_at=2.0, created_at=2.0)
    new2.predicate = res.canonical(new2, [old2])
    assert new2.predicate == "subscription", f"resolver should remap onto the slot, got {new2.predicate!r}"
    action, inv = ConflictResolver(embedder=None).reconcile(new2, [old2])
    assert action == "add" and inv == [old2], "remapped fact should supersede the old value"
    assert not old2.is_live() and new2.supersedes == old2.id, "old retired, chain points back"

    # under-merge safety: a DIFFERENT attribute (employer) with no matching slot keeps its own predicate
    emp = F(subject="user", predicate="works_at", object="Naver", user_id="u", valid_at=3.0, created_at=3.0)

    class FakeNew:
        def complete(self, prompt, system=None, **k):
            return "NEW"

    assert SlotResolver(FakeNew()).canonical(emp, [old2]) == "works_at", "different attribute must stay separate"
    print("resolver demo: OK")


if __name__ == "__main__":
    demo()
