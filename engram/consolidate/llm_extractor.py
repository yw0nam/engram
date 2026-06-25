"""LLM-backed fact extractor. Same `.extract(episode) -> list[Fact]` API as the offline RuleExtractor,
so the entire pipeline (graph build, conflict resolution, retrieval) is unchanged -- only the extraction
quality goes up. Used automatically when a Memory is given an `llm`."""
from __future__ import annotations

import json
import re

from ..llm import LLM
from ..types import Episode, Fact

EXTRACT_SYSTEM = (
    "You are a precise information-extraction engine for a long-term memory system. "
    "From a multi-turn conversation, extract the atomic, durable facts it states about the user and the "
    "people/things they mention (identities, attributes, preferences, relationships, possessions, "
    "goals/plans, and events with their times). Output ONLY a JSON array of objects, each with keys "
    "\"subject\", \"predicate\", \"object\", \"text\". Use short snake_case predicates (e.g. works_at, lives_in, "
    "favorite_color, owns, married_to, born_in, visited). Capture PREFERENCES explicitly and completely "
    "with predicates like likes, dislikes, prefers, avoids, allergic_to, favorite_<thing> "
    "(e.g. likes/'spicy food', dislikes/'crowds', prefers/'window seat', allergic_to/'peanuts'). "
    "For each preference or dislike stated, output a SEPARATE fact. "
    "Resolve first-person ('I','my','me') to the user's name when it is known in the conversation, "
    "otherwise to \"user\". Capture a stated name as "
    "{\"subject\":\"user\",\"predicate\":\"name\",\"object\":\"<Name>\"}. "
    "LANGUAGE: the user converses in Korean, Japanese, or English. Write the predicate AND the \"text\" "
    "sentence in ENGLISH regardless of the input language. For subject and object VALUES (names, places, "
    "brands, products), use the common English spelling when one exists; keep the value in its original "
    "form only when no common English spelling exists. Example: "
    "{\"subject\":\"Jin\",\"predicate\":\"works_at\",\"object\":\"Naver\",\"text\":\"Jin works at Naver\"}. "
    "A STANDING PREFERENCE or POLICY counts even when phrased as a command, a negation, or a question, and "
    "even when stated only in passing inside another request — a fact embedded in an instruction is still a "
    "fact. A ONE-OFF TASK COMMAND is NOT durable. The test: would this still be true and useful next week? "
    "standing preference/policy/identity -> record it; transient action -> output nothing. Examples:\n"
    "Input: Don't use Korean — only Japanese or English for reports from now on.\n"
    "Output: [{\"subject\":\"user\",\"predicate\":\"prefers\",\"object\":\"Japanese or English\",\"text\":\"The user prefers Japanese or English over Korean\"}]\n"
    "Input: Stop reporting Claude cost — that's just my Claude Code Max subscription, not API spend.\n"
    "Output: [{\"subject\":\"user\",\"predicate\":\"subscription_plan\",\"object\":\"Claude Code Max\",\"text\":\"The user is on the Claude Code Max plan\"}]\n"
    "Input: PRs only to my own repo, never to upstream.\n"
    "Output: [{\"subject\":\"user\",\"predicate\":\"prefers\",\"object\":\"PRs only to own repo\",\"text\":\"The user prefers PRs only to their own repo\"}]\n"
    "Input: Check the gateway error logs and restart it if needed.\n"
    "Output: []\n"
    "Input: Did you do that, or did my fix make it pass?\n"
    "Output: []\n"
    "Do NOT infer or invent facts that are not stated. If there are no durable facts, output []."
)

EXTRACT_TEMPLATE = "Conversation:\n{content}\n\nJSON facts:"

# predicates that declare a name/nickname for the user — all registered as the user's aliases so any of
# them used later as a subject normalizes to one canonical identity (the user may have several names).
_NAME_PREDICATES = {"name", "name_is", "is_named", "called", "nickname", "call_me", "preferred_name",
                    "name_preference"}
# first-person / user references normalized to the user's canonical name (EN + KO + JA coreference)
_SELF_REFS = {"user", "i", "me", "myself", "나", "내", "제", "私", "僕", "俺", "わたし", "ぼく"}


def _norm_predicate(p: str) -> str:
    p = p.strip().lower().replace("-", " ").replace("/", " ")
    p = re.sub(r"\s+", "_", p).strip("_")
    return p


def parse_json_facts(raw: str) -> list[dict]:
    """Tolerant JSON-array parsing: handle code fences and surrounding prose."""
    if not raw:
        return []
    text = raw.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("["), text.rfind("]")
    candidate = text[start : end + 1] if (start != -1 and end > start) else text
    for attempt in (candidate, text):
        try:
            data = json.loads(attempt)
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            continue
    return []


class LLMExtractor:
    def __init__(self, llm: LLM, system: str = EXTRACT_SYSTEM, template: str = EXTRACT_TEMPLATE) -> None:
        self.llm = llm
        self.system = system
        self.template = template
        # Optional per-user directive appended to the system prompt: WHAT the user wants recorded (or not).
        # Set from Memory.policy["extract_instruction"] — the headline "what to record" knob in the console.
        self.instruction = ""
        self.self_name: dict[str, str] = {}        # user_id -> canonical name (the first one declared)
        self.aliases: dict[str, set] = {}          # user_id -> {all declared names/nicknames} (coreference)

    def self_of(self, user_id: str) -> str:
        return self.self_name.get(user_id, user_id)

    def _effective_system(self) -> str:
        if self.instruction.strip():
            return (self.system + "\n\nADDITIONAL USER DIRECTIVE on what to record — obey it, but still "
                    "output ONLY the JSON array described above:\n" + self.instruction.strip())
        return self.system

    def extract(self, ep: Episode) -> list[Fact]:
        raw = self.llm.complete(self.template.format(speaker=ep.speaker, content=ep.content),
                                system=self._effective_system())
        facts: list[Fact] = []
        for item in parse_json_facts(raw):
            subj = str(item.get("subject", "")).strip()
            pred = _norm_predicate(str(item.get("predicate", "")))
            obj = str(item.get("object", "")).strip()
            if not subj or not pred or not obj:
                continue
            if pred in _NAME_PREDICATES:
                # register every declared name/nickname as a user alias; the FIRST one is the canonical
                # subject. So all of the user's names plus first-person refs normalize to one identity below.
                self.aliases.setdefault(ep.user_id, set()).add(obj.lower())
                self.self_name.setdefault(ep.user_id, obj)
                continue
            if subj.lower() in _SELF_REFS or subj.lower() in self.aliases.get(ep.user_id, ()):
                # first-person / user reference OR any of the user's declared names ->
                # normalize to ONE canonical subject, so all of the user's own facts share it (otherwise they
                # split across subjects and the profile / conflict-resolution can't tell they're the user's).
                subj = self.self_of(ep.user_id) if self.self_of(ep.user_id) != ep.user_id else subj
            facts.append(
                Fact(
                    subject=subj,
                    predicate=pred,
                    object=obj,
                    # one-line phrasing from the model, shown in the UI. The canonical English-predicate
                    # `text` is still built for embedding/retrieval.
                    display=str(item.get("text", "")).strip(),
                    user_id=ep.user_id,
                    valid_at=ep.event_time,
                    created_at=ep.ingested_at,
                    provenance=[ep.id],
                )
            )
        return facts
