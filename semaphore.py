"""
ZGA Semaphore — Agent State Router
==================================
PLAIN ENGLISH: This is the "traffic light" of the agent. The interpreter has five
"states of being" (CONDUIT, CLARIFIER, CULTURAL_BROKER, RESEARCHER, ADVOCATE). This
file decides which one is active at any moment, so the UI can light up the right badge.

HOW IT DECIDES: We do NOT guess from tone or length. The interpreter is trained to
SAY a specific phrase out loud before it intervenes (e.g. "Interpreter speaking..." or
"Interpreter pause..."). We simply listen for those exact phrases in what the model
just said, and switch the state accordingly.

RULES:
- Default is always CONDUIT (faithful translation). We return to CONDUIT after every turn.
- A state only changes when the model actually announces it — never on tiny fragments.
- Extra trigger phrases can be added live from config.json (they only ADD coverage).
- Detecting allergies/folk terms from the PATIENT'S words happens in state_manager.py,
  not here. This file only reads the MODEL'S output.
"""
import re
from state_manager import SessionState

# ── Agent State Definitions ───────────────────────────────────────────────────

AGENT_STATES: dict[str, str] = {
    "CONDUIT": (
        "Speak in first person. Transmit exactly. Be invisible. "
        "No additions, no editorializing."
    ),
    "CLARIFIER": (
        "Interrupt. Say: 'Interpreter speaking: I need to clarify…' "
        "Manage the flow until the ambiguity is resolved, then return to Conduit."
    ),
    "CULTURAL_BROKER": (
        "Interrupt. Say: 'Interpreter note: [term] means…' "
        "Explain the folk or cultural concept objectively and return to Conduit."
    ),
    "ADVOCATE": (
        "Emergency mode. Firm, authoritative voice. "
        "Say: 'Interpreter pause — there may be a critical misunderstanding…' "
        "Prevent medical harm before the session continues."
    ),
    "RESEARCHER": (
        "Verification mode. Say: 'Interpreter speaking: let me verify that term…' "
        "Query the knowledge base for the uncertain term, then interpret with the "
        "confirmed meaning and return to Conduit."
    ),
}

ROLE_LABELS = {
    "CONDUIT":        "CONDUIT",
    "CLARIFIER":      "CLARIFIER",
    "CULTURAL_BROKER":"CULTURAL BROKER",
    "ADVOCATE":       "ADVOCATE ⚠",
    "RESEARCHER":     "RESEARCHER 🔬",
}

# ── State Detection: What did the MODEL announce? ─────────────────────────────
# The system prompt trains the model to say these exact phrases.
# We detect COMPLETE phrases only — never partial streaming chunks.

_ADVOCATE_PATTERN = re.compile(
    r"interpreter\s+(pause|stop|halt)|"
    r"critical\s+misunderstanding|"
    r"debo\s+detener|"
    r"pausa\s+del\s+intérprete",
    re.I,
)

_CULTURAL_PATTERN = re.compile(
    r"interpreter\s+note|"
    r"nota\s+del\s+intérprete|"
    r"cultural\s+(clarification|note|brokering)|"
    r"aclaración\s+cultural|"
    r"folk\s+(belief|illness|concept)",
    re.I,
)

_CLARIFIER_PATTERN = re.compile(
    r"(this\s+is\s+)?the\s+interpreter\s+(speaking|needs|requests|asks)|"
    r"habla\s+el\s+intérprete|"
    r"interpreter\s+speaking|"
    r"as\s+the\s+interpreter[,\s]|"
    r"como\s+intérprete[,\s]",
    re.I,
)

# RESEARCHER — model wants to verify a term in the knowledge base.
# Must be checked BEFORE CLARIFIER (its announcement also contains
# "interpreter speaking"), so the verify-intent wins.
_RESEARCHER_PATTERN = re.compile(
    r"let\s+me\s+(verify|look\s+up|check|consult)|"
    r"verify\s+(that|the)\s+term|"
    r"verificar\s+(ese|el)\s+término|"
    r"consultar\s+(mi|la)\s+base|"
    r"knowledge\s+base|"
    r"base\s+de\s+conocimiento|"
    r"déjame\s+(verificar|consultar|confirmar)",
    re.I,
)

# Extract the uncertain term from the RESEARCHER announcement.
# Prefers a quoted term: ...verify the term 'susto'... → susto
_TERM_QUOTED = re.compile(r"['\"“”‘’]([^'\"“”‘’]{2,40})['\"“”‘’]")
_TERM_AFTER  = re.compile(
    r"(?:term|word|término|palabra|expression|expresión)\s+(?:['\"]?)([a-záéíóúñü\- ]{2,40})",
    re.I,
)


def extract_research_term(text: str) -> str:
    """Pull the term the model wants to research from its announcement."""
    if not text:
        return ""
    m = _TERM_QUOTED.search(text)
    if m:
        return m.group(1).strip()
    m = _TERM_AFTER.search(text)
    if m:
        return m.group(1).strip().strip(".,;:!?")
    return ""

# ── Rollback Detection: Model returning to CONDUIT ────────────────────────────
_ROLLBACK_PATTERN = re.compile(
    r"resuming\s+interpretation|"
    r"retomando\s+la\s+interpretación|"
    r"retomando\s+interpretación|"
    r"volvemos\s+a\s+la\s+interpretación",
    re.I,
)

# ── Minimum length to trust a routing decision ────────────────────────────────
# Streaming chunks shorter than this are IGNORED — we wait for more text.
# This prevents false triggers on 1-2 word fragments like "Hello" or "I have".
_MIN_CHARS_TO_ROUTE = 30

# ── Config-driven EXTRA triggers (additive, never destructive) ────────────────
# These phrases come from config.json and are matched as case-insensitive
# substrings ON TOP of the built-in regex above. They can only IMPROVE
# detection — if config is empty or broken, the regex still works.
_EXTRA_TRIGGERS: dict[str, list[str]] = {
    "CLARIFIER": [], "CULTURAL_BROKER": [], "RESEARCHER": [], "ADVOCATE": [],
}


def set_extra_triggers(triggers: dict) -> None:
    """Load extra trigger phrases from config (called once at startup)."""
    mapping = {
        "clarifier": "CLARIFIER",
        "cultural_broker": "CULTURAL_BROKER",
        "researcher": "RESEARCHER",
        "advocate": "ADVOCATE",
    }
    for cfg_key, role in mapping.items():
        phrases = (triggers or {}).get(cfg_key, [])
        if isinstance(phrases, list):
            _EXTRA_TRIGGERS[role] = [p.lower() for p in phrases if isinstance(p, str) and p.strip()]


def _extra_match(role: str, text_lower: str) -> bool:
    """True if any config phrase for this role appears in the text."""
    return any(p in text_lower for p in _EXTRA_TRIGGERS.get(role, []))


# ── Public API ─────────────────────────────────────────────────────────────────

def route(text: str, session: SessionState) -> tuple[str, str]:
    """
    Route based on what the MODEL announces in its output transcription.

    PRIORITY ORDER:
    1. Text too short → keep current state (don't change on fragments)
    2. Model announces rollback → CONDUIT
    3. Model announces ADVOCATE → ADVOCATE (safety emergency)
    4. Model announces CULTURAL_BROKER → CULTURAL_BROKER
    5. Model announces CLARIFIER → CLARIFIER
    6. Allergy conflict detected in session → ADVOCATE
    7. Default → CONDUIT

    NOTE: Folk/cultural term detection runs on INPUT transcription in state_manager,
    NOT here. The CULTURAL_BROKER state only fires when the model ANNOUNCES it.
    """

    stripped = text.strip()

    # 1. Fragment too short — do not reroute, keep current state
    if len(stripped) < _MIN_CHARS_TO_ROUTE:
        current = session.current_locked_state or "CONDUIT"
        return current, AGENT_STATES[current]

    # 2. Explicit rollback announcement → CONDUIT
    if _ROLLBACK_PATTERN.search(stripped):
        session.unlock_clarifier()
        return "CONDUIT", AGENT_STATES["CONDUIT"]

    low = stripped.lower()

    # 3. ADVOCATE — highest priority intervention
    if _ADVOCATE_PATTERN.search(stripped) or _extra_match("ADVOCATE", low):
        session.lock_state("ADVOCATE")
        return "ADVOCATE", AGENT_STATES["ADVOCATE"]

    # 3.5 RESEARCHER — verify-intent. Checked before CULTURAL/CLARIFIER because
    # its announcement also says "interpreter speaking".
    if _RESEARCHER_PATTERN.search(stripped) or _extra_match("RESEARCHER", low):
        session.lock_state("RESEARCHER")
        return "RESEARCHER", AGENT_STATES["RESEARCHER"]

    # 4. CULTURAL BROKER
    if _CULTURAL_PATTERN.search(stripped) or _extra_match("CULTURAL_BROKER", low):
        session.lock_state("CULTURAL_BROKER")
        return "CULTURAL_BROKER", AGENT_STATES["CULTURAL_BROKER"]

    # 5. CLARIFIER
    if _CLARIFIER_PATTERN.search(stripped) or _extra_match("CLARIFIER", low):
        session.lock_clarifier(stripped)
        return "CLARIFIER", AGENT_STATES["CLARIFIER"]

    # 6. Allergy conflict — model is saying a substance the patient is allergic to
    if session.allergies:
        text_lower = stripped.lower()
        for allergy in session.allergies:
            if allergy.lower() in text_lower:
                return "ADVOCATE", AGENT_STATES["ADVOCATE"]

    # 7. Default — transparent conduit
    session.unlock_clarifier()
    return "CONDUIT", AGENT_STATES["CONDUIT"]


def force_conduit(session: SessionState) -> tuple[str, str]:
    """
    Hard reset to CONDUIT. Call on every turn_complete.
    Guarantees the agent always starts fresh in the next PTT cycle.
    """
    session.unlock_clarifier()
    return "CONDUIT", AGENT_STATES["CONDUIT"]
