"""
ZGA Messages — Single Source of Truth for User-Facing Strings
=============================================================
Every string the user (or judge) can see or hear lives HERE — never inline in
core logic. main.py and the recovery directives import from this module, so the
system is parametric: changing the wording (or adding a locale) touches one file.

CONVENTIONS:
- Spoken recovery lines are intentionally BILINGUAL (EN then ES): that is product
  behavior — the interpreter must never strand either party. They are data, not
  hardcoded logic.
- UI/transcript labels are language-agnostic keys rendered for an English-first
  judge audience.
- Internal logs are NOT here: logs are observability, written in English in code.
"""

# ── Transcript speaker labels (prefixes shown in the conversation feed) ───────
SPEAKER_LABELS = {
    "provider": "(provider · EN)",
    "patient": "(patient · ES)",
    "interpreter": "(interpreter)",
}


def label_for(role_key: str) -> str:
    """Safe accessor — unknown keys degrade to the interpreter label."""
    return SPEAKER_LABELS.get(role_key, SPEAKER_LABELS["interpreter"])


# ── Onboarding (text-only intro chat message, once per connection) ────────────
INTRO_TEXT = (
    "Hello — I am Zero Gravity Agent, your medical interpreter for this session. "
    "Everything said here is confidential (HIPAA). Speak naturally in short phrases, "
    "take turns, and pause when you finish so I can interpret. "
    "Spanish ⇄ English, first person, with clinical safety monitoring active."
)

# ── Bilingual spoken recovery lines (delivered in the agent's OWN live voice) ──
SHORT_PHRASE_SENTENCE = (
    "Please speak in short phrases so I can interpret accurately. "
    "Por favor, hable en frases más cortas para poder interpretar con precisión."
)

NOISE_SENTENCE = (
    "This is the interpreter, I cannot hear you because of background noise. "
    "Habla el intérprete: no puedo escucharle por el ruido de fondo."
)

CLARIFY_LINE_ES = (
    "Habla el intérprete: necesito aclarar algo antes de continuar. "
    "¿Podría repetir o explicar lo que dijo, por favor?"
)

CLARIFY_LINE_EN = (
    "Doctor, the interpreter needs to clarify something with the patient "
    "before continuing."
)

# ── Injection directives (system text pushed into the live session) ───────────

def short_phrase_directive() -> tuple[str, str]:
    """(inject_instruction, ui_note). Bilingual ask — short phrases, both parties informed."""
    inject = (
        "[SYSTEM RECOVERY] The previous utterance was too long to interpret reliably. "
        f"Say EXACTLY this, first the English then the Spanish, and nothing else: \"{SHORT_PHRASE_SENTENCE}\""
    )
    return inject, SHORT_PHRASE_SENTENCE


def noise_directive() -> tuple[str, str]:
    """(inject_instruction, ui_note). Bilingual notice that background noise blocked interpretation."""
    inject = (
        "[SYSTEM RECOVERY] The last input was contaminated by background noise or a second "
        "speaker and cannot be interpreted safely. Do NOT attempt to translate it. "
        f"Say EXACTLY this, first the English then the Spanish, and nothing else: \"{NOISE_SENTENCE}\""
    )
    return inject, NOISE_SENTENCE


def bilingual_clarify_directive(missing_lang: str) -> str:
    """
    Forces the model to repeat its clarification request in the language it skipped,
    in the SAME live voice. missing_lang is "Spanish" or "English".
    """
    line = CLARIFY_LINE_ES if missing_lang == "Spanish" else CLARIFY_LINE_EN
    return (
        "[SYSTEM RECOVERY] Your previous clarification request was delivered in only one "
        f"language, leaving one party lost. Now say EXACTLY this, in {missing_lang}, and "
        f"nothing else: \"{line}\""
    )


def retry_directive(transcribed_text: str) -> str:
    """Silently re-ask the model to interpret the buffered text after a muted turn."""
    return (
        "[SYSTEM] The previous interpretation did not complete. Interpret the following "
        "now, in first person, into the other language, with no commentary and no system "
        "words: " + transcribed_text
    )


def knowledge_inject(term: str, knowledge: str) -> str:
    """Feed a confirmed RESEARCHER result back into the live session."""
    return (
        f"[KNOWLEDGE BASE RESULT for '{term}']: {knowledge}\n\n"
        "Now deliver the accurate interpretation using this confirmed meaning. "
        "Do NOT mention the lookup or the knowledge base again."
    )


def knowledge_miss_inject(term: str) -> str:
    """Tell the live session the lookup found nothing — interpret unaided, no stalling."""
    return (
        f"[KNOWLEDGE BASE: no confirmed entry for '{term}'.] "
        "Use your best professional judgment and interpret the message normally now. "
        "Do NOT mention the knowledge base."
    )


# ── Shield mode (unauthorized public visitors — text-only simulation) ─────────
SHIELD_STATUS = "DEMO SHIELD ACTIVE"
SHIELD_USE_SCENARIO_PANEL = (
    "Protected demo mode: recording is locked. Please use the scenario panel."
)
SHIELD_MIC_LOCKED = (
    "Protected demo mode: live voice recording is disabled for public visitors. "
    "Please use the pre-recorded scenario panel on the side."
)

# ── Generic client-facing errors / status ─────────────────────────────────────
SESSION_FAILED = "Session failed: {error}"
MODEL_BUSY_NOTICE = "Interpretation in progress — please wait for the current turn to finish."
