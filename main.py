"""
Zero Gravity Agent — Main Server (FastAPI + WebSocket)
======================================================
HIPAA-compliant, real-time Spanish<->English medical interpreter.

READER'S GUIDE (plain English — what this file does):
  This is the heart of the app. It runs a web server that the browser talks to.
  1. The browser opens a WebSocket and streams the patient's/doctor's voice as audio.
  2. We forward that audio to Gemini 3.1 Flash Live (Google's real-time voice model),
     which speaks back the interpretation almost instantly.
  3. While interpreting, the model can ANNOUNCE a special "state" (e.g. it needs to
     clarify, or it spotted a safety risk). `semaphore.py` reads that announcement and
     tells the UI which colored badge to light up.
  4. If the model is unsure of a rare term (the RESEARCHER state), we hand the term to
     the ADK multi-agent reasoning team (`adk_agents.py`), which looks it up on the web
     via MCP and in the Vertex AI knowledge base, then feeds the answer back so the model
     can interpret correctly.
  5. Throughout, `state_manager.py` keeps a small clinical memory (allergies, symptoms...)
     so the agent stays grounded and never invents facts.

  Where to look:
    - SYSTEM_PROMPT below ........ the rules that shape how the interpreter behaves.
    - ws_endpoint() .............. the live session: receives audio, runs the two loops.
    - receiver() ................. reads Gemini's output, routes states, triggers RESEARCHER.
    - sender() ................... forwards the user's microphone audio to Gemini.

Powered by: Gemini 3.1 Flash Live (audio) + ADK/Vertex (reasoning) + Cloud Run.
SingularityOS AI — Gabriel Bustos.
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel
from google import genai
from google.genai import types
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from state_manager import SessionState
from semaphore import route as semaphore_route, force_conduit, ROLE_LABELS, extract_research_term, set_extra_triggers
import rag_engine
import adk_agents
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] ZERO-GRAVITY: %(message)s",
)
logger = logging.getLogger("ZeroGravity")

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).parent

STATIC_DIR = BASE_DIR / "static"
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

# Cascade: LLC key first, personal key as fallback
_GEMINI_KEYS: list[str] = [
    k for k in [
        os.environ.get("GEMINI_API_KEY_LLC"),
        os.environ.get("GEMINI_API_KEY"),
    ] if k
]
if not _GEMINI_KEYS:
    raise RuntimeError(f"Set GEMINI_API_KEY_LLC or GEMINI_API_KEY in .env at {env_path}")

import re

# ── Language stopword lexicons (module-level, shared) ─────────────────────────
# Used both to detect the spoken language (detect_is_english) and to judge whether
# a transcription is coherent human speech vs. background-noise/cross-talk garble
# (looks_incoherent). Kept at module scope so both functions share one source.
_ES_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "y", "en", "que", "qué", "me", "mi", "mí", "tengo",
    "dolor", "esta", "está", "es", "con", "para", "por", "si", "no", "como", "cómo", "al",
    "del", "lo", "se", "te", "le", "su", "sus", "doctor", "médico", "paciente", "siento",
    "pastilla", "medicina", "alergia", "susto", "empacho", "mal", "de", "a", "más", "muy",
    "pero", "porque", "cuando", "desde", "hace", "ya", "está", "estoy", "duele",
    # clarification-context words (help tell a real Spanish sentence from a quoted term)
    "necesito", "usted", "dijo", "quiere", "decir", "puede", "podría", "repetir", "repítalo",
    "habla", "intérprete", "interprete", "aclarar", "favor", "algo", "otra", "forma", "explicar",
    "entiendo", "escuchar", "escucharle", "ruido", "fondo",
}
_EN_STOPWORDS = {
    "the", "a", "an", "and", "in", "of", "to", "for", "with", "on", "is", "are", "am", "was",
    "were", "my", "your", "his", "her", "he", "she", "it", "they", "we", "you", "i", "have",
    "has", "had", "do", "does", "did", "pain", "doctor", "patient", "allergic", "allergy",
    "this", "that", "but", "because", "when", "since", "very", "much", "feel", "feeling",
    # clarification-context words
    "need", "clarify", "said", "mean", "please", "repeat", "could", "would", "again",
    "interpreter", "understand", "hear", "say", "what",
}
# Combined function-word set — if a long transcription contains almost none of these,
# it is very likely noise/garble rather than a real sentence.
_ALL_FUNCTION_WORDS = _ES_STOPWORDS | _EN_STOPWORDS
_WORD_RE = re.compile(r"\b[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]+\b")


def detect_is_english(text: str) -> bool:
    """
    Returns True if the text is English, False if it is Spanish.
    Uses a fast and extremely reliable stopword-based density check.
    """
    if not text:
        return True

    words = _WORD_RE.findall(text.lower())
    if not words:
        return True

    es_count = sum(1 for w in words if w in _ES_STOPWORDS)
    en_count = sum(1 for w in words if w in _EN_STOPWORDS)

    if es_count == en_count:
        # Default to English unless Spanish stopwords exist and English don't
        return es_count == 0
    return en_count > es_count


def lang_presence(text: str) -> tuple[bool, bool]:
    """
    (has_spanish, has_english) — True if that language is meaningfully present in the text.
    Used to verify an intervention was delivered bilingually; if a language is missing we
    play a deterministic backstop clip so neither party is ever stranded.
    """
    if not text:
        return (False, False)
    words = _WORD_RE.findall(text.lower())
    es = sum(1 for w in words if w in _ES_STOPWORDS)
    en = sum(1 for w in words if w in _EN_STOPWORDS)
    # Threshold 3: a real sentence in a language has several function words, while an
    # incidental QUOTED foreign term (e.g. "mal de san vito" inside an English request)
    # only contributes 1-2 — so a quoted term cannot fake the presence of a language.
    return (es >= 3, en >= 3)


def looks_incoherent(text: str, min_words: int = 6, max_stopword_ratio: float = 0.08) -> bool:
    """
    Deterministic 'is this garbled / contaminated by background noise?' check.

    PLAIN ENGLISH: When a second speaker, crosstalk, or loud background noise bleeds
    into the microphone, the speech-to-text returns a long string of fragments with
    almost no real function words ("the", "y", "de", "is"...). Real sentences — even
    short ones — are dense with these. So: only flag LONG utterances whose function-word
    density is implausibly low, or that are dominated by one repeated token. We err on
    the side of 'coherent' (return False) so we never wrongly interrupt a real sentence —
    minor background noise is simply ignored, exactly as a human interpreter would.

    Returns True ONLY when contamination is bad enough to endanger the interpretation.
    """
    if not text:
        return False
    words = _WORD_RE.findall(text.lower())
    n = len(words)
    if n < max(3, min_words):
        # Too short to judge — ignore minor noise, treat as coherent.
        return False
    fn = sum(1 for w in words if w in _ALL_FUNCTION_WORDS)
    if (fn / n) < max_stopword_ratio:
        return True
    # Pathological single-token repetition (e.g. "no no no no ...", a stuck mic).
    most = max((words.count(w) for w in set(words)), default=0)
    if n >= 8 and (most / n) > 0.6:
        return True
    return False


# ── Recovery directives (spoken by the agent IN ITS OWN LIVE VOICE) ───────────
# No pre-recorded clips anywhere: every recovery line is injected into the live
# session so the SAME voice (Charon/Aoede) the parties have been hearing delivers
# it. ALL recovery lines are BILINGUAL (EN then ES) so neither party is ever lost.
SHORT_PHRASE_SENTENCE = (
    "Please speak in short phrases so I can interpret accurately. "
    "Por favor, hable en frases más cortas para poder interpretar con precisión."
)


def short_phrase_directive() -> tuple[str, str]:
    """(inject_instruction, ui_note). Bilingual ask — short phrases, both parties informed."""
    inject = (
        "[SYSTEM RECOVERY] The previous utterance was too long to interpret reliably. "
        f"Say EXACTLY this, first the English then the Spanish, and nothing else: \"{SHORT_PHRASE_SENTENCE}\""
    )
    return inject, SHORT_PHRASE_SENTENCE


def noise_directive() -> tuple[str, str]:
    """(inject_instruction, ui_note). Bilingual notice that background noise blocked interpretation."""
    sentence = (
        "This is the interpreter, I cannot hear you because of background noise. "
        "Habla el intérprete: no puedo escucharle por el ruido de fondo."
    )
    inject = (
        "[SYSTEM RECOVERY] The last input was contaminated by background noise or a second "
        "speaker and cannot be interpreted safely. Do NOT attempt to translate it. "
        f"Say EXACTLY this, first the English then the Spanish, and nothing else: \"{sentence}\""
    )
    return inject, sentence


def bilingual_clarify_directive(missing_lang: str) -> str:
    """
    Forces the model to repeat its clarification request in the language it skipped,
    in the SAME live voice — the deterministic backstop that used to be a pre-recorded
    clip. missing_lang is "Spanish" or "English".
    """
    if missing_lang == "Spanish":
        line = ("Habla el intérprete: necesito aclarar algo antes de continuar. "
                "¿Podría repetir o explicar lo que dijo, por favor?")
    else:
        line = ("Doctor, the interpreter needs to clarify something with the patient "
                "before continuing.")
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

MODEL_ID = "gemini-3.1-flash-live-preview"

SYSTEM_PROMPT = """You are Zero Gravity Agent — a HIPAA-compliant professional medical consecutive interpreter operating under NCIHC, IMIA, and Joint Commission standards for healthcare interpreting.

YOUR ROLE: You are the interpreter — not the physician, not the patient, not a medical advisor. You convey messages. You do not diagnose, recommend, or advise.

YOUR MODE: Consecutive interpretation. Wait for a complete utterance (1-3 sentences), then render it fully and accurately in the target language.

YOUR VOICE: Always interpret in FIRST PERSON.
  - Patient says: "Me duele la cabeza" → You say: "I have a headache."
  - Provider says: "Take this twice a day" → You say: "Tómelo dos veces al día."
  - NEVER say: "The patient says he has a headache" or "El médico dice que..."

TURN HANDLING: Each audio turn ends when the speaker pauses. When a turn ends, deliver the complete consecutive interpretation of what was just said. Do NOT vocalize, repeat, or include any system or trigger words (such as "INTERPRET NOW") — those are internal signals, never spoken. Output ONLY the interpretation or a declared intervention.

ACCURACY STANDARD (NCIHC/IMIA):
- Render ALL content accurately and completely.
- Do NOT add, omit, distort, summarize, or editorialize.
- Convey both the CONTENT and the SPIRIT of the original message.
- Preserve the register (formal/informal) of the speaker.
- Preserve emotional tone: if a patient is panicked, the interpretation must convey that urgency.
- NEVER omit clinically relevant content — including repetitions, hesitations, or emotional expressions.

PROHIBITED ACTIONS:
- Never explain, advise, or add information not said by the speaker.
- Never soften bad news or add reassurances not stated by the provider.
- Never upgrade or downgrade the severity of a patient's complaint.
- Never skip profanity or strong language — interpret it (note "strong language used" if needed).

⚡ TWO-PHASE INTERVENTION WITH ELITE TRANSPARENCY PROTOCOL — THIS IS THE MOST IMPORTANT RULE.
A robot translates first and explains later. You are NOT a robot. You intervene FIRST with transparent declaration, then translate LATER.

When an utterance needs an intervention (clarification of ambiguity, folk illness, safety hazard, unknown medical term, or incoherent speech):
  PHASE 1 — DECLARED INTERVENTION ONLY (No translation in this turn):
    • You MUST FIRST address the Physician (the English speaker) in English to request permission to clarify or verify:
      - Script: "Mr. Provider, the interpreter requests permission to clarify the term '[term]' with the patient."
    • Immediately after that English declaration, in the SAME turn, you address the patient in Spanish to ask the clarifying/cultural question:
      - Script: "Habla el intérprete: necesito aclarar — usted dijo '[term]'. ¿Qué quiere decir con eso?" (or corresponding phrase below).
    • STOP and wait. Do not produce any translation yet.

  PHASE 2 — INFORMED TRANSLATION (next turn, after they answer):
    • Combine the ORIGINAL utterance + their CLARIFICATION.
    • Deliver the single, correct English interpretation to the Physician.
    • Then you are done — control returns to active listening (Conduit).

NEVER collapse both phases into one turn. NEVER translate the ambiguous content "just in case" before clarifying.
If you are unsure whether to intervene — INTERVENE. Timidity causes medical errors.

🌐🌐 ABSOLUTELY CRITICAL — BILINGUAL INTERVENTION RULE (rule #0, overrides everything):
THE PATIENT DOES NOT SPEAK ENGLISH. If you intervene in English ONLY, the patient is left completely
lost and the encounter fails. Therefore, EVERY time you intervene — CLARIFIER, CULTURAL_BROKER,
ADVOCATE, or RESEARCHER — you MUST say it in BOTH languages, in the SAME turn, NO exceptions:
  1) FIRST in ENGLISH (addressing the physician), THEN
  2) in SPANISH (addressing the patient).
You are FORBIDDEN from delivering any intervention, clarification request, note, or pause in only one
language. This applies regardless of who spoke last or which language triggered the intervention.
Concrete template you MUST follow for a clarification:
  EN: "Doctor, the interpreter needs to clarify something with the patient."
  ES: "Habla el intérprete: necesito aclarar algo. ¿Podría repetir o explicar lo que dijo, por favor?"
If you ever catch yourself about to end an intervention turn having spoken only English, STOP and add
the Spanish. A monolingual intervention is a critical failure.

INTERVENTION PROTOCOL — you MUST announce the EXACT opening phrase before EVERY intervention. These phrases are how the system detects your state — say them verbatim:

0. INCOHERENT / NONSENSICAL INPUT — HYBRID TWO-STRIKE RULE (clinically critical):
   WHY: In medicine, incoherent speech ("word salad") may signal stroke, aphasia, intoxication, or psychosis. It is itself a CLINICAL SIGN — the clinician must ultimately hear it.
   STRIKE 1 — first time an utterance is incoherent (e.g. "my unicorn wouldn't turn on because the bus was in reverse"): assume you may have misheard. First state in English to the doctor:
     "Mr. Provider, the interpreter requests permission to clarify with the patient."
     Then in Spanish to the patient (this is CLARIFIER):
     "Habla el intérprete: necesito aclarar — lo que dijo no me quedó claro. ¿Podría decirlo de otra manera?" Then STOP. Do not translate yet.
   STRIKE 2 — if the REPEAT is STILL incoherent: do NOT keep asking. Now CONVEY it so the clinician hears the sign (this is ADVOCATE-level):
     Translate the words literally to the provider AND add: "Interpreter pause — the patient's speech is persistently incoherent and disorganized; this may itself be clinically significant."

1. CLARIFICATION NEEDED:
   Trigger: Ambiguous term, regional slang, or unknown expression.
   Phase 1 script: "Mr. Provider, the interpreter requests permission to clarify the term '[term]' with the patient. Habla el intérprete: necesito aclarar — usted dijo '[term]'. ¿Qué quiere decir con eso?" Then STOP and wait for their answer.

2. CULTURAL BROKERING:
   Trigger: Folk illness term or cultural belief affecting clinical understanding (susto, empacho, mal de ojo, daño, sereno, pasmo).
   Phase 1 script: "Mr. Provider, the interpreter requests permission to clarify the cultural term '[term]' with the patient. Interpreter note: '[term]' is a cultural health concept — let me confirm what you are experiencing. Habla el intérprete: ¿Me puede describir qué experimenta con el '[term]'?" Then STOP.

3. SAFETY ADVOCACY — highest level:
   Trigger: Potential for clinical harm from misunderstanding (e.g. provider about to prescribe something the patient is allergic to), or any expression of patient self-harm or suicidal ideation (e.g., "mejor no estar aquí", "pills stored at home", "end my life", "pastillas guardadas").
   Phase 1 script: "Interpreter pause — there may be a critical misunderstanding / safety concern. [State the safety or self-harm concern clearly to both parties, e.g., 'The patient has expressed thoughts of self-harm.']" Then STOP and let them resolve it.

4. RESEARCHER — VERIFY A TERM IN THE KNOWLEDGE BASE:
   Trigger: You encounter a medical term, regional idiom, or rare-disease/ethics concept you are NOT confident about.
   - NOTE: "mal de San Vito" (Saint Vitus' Dance / Sydenham's chorea) is a rare medical illness that you MUST verify in the knowledge base using the RESEARCHER state. Never treat "mal de San Vito" as a cultural folk illness (CULTURAL_BROKER) — it is a serious neurological condition requiring medical RESEARCHER verification.
   WHY: Guessing in medicine is dangerous. You have a knowledge base — use it instead of hallucinating.
   Phase 1 script: "Mr. Provider, the interpreter requests permission to verify the term '[term]' in the knowledge base." Then STOP. The system will retrieve the definition and feed it back to you; in the next turn, interpret using the confirmed meaning. When you receive a line beginning with "[KNOWLEDGE BASE", trust it and interpret accordingly — do NOT announce the lookup again.

5. NUMBER/DOSAGE VERIFICATION:
   Trigger: Any number involving medication dose, frequency, or critical vital sign where uncertainty exists.
   Phase 1 script: "Interpreter: to confirm — did you say [X] milligrams / [X] times a day / [X] degrees?" Then STOP.

CRITICAL NUMBERS PROTOCOL:
- ALL medication dosages: interpret EXACTLY. Never approximate. 25 mg ≠ 20 mg.
- ALL frequencies: interpret as stated. "Three times a day" ≠ "every 8 hours."
- ALL vital signs: interpret exactly. Verify if uncertain.
- Temperature conversion — Latin American patient in US context: state both scales.
  "38 grados" → "38 degrees Celsius, that is 100.4 Fahrenheit"
  "39 grados" → "39 degrees Celsius, 102.2 Fahrenheit"
  "40 grados" → "40 degrees Celsius, 104 Fahrenheit — high fever"
- Format numbers as spoken, not as digits ("twenty-five milligrams", not "25 mg").

REGIONAL SPANISH MEDICAL AWARENESS:
Spanish spoken by patients from Venezuela, Colombia, Mexico, Puerto Rico, Cuba, and Central America contains significant regional variation in medical vocabulary.

ANATOMICAL GOLDEN RULE: "El estómago" in Latin American vernacular may refer to any abdominal organ or area. Always interpret as "abdominal area/pain" and add an interpreter note if anatomical precision matters clinically.

KEY REGIONAL MAPPINGS — apply automatically:
- "calentura" → fever (NOT "warmth")
- "chorro" (Central America/Mexico) → diarrhea
- "el suero" / "me están dando el suero" (Venezuela/Caribbean) → IV fluids / I'm getting IV fluids
- "me dieron puntos" → I was sutured / I got stitches
- "me dio un aire" → sudden chill / possible neurological event — ALWAYS CLARIFY which
- "la regla" → menstruation
- "el flujo" → vaginal discharge
- "presión alta/baja" → high/low blood pressure
- "punzada" → stabbing pain
- "ardor" → burning sensation
- "frío de huesos" (Venezuela/Colombia) → chills / rigors
- "se me fue la vista" → I had sudden vision loss / my vision went dark

FOLK ILLNESS TERMS — always deliver interpretation AND add interpreter note:
- "empacho" → GI distress / indigestion [Interpreter note: folk belief about stuck food causing GI upset]
- "susto" → anxiety / fright response / somatization [Interpreter note: cultural syndrome from traumatic fright]
- "mal de ojo" → [Interpreter note: folk belief — evil eye; clinically: fever, fussiness, irritability in child]
- "daño" / "mal puesto" → [Interpreter note: patient believes illness has a spiritual/hex cause — may affect treatment adherence]
- "sereno" → cold night air exposure [Interpreter note: upper respiratory symptoms attributed to nighttime air]
- "pasmo" → muscle spasm / cramp [Interpreter note: attributed to hot-cold imbalance in folk medicine]

ABSOLUTE PROHIBITIONS:
1. NEVER provide medical advice, diagnosis, or treatment recommendations.
2. NEVER tell the patient "everything will be fine" or add reassurances not said by the provider.
3. NEVER ask a follow-up clinical question on your own initiative — only to clarify a term for accuracy.
4. NEVER answer a clinical question directed at you as if you were the clinician.
5. NEVER omit — not even filler words when they reflect the patient's communication style or distress.
6. NEVER upgrade the register of a patient who speaks informally or colloquially.
7. NEVER substitute a folk illness term with its clinical equivalent WITHOUT announcing an interpreter note.
8. NEVER assume a Latin American patient in the US uses Fahrenheit — default to Celsius unless specified.

HIPAA COMPLIANCE:
- All content is Protected Health Information (PHI).
- No storage, repetition, or disclosure beyond interpretation duties.
- Treat every utterance as strictly confidential.
- If asked by either party to keep a secret: "The interpreter's role requires interpreting everything said to all parties. I cannot withhold information."

TONE: Professional medical interpreter. Neutral. Clinical. Precise. First person. No opinions or additions.

ABSOLUTE OUTPUT RULE — ZERO DISCLAIMERS (CRITICAL):
You are a licensed professional interpreter operating in a controlled clinical setting WITH a physician present.
You NEVER append disclaimers, safety notices, legal warnings, or editorial statements of ANY kind.
The following phrases — and ANY paraphrase of them — are STRICTLY, ABSOLUTELY FORBIDDEN in your output:
  ✗ "Please be advised that I am not providing medical advice"
  ✗ "I am not providing a diagnosis"
  ✗ "you should see a healthcare professional" / "seek care" / "consult a doctor"
  ✗ "This service is provided for informational purposes only"
  ✗ "I'm an AI" / "as an AI" / "I cannot provide medical advice"
Interpreting a serious diagnosis is your JOB — the doctor is the one giving it, you only convey it faithfully.
Adding a disclaimer is a CRITICAL PROFESSIONAL FAILURE that breaks the patient's trust. The physician handles all advice.
Output ONLY: the faithful interpretation, OR a declared intervention. Nothing else. Ever.

SESSION MEMORY — STAY GROUNDED, DO NOT HALLUCINATE:
A running record of the session (allergies, symptoms, medications, clarified terms) is maintained for you.
- Rely on what was ACTUALLY said. Never invent symptoms, doses, or history that the speakers did not state.
- If you already clarified a term earlier in this session, reuse that meaning — do not re-ask.
- If a new statement contradicts something said earlier (e.g. a dose changed), flag it via the verification protocol."""


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Zero Gravity Agent — Medical Interpreter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")



@app.get("/")
async def root():
    # No-cache headers: during active development the UI changes constantly.
    # Without this, browsers serve a stale index.html and new features (VAD button,
    # semaphore telemetry) silently don't appear until a manual hard-refresh.
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/config")
async def get_config():
    # Return stripped Google Client ID to prevent \r carriage return injection bug
    return {
        "google_client_id": os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    }


class VerifyRequest(BaseModel):
    token: str | None = None
    judge_token: str | None = None


@app.post("/api/verify-token")
async def verify_token_endpoint(req: VerifyRequest):
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    env_judge_token  = os.environ.get("JUDGE_SECRET_TOKEN", "").strip()

    is_valid = False
    auth_type = None
    email_addr = None

    # 1) Check Judge Token
    if req.judge_token:
        jt = req.judge_token.strip()
        if env_judge_token and jt == env_judge_token:
            is_valid = True
            auth_type = "judge"
            logger.info("==================================================")
            logger.info("🔑 SECURE LOGIN VERIFICATION | Method: JUDGE TOKEN")
            logger.info("==================================================")

    # 2) Check SSO Token
    elif google_client_id and req.token:
        idinfo = verify_google_token(req.token, google_client_id)
        if idinfo:
            email = idinfo.get("email", "").lower().strip()
            if email in ALLOWED_EMAILS:
                is_valid = True
                auth_type = "sso"
                email_addr = email
                logger.info("==================================================")
                logger.info(f"🔑 SECURE LOGIN VERIFICATION | Method: SSO ({email})")
                logger.info("==================================================")
            else:
                logger.warning(f"Rejected SSO token verification: email {email} not allowed.")
        else:
            logger.warning("Rejected SSO token verification: invalid or expired token.")

    return {
        "valid": is_valid,
        "type": auth_type,
        "email": email_addr
    }



@app.get("/api/rag-status")
async def get_rag_status():
    """UI badge: is the Vertex RAG knowledge base online for the RESEARCHER state?"""
    return rag_engine.status()


# ═══════════════════════════════════════════════════════════════════════════
# BLACK BOX CONFIG — FIXED, NOT TUNABLE.
# These values were battle-tested during the June 2026 intensive roleplay
# (39/41 PASS). They are constants on purpose: a judge or client cannot
# misconfigure the agent into a broken state. Edit the source to change them.
# ═══════════════════════════════════════════════════════════════════════════
FIXED_CONFIG = {
    "vad": {
        "silence_ms": 2000,        # interpretation pause: 2 seconds (CEO-approved)
        "onset_ms": 180,
        "abs_floor": 0.012,
        "noise_mult": 3.2,
        "min_utter_ms": 300,
        "max_utterance_ms": 30000,
    },
    "timeouts": {
        "interpret_timeout_s": 5,
        "max_reconnects": 5,
        "reconnect_backoff_cap_s": 16,
        "session_warn_seconds": 870,
    },
    "recovery": {"notify_on_mute": True, "long_input_warning": True},
    "interpretation": {
        "max_retries": 1,
        "incoherence_min_words": 6,
        "incoherence_max_stopword_ratio": 0.08,
    },
    # Built-in state triggers — hardcoded, no UI editing.
    "triggers": {
        "clarifier": [
            "interpreter speaking", "habla el intérprete", "como intérprete",
            "requests permission", "interpreter requests permission",
        ],
        "cultural_broker": ["interpreter note", "nota del intérprete", "cultural concept"],
        "researcher": [
            "let me verify", "let me look up", "knowledge base", "déjame verificar",
            "consultar mi base", "mal de san vito", "saint vitus",
        ],
        "advocate": [
            "interpreter pause", "pausa del intérprete", "critical misunderstanding",
            "pills stored", "self-harm", "suicid", "better if I weren't here", "no estar aquí",
        ],
    },
    # Hybrid: ask first if unsure; if clear, translate literally (CEO-approved).
    "nonsense_handling": "hybrid",
}


@app.get("/api/runtime-config")
async def get_runtime_config():
    """Read-only: the frontend VAD worklet reads its fixed timing from here."""
    return FIXED_CONFIG


# ── BLACK BOX: only two voices, Charon (male) is the fixed default ───────────
# Pre-recorded audio is GONE. One session = ONE voice for everything the agent
# says (interpretations, clarifications, recovery lines) — no second "mystery
# voice" ever breaks the magic.
AVAILABLE_VOICES = ["Charon", "Aoede"]
DEFAULT_VOICE = "Charon"
ALLOWED_EMAILS = {"gabriel@singularityos-ai.com", "gabobustos382@gmail.com"}

# ── Robot intro (TEXT ONLY — replaces the old intro_bilingual.pcm clip) ───────
# Sent once per connection as a chat message from "Zero Gravity Agent" so a page
# reset never forces anyone to sit through the same spoken speech again.
INTRO_TEXT = (
    "Hello — I am Zero Gravity Agent, your medical interpreter for this session. "
    "Everything said here is confidential (HIPAA). Speak naturally in short phrases, "
    "take turns, and pause when you finish so I can interpret. "
    "Spanish ⇄ English, first person, with clinical safety monitoring active."
)

# FAST START (local dev only): the Vertex RAG probe below makes a live network call
# that adds ~15-20s to boot. Set ZGA_FAST_START=1 to SKIP the eager probes — RAG and
# ADK then initialize lazily on the first RESEARCHER use instead. The /api/rag-status
# endpoint still probes live, so the UI badge stays accurate. Never set this in production.
_FAST_START = os.environ.get("ZGA_FAST_START", "").strip().lower() not in ("", "0", "false", "no")

if _FAST_START:
    logger.info("⚡ ZGA_FAST_START activo — omitiendo probes de RAG/ADK (carga perezosa al primer uso)")
    _rag_status = {"available": False, "error": "probe skipped (fast start)", "corpus": ""}
    _adk_status = {"available": False, "error": "probe skipped (fast start)", "model": ""}
else:
    # Probe the RAG knowledge base once at startup (instant if no service-account.json)
    _rag_status = rag_engine.status()
    if _rag_status["available"]:
        logger.info(f"🔬 RESEARCHER online — Vertex RAG corpus: {_rag_status['corpus']}")
    else:
        logger.warning(f"🔬 RESEARCHER offline — {_rag_status['error']} (agent interpreta normal)")

    # Probe the ADK multi-agent reasoning layer (lazy — first call builds it)
    _adk_status = adk_agents.status()
    if _adk_status["available"]:
        logger.info(f"🧠 ADK multi-agent online — model: {_adk_status['model']} (Vertex)")
    else:
        logger.warning(f"🧠 ADK reasoning offline — {_adk_status['error']} (RESEARCHER usa RAG directo)")

# Wire the FIXED triggers into the semaphore (black box — no dynamic reload)
set_extra_triggers(FIXED_CONFIG["triggers"])
logger.info(
    f"⚙️ BLACK BOX config — VAD silence={FIXED_CONFIG['vad']['silence_ms']}ms | "
    f"interpret_timeout={FIXED_CONFIG['timeouts']['interpret_timeout_s']}s | "
    f"nonsense={FIXED_CONFIG['nonsense_handling']} | voices={AVAILABLE_VOICES}"
)


def verify_google_token(token: str, client_id: str) -> dict | None:
    try:
        idinfo = id_token.verify_oauth2_token(token, google_requests.Request(), client_id)
        return idinfo
    except Exception as e:
        logger.error(f"SSO Token verification failed: {e}")
        return None


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    # ── ACCESS CONTROL: judge_token  OR  Google SSO (Gabriel)  OR  local dev ──
    # Judges receive a private link (…/?judge_token=SECRET) so they get the full LIVE
    # experience with NO login and NO email — that secret token IS their key. Anyone
    # without a valid token (and not the SSO owner) is rejected, so random public
    # traffic can never spend the Gemini Live / Vertex budget. No simulation mode.
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    env_judge_token  = os.environ.get("JUDGE_SECRET_TOKEN", "").strip()
    token       = websocket.query_params.get("token")
    judge_token = websocket.query_params.get("judge_token")

    is_authorized = False
    auth_label = ""

    # 1) Judge: a valid secret token → full live access, no login required.
    if env_judge_token and judge_token and judge_token.strip() == env_judge_token:
        is_authorized = True
        auth_label = "judge_token"
    # 2) Owner/team: Google SSO with an allow-listed email.
    elif google_client_id and token:
        idinfo = verify_google_token(token, google_client_id)
        if idinfo:
            email = idinfo.get("email", "").lower().strip()
            if email in ALLOWED_EMAILS:
                is_authorized = True
                auth_label = f"SSO {email}"
            else:
                logger.warning(f"Rejecting WebSocket: unauthorized email {email}")

    # 3) Local development (not on Cloud Run) → always allow for convenience.
    if not os.environ.get("K_SERVICE"):
        is_authorized = True
        auth_label = auth_label or "local-dev"
    # 4) No protection configured at all (bare local run) → unsecured mode.
    if not google_client_id and not env_judge_token:
        is_authorized = True
        auth_label = auth_label or "unsecured"

    if not is_authorized:
        logger.warning("Client connected under SHIELD Mode (Simulation / Cost-Free).")
        await websocket.accept()
        await websocket.send_json({
            "type": "status",
            "message": "DEMO SHIELD ACTIVE",
            "live_ai": False
        })
        
        active_scenario_id = None
        try:
            while True:
                data = await websocket.receive()
                # Discard incoming raw bytes (PCM stream from the client)
                if "bytes" in data:
                    continue
                
                text_msg = data.get("text")
                if text_msg:
                    msg_json = json.loads(text_msg)
                    msg_type = msg_json.get("type")
                    if msg_type == "ptt_start":
                        active_scenario_id = msg_json.get("scenario_id")
                        logger.info(f"Shield Mode: Scenario {active_scenario_id} started")
                    elif msg_type == "ptt_release":
                        if active_scenario_id:
                            # 1. Simulate clinical thinking latency
                            await asyncio.sleep(1.5)
                            
                            # 2. Extract matching scenario details from DEMO_SCENARIOS
                            scenario = DEMO_SCENARIOS.get(active_scenario_id)
                            if scenario:
                                role = scenario.get("expected_state", "CONDUIT")
                                trans = scenario.get("english_script", "")
                                patient = scenario.get("patient", "")
                                
                                # 3. Simulate high-fidelity system transitions
                                await websocket.send_json({
                                    "type": "state_preview",
                                    "active_role": role,
                                    "session_memory": {"symptoms": scenario.get("dna", "")}
                                })
                                
                                await websocket.send_json({
                                    "type": "log",
                                    "level": "info",
                                    "msg": f"🌿 [SHIELD MOCK] ADK Resolved state for {patient} -> {role}"
                                })
                                
                                # Update clinical scratchpad
                                await websocket.send_json({
                                    "type": "session_update",
                                    "session_memory": {
                                        "allergies": "N/A" if role != "ADVOCATE" else "Penicillin, Sulfa",
                                        "symptoms": scenario.get("dna", "")[:120],
                                        "cultural_context": "susto/empacho" if "folk" in active_scenario_id or "susto" in active_scenario_id or "empacho" in active_scenario_id else "N/A"
                                    }
                                })
                                
                                # 4. (Pre-recorded audio removed — Shield mode is text-only)

                                # 5. Flush translation transcript & complete turn
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": f"[DEMO SIMULATION] {trans}",
                                    "role": role
                                })
                                
                                await websocket.send_json({
                                    "type": "turn_complete",
                                    "active_role": "CONDUIT"
                                })
                            else:
                                await websocket.send_json({
                                    "type": "transcript",
                                    "text": "Modo Demostración Protegida: Grabación bloqueada. Utiliza el panel de escenarios.",
                                    "role": "CLARIFIER"
                                })
                        else:
                            await websocket.send_json({
                                "type": "transcript",
                                "text": "Modo Demostración Protegida: Grabación de voz en caliente bloqueada para visitas públicas. Por favor, utiliza el panel lateral de escenarios pre-grabados.",
                                "role": "CLARIFIER"
                            })
        except WebSocketDisconnect:
            logger.info("Simulation client disconnected.")
        except Exception as e:
            # e.g. RuntimeError "receive after disconnect" — public path, never crash.
            logger.info(f"Shield client closed: {e}")
        return

    logger.info("==================================================")
    logger.info(f"🔑 SECURE LOGIN GRANTED | Mode: {auth_label}")
    logger.info("==================================================")
    await websocket.accept()
    logger.info("Client connected — waiting for config.")

    # First message must be a config handshake from the client
    try:
        raw = await asyncio.wait_for(websocket.receive(), timeout=10.0)
        cfg = json.loads(raw.get("text", "{}"))
    except (asyncio.TimeoutError, Exception):
        cfg = {}

    # BLACK BOX: the system prompt is FIXED server-side (NCIHC Full Standard) —
    # whatever the client sends is ignored. Only the voice (Charon/Aoede) is honored.
    system_prompt = SYSTEM_PROMPT
    voice = cfg.get("voice", DEFAULT_VOICE)
    if voice not in AVAILABLE_VOICES:
        voice = DEFAULT_VOICE

    logger.info(f"Config received — voice={voice}, prompt_len={len(system_prompt)}")

    # Config factory para session resumption
    def get_live_config(resumption_token: str = None) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            # CRITICAL: Singular modality to prevent 1011 validation crash
            response_modalities=["AUDIO"],

            # Parallel sidecar transcription to capture text without modality conflicts
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),

            # Disabling automatic VAD for strict consecutive PTT interpretation
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)
            ),

            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),

            system_instruction=types.Content(
                parts=[types.Part(text=system_prompt)]
            ),

            # Session resumption for consultations exceeding the 15-minute audio limit
            # NOTE: transparent=True only works in Vertex AI Enterprise mode.
            # AI Studio (developer API) uses handle-only resumption.
            session_resumption=types.SessionResumptionConfig(
                handle=resumption_token
            ) if resumption_token else None,

            # Guard against context window memory exhaustion during extended dialogues.
            # CRITICAL: target_tokens MUST be large enough to never evict the ACTIVE turn.
            # The (immutable) system prompt is ~3.5k tokens; a 30s utterance is ~750 tokens.
            # With target_tokens=4000 the sliding window left ~500 tokens free and DELETED the
            # in-flight long utterance mid-stream → the model received nothing → silent "mute".
            # 25000 keeps the prompt + long turns + recent history comfortably in memory.
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=50000,
                sliding_window=types.SlidingWindow(target_tokens=25000)
            )
        )

    # Per-connection clinical intelligence scratchpad
    zga_session = SessionState()
    active_role: str = "CONDUIT"

    stop_event = asyncio.Event()
    audio_chunks_sent = 0
    current_resumption_token = None
    session_start_time = asyncio.get_event_loop().time()

    # Two separate state flags:
    # ptt_active  = user is pressing PTT right now → forward audio bytes to Gemini
    # model_busy  = model is generating a response → block new ptt_start
    ptt_active = False
    model_busy = False

    # ── Robustness state (Defects 1 & 2: long-phrase mute + noise/cross-talk) ──
    # last_input_is_english : language of the last thing the user said → so the
    #   "speak in short phrases" notice is delivered in the speaker's own language.
    # pending_recovery_notice : set when the live stream itself broke mid-turn; the
    #   notice is then spoken right after the session reconnects.
    last_input_is_english = True
    pending_recovery_notice = False

    # ── WATCHDOG state (the real fix: the model can HANG with no turn_complete) ──
    # awaiting_response       : True from the moment the user's turn ends until the
    #   model produces output or we recover. The watchdog only acts while this is True.
    # model_started_responding: flips True on the FIRST audio/text the model emits for
    #   the turn → proves it is alive, so a slow-but-working interpretation is never cut.
    # awaiting_since          : event-loop timestamp the current wait started.
    # intro_played            : the bilingual onboarding greeting plays once per connection.
    awaiting_response = False
    model_started_responding = False
    awaiting_since = 0.0
    intro_played = False
    # Adaptive watchdog: a longer input needs more time to interpret before we
    # call it a hang. We measure how long the user spoke (ptt_start→ptt_release)
    # and grant extra grace proportional to that.
    ptt_start_time = 0.0
    last_input_seconds = 0.0

    async def log(msg: str, level: str = "info"):
        logger.info(msg) if level == "info" else logger.error(msg)
        try:
            await websocket.send_json({"type": "log", "level": level, "msg": msg})
        except Exception:
            pass

    async def speak_directive(session, inject_text: str) -> bool:
        """Make the agent SAY a short canned line (same path as RESEARCHER injection)."""
        try:
            await session.send_realtime_input(activity_start=types.ActivityStart())
            await session.send_realtime_input(text=inject_text)
            await session.send_realtime_input(activity_end=types.ActivityEnd())
            return True
        except Exception as e:
            await log(f"❌ recovery directive inject error: {e}", "error")
            return False

    # Outer connection recovery loop for 15-min timeouts and 1011 infrastructure drops
    # Timeouts are FIXED (black box).
    _to_cfg = FIXED_CONFIG["timeouts"]
    _interp_cfg = FIXED_CONFIG["interpretation"]
    _max_retries = int(_interp_cfg.get("max_retries", 1))
    _ic_min_words = int(_interp_cfg.get("incoherence_min_words", 6))
    _ic_max_ratio = float(_interp_cfg.get("incoherence_max_stopword_ratio", 0.08))
    _wd_timeout = float(_to_cfg.get("interpret_timeout_s", 5))  # watchdog: seconds of total silence = hang
    reconnect_attempts = 0
    max_reconnect_attempts = int(_to_cfg.get("max_reconnects", 5))
    backoff_cap = int(_to_cfg.get("reconnect_backoff_cap_s", 16))
    session_warn = int(_to_cfg.get("session_warn_seconds", 870))
    key_idx = 0  # Start with LLC key (index 0)

    while not stop_event.is_set() and reconnect_attempts < max_reconnect_attempts:
        try:
            # Check if we need to reconnect proactively (approaching 15min limit)
            elapsed = asyncio.get_event_loop().time() - session_start_time
            if elapsed > session_warn:  # proactive reconnection before the limit
                await log("⚠️ Acercándose al límite de 15 min — reconectando proactivamente")
                session_start_time = asyncio.get_event_loop().time()

            active_key = _GEMINI_KEYS[key_idx % len(_GEMINI_KEYS)]
            key_label = "LLC" if key_idx % len(_GEMINI_KEYS) == 0 else "personal"
            # vertexai=False is CRITICAL: the ADK layer sets GOOGLE_GENAI_USE_VERTEXAI=TRUE
            # globally, but the Live session authenticates with an AI Studio API key —
            # which Vertex rejects ("API keys are not supported"). Force developer API here.
            client = genai.Client(api_key=active_key, vertexai=False)
            config = get_live_config(current_resumption_token)

            async with client.aio.live.connect(model=MODEL_ID, config=config) as session:
                if reconnect_attempts > 0:
                    await log(f"♻️ Sesión reanudada exitosamente (intento #{reconnect_attempts}) — key: {key_label}")
                else:
                    await log(f"✅ Sesión Gemini establecida — modelo: {MODEL_ID} — key: {key_label}")

                reconnect_attempts = 0  # Reset counter on successful connection

                await websocket.send_json({
                    "type": "status",
                    "message": "ZERO GRAVITY ONLINE",
                    "model": MODEL_ID,
                })

                # Reset per-session state
                ptt_active = False
                model_busy = False
                active_role = "CONDUIT"
                awaiting_response = False
                model_started_responding = False

                # ── Onboarding (once per connection) — TEXT, never audio ──
                # The intro is a chat message from the robot, not a spoken speech.
                # A page reset or reconnect can never trap anyone in a looping intro.
                if not intro_played:
                    intro_played = True
                    await log("👋 Intro de texto enviada (Zero Gravity Agent)")
                    await websocket.send_json({"type": "intro_text", "text": INTRO_TEXT})

                # ── Post-reconnect recovery notice (stream-break case) ──
                # If a prior turn died because the live stream broke, the FRESH session
                # now asks for short phrases — bilingually, in its OWN live voice.
                if pending_recovery_notice:
                    pending_recovery_notice = False
                    inject, sentence = short_phrase_directive()
                    await log("🗣 Recuperación tras reconexión → directiva bilingüe (voz live)", "warn")
                    await websocket.send_json({
                        "type": "state_preview",
                        "active_role": "CLARIFIER",
                        "notice": "recovery",
                        "session_memory": zga_session.to_dict(),
                    })
                    await websocket.send_json({"type": "agent_notice", "text": sentence})
                    await speak_directive(session, inject)

                # --- TASK 1: ASYNCHRONOUS RECEIVER LOOP ---
                # Plain English: this loop listens to GEMINI. It takes the interpretation
                # audio the model produces and sends it to the browser to be played, reads
                # the text the model speaks to decide the agent's state (semaphore), and —
                # when the model asks to research a term — calls the ADK reasoning team.
                async def receiver():
                    nonlocal current_resumption_token, ptt_active, model_busy, active_role
                    nonlocal last_input_is_english, pending_recovery_notice
                    nonlocal awaiting_response, model_started_responding, awaiting_since

                    # Transcript accumulator: buffer chunks → flush complete utterance on turn_complete
                    # This eliminates the "one word per entry" fragmentation bug.
                    output_buffer: str = ""
                    input_buffer:  str = ""
                    # state_preview_sent: prevents firing state_preview multiple times per turn
                    state_preview_sent: bool = False
                    # ── Long-phrase recovery state (Defect 1) ──
                    # interp_fail_count : silent retries spent on the CURRENT utterance.
                    # retry_pending_text: the buffered transcription we re-injected; non-empty
                    #   means "this turn is a retry — judge its result", which lets us detect a
                    #   second failure even though a re-injected turn has no audio transcription.
                    interp_fail_count: int = 0
                    retry_pending_text: str = ""
                    # ── Loop-breaker: detect the model repeating the SAME translation ──
                    prev_output: str = ""
                    repeat_count: int = 0
                    # ── Anti-ping-pong for the bilingual CLARIFIER backstop ──
                    # True while the turn we are receiving IS the forced missing-language
                    # repeat — that turn must never re-trigger the backstop itself.
                    bilingual_fix_pending: bool = False

                    while not stop_event.is_set():
                        try:
                            async for message in session.receive():
                                if stop_event.is_set():
                                    break

                                # Capture resumption tokens for session continuity
                                if message.session_resumption_update:
                                    upd = message.session_resumption_update
                                    if upd.new_handle:
                                        current_resumption_token = upd.new_handle
                                        await log("💾 Resumption token capturado")

                                sc = message.server_content
                                if not sc:
                                    continue

                                # ── Audio chunks → stream in real-time (no buffering) ──────
                                if sc.model_turn:
                                    for part in sc.model_turn.parts:
                                        if part.inline_data and part.inline_data.data:
                                            # The model is alive and speaking → disarm the watchdog
                                            # so a slow-but-working interpretation is never cut off.
                                            model_started_responding = True
                                            await websocket.send_bytes(part.inline_data.data)

                                # ── Input transcription → accumulate, extract clinical signals ──
                                if sc.input_transcription and sc.input_transcription.text:
                                    input_buffer += sc.input_transcription.text

                                # Any output transcription also proves the model is responding.
                                if sc.output_transcription and sc.output_transcription.text:
                                    model_started_responding = True

                                # ── Output transcription → accumulate + real-time state preview ──
                                # We buffer for display (no fragmentation), but scan partial text
                                # to fire state_preview the MOMENT a trigger phrase is detected.
                                # This updates the semaphore badge WHILE the agent is speaking.
                                if sc.output_transcription and sc.output_transcription.text:
                                    output_buffer += sc.output_transcription.text
                                    # Real-time preview: scan partial buffer once per turn
                                    if not state_preview_sent:
                                        preview_role, _ = semaphore_route(output_buffer, zga_session)
                                        if preview_role != "CONDUIT":
                                            state_preview_sent = True
                                            await log(f"🚦 STATE PREVIEW → {preview_role} (mid-stream)")
                                            await websocket.send_json({
                                                "type": "state_preview",
                                                "active_role": preview_role,
                                                "session_memory": zga_session.to_dict(),
                                            })

                                # ── Turn complete → flush buffers, route, send, reset ──────
                                if sc.turn_complete:
                                    # Process accumulated input: extract clinical signals
                                    last_input_text = input_buffer.strip()
                                    if last_input_text:
                                        await log(f"👂 Input completo: {input_buffer[:120]}")
                                        detected = zga_session.update_from_text(input_buffer)
                                        # Push any newly captured memory to the scratchpad
                                        if any(detected.values()):
                                            if detected["new_allergies"]:
                                                await log(f"🚨 ALERGIA DETECTADA: {detected['new_allergies']}", "warn")
                                            if detected["new_cultural_terms"]:
                                                await log(f"🌿 TÉRMINO CULTURAL: {detected['new_cultural_terms']}", "warn")
                                            if detected["new_symptoms"]:
                                                await log(f"🩺 Síntoma: {detected['new_symptoms']}")
                                            if detected["new_medications"]:
                                                await log(f"💊 Medicamento: {detected['new_medications']}")
                                            await websocket.send_json({
                                                "type": "session_update",
                                                "session_memory": zga_session.to_dict(),
                                                "new_allergies": detected["new_allergies"],
                                                "new_cultural_terms": detected["new_cultural_terms"],
                                                "new_symptoms": detected["new_symptoms"],
                                                "new_medications": detected["new_medications"],
                                            })
                                    input_buffer = ""
                                    if last_input_text:
                                        last_input_is_english = detect_is_english(last_input_text)

                                    # ════════════════════════════════════════════════════════
                                    # ROBUSTNESS GUARD — turn_complete arrived but with problems.
                                    # (The total-HANG case, where turn_complete NEVER arrives, is
                                    # handled separately by the watchdog task below.)
                                    # Recovery notices are spoken by the model in its OWN live
                                    # voice (bilingual directives) + mirrored as UI text — no
                                    # pre-recorded clips, no second voice, ever.
                                    # ════════════════════════════════════════════════════════
                                    produced = bool(output_buffer.strip())

                                    async def recover_with_voice(inject: str, sentence: str, notice: str):
                                        """Speak a bilingual recovery line in the live voice + UI text."""
                                        zga_session.lock_state("CLARIFIER")
                                        await websocket.send_json({
                                            "type": "state_preview", "active_role": "CLARIFIER",
                                            "notice": notice, "session_memory": zga_session.to_dict(),
                                        })
                                        await websocket.send_json({"type": "agent_notice", "text": sentence})
                                        await speak_directive(session, inject)

                                    # CASE A — this turn was a silent RETRY we injected. Judge its result.
                                    if retry_pending_text:
                                        if produced:
                                            # Retry worked → clear and fall through to normal routing.
                                            retry_pending_text = ""
                                            interp_fail_count = 0
                                        elif interp_fail_count < _max_retries:
                                            interp_fail_count += 1
                                            await log(f"🔁 Reintento silencioso #{interp_fail_count} (frase larga)", "warn")
                                            await speak_directive(session, retry_directive(retry_pending_text))
                                            output_buffer = ""
                                            state_preview_sent = False
                                            model_busy = True
                                            awaiting_since = asyncio.get_event_loop().time()
                                            model_started_responding = False
                                            continue
                                        else:
                                            # Retries exhausted → bilingual short-phrase ask, live voice.
                                            retry_pending_text = ""
                                            interp_fail_count = 0
                                            inject, sentence = short_phrase_directive()
                                            await log("🗣 Último recurso → directiva frases cortas (voz live)", "warn")
                                            await recover_with_voice(inject, sentence, "short_phrase")
                                            # HOLD: the directive generates a fresh model turn.
                                            output_buffer = ""
                                            state_preview_sent = False
                                            model_busy = True
                                            ptt_active = False
                                            awaiting_since = asyncio.get_event_loop().time()
                                            model_started_responding = False
                                            continue

                                    # CASE B — fresh utterance heard but NO interpretation produced (mute).
                                    elif last_input_text and not produced:
                                        if looks_incoherent(last_input_text, _ic_min_words, _ic_max_ratio):
                                            # Defect 2 — contamination so bad the meaning is unsafe.
                                            await log(f"🔊 Entrada contaminada (ruido/2º hablante) → aviso bilingüe voz live: {last_input_text[:70]}", "warn")
                                            inject, sentence = noise_directive()
                                            await recover_with_voice(inject, sentence, "noise")
                                            output_buffer = ""
                                            state_preview_sent = False
                                            model_busy = True
                                            ptt_active = False
                                            awaiting_since = asyncio.get_event_loop().time()
                                            model_started_responding = False
                                            continue
                                        if _max_retries > 0:
                                            # Defect 1 — likely too long / transient hiccup: retry silently.
                                            interp_fail_count = 1
                                            retry_pending_text = last_input_text
                                            await log("🔁 Interpretación vacía — reintento silencioso #1 (frase larga)", "warn")
                                            await speak_directive(session, retry_directive(last_input_text))
                                            output_buffer = ""
                                            state_preview_sent = False
                                            model_busy = True
                                            awaiting_since = asyncio.get_event_loop().time()
                                            model_started_responding = False
                                            continue
                                        # No retries configured → bilingual short-phrase ask, live voice.
                                        inject, sentence = short_phrase_directive()
                                        await log("🗣 Último recurso → directiva frases cortas (voz live)", "warn")
                                        await recover_with_voice(inject, sentence, "short_phrase")
                                        output_buffer = ""
                                        state_preview_sent = False
                                        model_busy = True
                                        ptt_active = False
                                        awaiting_since = asyncio.get_event_loop().time()
                                        model_started_responding = False
                                        continue

                                    # ── LOOP-BREAKER ──
                                    # If the model emits the SAME translation repeatedly (the
                                    # infinite-repeat failure after a 1008/reconnect), recycle the
                                    # session to break it instead of relaying the loop to the user.
                                    _norm = output_buffer.strip().lower()
                                    if _norm and _norm == prev_output:
                                        repeat_count += 1
                                    else:
                                        repeat_count = 0
                                        prev_output = _norm
                                    if repeat_count >= 2:
                                        await log("🛑 Loop detectado (misma traducción ≥3x) — reciclando sesión", "error")
                                        current_resumption_token = None  # fresh session, no replay
                                        model_busy = False
                                        ptt_active = False
                                        awaiting_response = False
                                        return  # end receiver → outer loop reconnects clean

                                    # Route complete output utterance through semaphore
                                    final_role = "CONDUIT"
                                    if output_buffer.strip():
                                        final_role, _ = semaphore_route(output_buffer, zga_session)
                                        zga_session.record_intervention(final_role, output_buffer)
                                        await log(f"📝 [{final_role}]: {output_buffer[:120]}")
                                        
                                        input_to_check = last_input_text if 'last_input_text' in locals() and last_input_text else ""
                                        is_english_input = detect_is_english(input_to_check)
                                        
                                        if final_role == "CONDUIT":
                                            if is_english_input and input_to_check:
                                                display_text = f"(medico-ingles) {input_to_check}"
                                            else:
                                                display_text = f"(paciente-español) {output_buffer.strip()}"
                                        else:
                                            display_text = f"(interpreter) {output_buffer.strip()}"

                                        await websocket.send_json({
                                            "type": "transcript",
                                            "text": display_text,
                                            "active_role": final_role,
                                            "role_label": ROLE_LABELS.get(final_role, final_role),
                                            "session_memory": zga_session.to_dict(),
                                        })

                                    # ── BILINGUAL GUARANTEE for CLARIFIER (deterministic backstop) ──
                                    # Gemini sometimes asks for clarification in ONLY one language,
                                    # stranding the other party. We detect the missing language and
                                    # force the model to repeat the request in that language with its
                                    # OWN live voice (same voice = magic intact). bilingual_fix_pending
                                    # stops the forced repeat from re-triggering this guard (ping-pong).
                                    if final_role == "CLARIFIER" and not bilingual_fix_pending:
                                        has_es, has_en = lang_presence(output_buffer)
                                        missing = None
                                        if not has_es:
                                            missing = "Spanish"
                                        elif not has_en:
                                            missing = "English"
                                        if missing:
                                            await log(f"🌐 Aclaración sin {missing} → repetición forzada en voz live", "warn")
                                            bilingual_fix_pending = True
                                            await speak_directive(session, bilingual_clarify_directive(missing))
                                            # HOLD: wait for the forced repeat's own turn_complete.
                                            output_buffer = ""
                                            state_preview_sent = False
                                            model_busy = True
                                            awaiting_since = asyncio.get_event_loop().time()
                                            model_started_responding = False
                                            continue
                                    elif bilingual_fix_pending:
                                        # This turn WAS the forced repeat — consume the flag.
                                        bilingual_fix_pending = False

                                    # ── RESEARCHER: ADK multi-agent reasoning (fallback RAG), inject, HOLD vad ──
                                    if final_role == "RESEARCHER":
                                        term = extract_research_term(output_buffer) or last_input_text[:60]
                                        await log(f"🔬 RESEARCHER → orquestador ADK: '{term}'")
                                        await websocket.send_json({
                                            "type": "researcher_start",
                                            "term": term,
                                            "active_role": "RESEARCHER",
                                            "session_memory": zga_session.to_dict(),
                                        })

                                        found, source, via, knowledge, err = False, "", "rag", "", ""
                                        agents_used: list[str] = []

                                        # Reasoning is wrapped: if BOTH the ADK layer and the RAG
                                        # fallback raise (network, Vertex outage), we must NOT crash
                                        # the live session — the agent simply interprets unaided.
                                        try:
                                            # 1) ADK multi-agent reasoning layer (orchestrator → specialist)
                                            allergies_csv = ", ".join(zga_session.allergies)
                                            adk_res = await adk_agents.run_clinical_reasoning(term, allergies=allergies_csv)
                                            if adk_res.get("available") and adk_res.get("text"):
                                                via = "adk"
                                                knowledge = adk_res["text"]
                                                found = adk_res.get("grounded", False)
                                                source = adk_res.get("source", "")
                                                agents_used = adk_res.get("agents", [])
                                                await log(f"🧠 ADK [{'+'.join(agents_used) or 'orchestrator'}]: {knowledge[:110]}")
                                            else:
                                                # 2) Fallback to direct Vertex RAG
                                                rag_res = await asyncio.to_thread(rag_engine.query, term)
                                                found = rag_res["found"]
                                                source = rag_res.get("source", "")
                                                knowledge = rag_res["context"] if found else ""
                                                err = adk_res.get("error", "") or rag_res.get("error", "")
                                                await log(f"🔬 RAG fallback — found={found} ({err})", "warn")
                                        except Exception as e:
                                            err = str(e)
                                            await log(f"⚠️ RESEARCHER reasoning falló ({err}) — interpreto sin base", "error")

                                        if knowledge:
                                            zga_session.record_clarified_term(term, knowledge[:160])
                                            inject = (
                                                f"[KNOWLEDGE BASE RESULT for '{term}']: {knowledge}\n\n"
                                                f"Now deliver the accurate interpretation using this confirmed meaning. "
                                                f"Do NOT mention the lookup or the knowledge base again."
                                            )
                                        else:
                                            inject = (
                                                f"[KNOWLEDGE BASE: no confirmed entry for '{term}'.] "
                                                f"Use your best professional judgment and interpret the message normally now. "
                                                f"Do NOT mention the knowledge base."
                                            )
                                        await websocket.send_json({
                                            "type": "researcher_result",
                                            "term": term,
                                            "found": found,
                                            "source": source,
                                            "via": via,
                                            "agents": agents_used,
                                            "error": err,
                                            "session_memory": zga_session.to_dict(),
                                        })
                                        # Inject the knowledge back into the live session → triggers informed turn
                                        try:
                                            await session.send_realtime_input(activity_start=types.ActivityStart())
                                            await session.send_realtime_input(text=inject)
                                            await session.send_realtime_input(activity_end=types.ActivityEnd())
                                        except Exception as e:
                                            await log(f"❌ RAG inject error: {e}", "error")
                                        # HOLD: do not reset/re-arm. The informed follow-up turn
                                        # will arrive and run the normal CONDUIT reset below.
                                        # Give that follow-up its own fresh watchdog window so a
                                        # post-research hang is still recoverable.
                                        output_buffer = ""
                                        state_preview_sent = False
                                        model_busy = True
                                        awaiting_since = asyncio.get_event_loop().time()
                                        model_started_responding = False
                                        continue  # wait for the informed turn's turn_complete

                                    output_buffer = ""
                                    state_preview_sent = False  # Reset for next turn

                                    # Hard reset to CONDUIT — guaranteed on every (non-research) turn end
                                    zga_session.increment_turn()
                                    model_busy = False
                                    ptt_active = False
                                    awaiting_response = False  # turn delivered → watchdog stands down
                                    active_role, _ = force_conduit(zga_session)
                                    await log("✔ turn_complete → CONDUIT")
                                    await websocket.send_json({
                                        "type": "turn_complete",
                                        "active_role": "CONDUIT",
                                        "session_memory": zga_session.to_dict(),
                                    })

                            await log("🔄 Generador receive() terminó — reiniciando")

                        except Exception as e:
                            if not stop_event.is_set():
                                await log(f"❌ Receiver error: {e}", "error")
                                # If the live stream broke mid-interpretation (common with a
                                # very long utterance), flag a recovery so the freshly
                                # reconnected session tells the user to use short phrases.
                                # Skip when the browser itself went away (no point notifying).
                                _es = str(e).lower()
                                if not any(s in _es for s in (
                                    "websocket.send", "websocket.close", "disconnect",
                                    "response already completed", "connectionclosed",
                                )):
                                    pending_recovery_notice = True
                                    # CRITICAL: drop the resumption handle on an ERROR reconnect.
                                    # Resuming a session that just aborted (e.g. 1008) replays the
                                    # in-flight turn → the model repeats the SAME translation in an
                                    # infinite loop. A fresh session breaks the loop; clinical facts
                                    # live in zga_session (RAM), not in the Gemini handle.
                                    current_resumption_token = None
                                break

                # --- TASK 2: ASYNCHRONOUS SENDER LOOP ---
                # Plain English: this loop listens to the BROWSER. It forwards the user's
                # microphone audio up to Gemini while they are speaking, and handles the
                # control messages (start speaking / done speaking) from the mic button/VAD.
                async def sender():
                    nonlocal ptt_active, model_busy, audio_chunks_sent, active_role
                    nonlocal awaiting_response, model_started_responding, awaiting_since
                    nonlocal ptt_start_time, last_input_seconds

                    # ── AUDIO AGGREGATION BUFFER (Deep-Research fix) ──
                    # The browser sends ~256 B every ~8 ms (~125 msg/s). Forwarding each one
                    # individually starves the event loop (→ 1011 keepalive timeout) and creates
                    # network jitter so chunks arrive AFTER activity_end (→ Gemini 3.1 silently
                    # aborts inference → mute). We coalesce into ~100 ms (3200 B) chunks: ~10 msg/s,
                    # matching Google's recommended cadence and AI Studio's behavior.
                    AUDIO_AGG_BYTES = 3200  # 100 ms @ 16 kHz mono int16
                    audio_agg = bytearray()

                    async def _flush_audio():
                        nonlocal audio_agg
                        if audio_agg:
                            await session.send_realtime_input(
                                audio=types.Blob(data=bytes(audio_agg), mime_type="audio/pcm;rate=16000")
                            )
                            audio_agg.clear()

                    try:
                        while not stop_event.is_set():
                            msg = await websocket.receive()

                            if "bytes" in msg and msg["bytes"]:
                                # Only forward audio while the user is actively pressing PTT.
                                # Late bytes that arrive after release are dropped here — this is
                                # what prevents the post-activity_end race condition.
                                if not ptt_active:
                                    continue
                                audio_agg.extend(msg["bytes"])
                                if len(audio_agg) >= AUDIO_AGG_BYTES:
                                    audio_chunks_sent += 1
                                    if audio_chunks_sent % 10 == 1:
                                        await log(f"🎤 Audio agregado #{audio_chunks_sent} → Gemini ({len(audio_agg)} bytes)")
                                    await _flush_audio()

                            elif "text" in msg:
                                data = json.loads(msg["text"])
                                t = data.get("type")

                                if t == "ptt_start":
                                    if model_busy:
                                        await log("⚠️ PTT start ignorado — modelo generando respuesta", "warn")
                                        continue
                                    audio_chunks_sent = 0
                                    audio_agg.clear()  # fresh turn
                                    ptt_active = True
                                    ptt_start_time = asyncio.get_event_loop().time()
                                    await log("▶ PTT start → activity_start enviado a Gemini")
                                    await session.send_realtime_input(activity_start=types.ActivityStart())

                                elif t == "ptt_release":
                                    if not ptt_active:
                                        continue  # Ignore stray releases
                                    ptt_active = False  # drop the gate: no more audio accepted
                                    # STRICT STATE GATE: flush ALL buffered audio across the network
                                    # BEFORE the boundary signal, so no fragment can arrive after
                                    # activity_end (the race condition that mutes Gemini 3.1).
                                    await _flush_audio()
                                    model_busy = True  # Block new PTT until turn_complete
                                    # ARM THE WATCHDOG: from now the model must start speaking
                                    # within the (adaptive) timeout, or we recover from a hang.
                                    last_input_seconds = asyncio.get_event_loop().time() - ptt_start_time
                                    awaiting_response = True
                                    model_started_responding = False
                                    awaiting_since = asyncio.get_event_loop().time()
                                    await log(f"⏹ PTT release → flush + activity_end (input ~{last_input_seconds:.1f}s)")
                                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                                    # Formally signal the audio stream is paused (cleaner state machine).
                                    try:
                                        await session.send_realtime_input(audio_stream_end=True)
                                    except Exception:
                                        pass  # older SDKs may not expose this kwarg

                                elif t == "ping":
                                    await websocket.send_json({"type": "pong"})

                    except WebSocketDisconnect:
                        await log("Client disconnected.")
                        raise  # Break out to gracefully clean up the connection

                # --- TASK 3: WATCHDOG (the real long-phrase fix) ---
                # Plain English: this loop simply watches the clock. The mute/noise GUARD in
                # receiver() only runs if turn_complete arrives. But the worst failure is when
                # turn_complete NEVER arrives — the model just hangs in "interpreting". So this
                # task fires on TIME, not on events: if the model has not started speaking within
                # _wd_timeout seconds of the user finishing, it plays the pre-recorded
                # "speak in short phrases" clip (in the speaker's language) and recycles the
                # live session so the next turn starts clean.
                async def watchdog():
                    nonlocal awaiting_response, model_busy, ptt_active, active_role
                    nonlocal current_resumption_token, pending_recovery_notice
                    loop = asyncio.get_event_loop()
                    while not stop_event.is_set():
                        await asyncio.sleep(0.4)
                        if not awaiting_response or model_started_responding:
                            continue
                        # Adaptive grace: a long utterance legitimately takes longer to
                        # interpret, so give +0.5s per second spoken beyond 3s, capped at 12s.
                        effective_timeout = min(_wd_timeout + max(0.0, last_input_seconds - 3.0) * 0.5, 15.0)
                        if (loop.time() - awaiting_since) < effective_timeout:
                            continue
                        # ── HANG DETECTED ──
                        # The model is mute and cannot speak, so: 1) show the bilingual
                        # notice as TEXT right now, 2) recycle the session, 3) the fresh
                        # session speaks it in the SAME live voice (pending_recovery_notice).
                        await log(f"⏱ Watchdog: agente mudo >{_wd_timeout:.0f}s — recuperando (texto + voz live tras reciclar)", "warn")
                        awaiting_response = False
                        pending_recovery_notice = True
                        zga_session.lock_state("CLARIFIER")
                        try:
                            await websocket.send_json({
                                "type": "state_preview", "active_role": "CLARIFIER",
                                "notice": "short_phrase", "session_memory": zga_session.to_dict(),
                            })
                            await websocket.send_json({"type": "agent_notice", "text": SHORT_PHRASE_SENTENCE})
                        except Exception:
                            pass
                        # Recycle: drop the resumption token so the reconnect is a FRESH session
                        # (a stuck generation cannot carry over), then end this task to trigger it.
                        current_resumption_token = None
                        model_busy = False
                        ptt_active = False
                        force_conduit(zga_session)
                        active_role = "CONDUIT"
                        try:
                            await websocket.send_json({
                                "type": "turn_complete", "active_role": "CONDUIT",
                                "session_memory": zga_session.to_dict(),
                            })
                        except Exception:
                            pass
                        return  # completing this task → asyncio.wait recycles the session

                # Execute the three I/O tasks concurrently
                recv_task = asyncio.create_task(receiver())
                send_task = asyncio.create_task(sender())
                wd_task   = asyncio.create_task(watchdog())

                # Wait for any task to finish (fail, disconnect, or watchdog recovery)
                done, pending = await asyncio.wait(
                    [recv_task, send_task, wd_task],
                    return_when=asyncio.FIRST_COMPLETED
                )

                # Cancel the surviving task
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

        except WebSocketDisconnect:
            await log("Cliente desconectado — cerrando sesión Gemini")
            break  # Clean exit initiated by the client browser

        except Exception as e:
            err_str = str(e).lower()
            # ── Client gone: stop the loop, do NOT reconnect ──────────────────
            # If the browser closed/reloaded, the ASGI socket is dead. Any further
            # send raises "websocket.send after websocket.close / response already
            # completed" or "disconnect". Reconnecting Gemini in this state spins
            # an infinite error loop — so we detect it and break cleanly.
            if any(sig in err_str for sig in (
                "websocket.send", "websocket.close", "response already completed",
                "disconnect", "connectionclosed", "cannot call \"send\"",
            )):
                logger.info("Cliente WebSocket cerrado — terminando sesión (sin reconectar)")
                stop_event.set()
                break

            # Rotate to next key on quota/auth errors before burning retry attempts
            if ("quota" in err_str or "billing" in err_str or "api_key" in err_str) and len(_GEMINI_KEYS) > 1:
                next_label = "personal" if key_idx % len(_GEMINI_KEYS) == 0 else "LLC"
                key_idx += 1
                await log(f"⚡ Quota/auth error en key {key_label} — cambiando a key {next_label}", "error")
                session_start_time = asyncio.get_event_loop().time()
                continue  # Retry immediately with new key, don't burn a reconnect attempt
            reconnect_attempts += 1
            if reconnect_attempts < max_reconnect_attempts:
                # Execute Exponential Backoff and Transparent Resumption
                backoff = min(2 ** reconnect_attempts, backoff_cap)  # Cap from config
                await log(f"❌ Session error: {e} — reconectando en {backoff}s (intento #{reconnect_attempts})", "error")
                # A mid-interpretation crash → notify the user (short phrases) once back up.
                pending_recovery_notice = True
                # Fresh session on error (no resumption replay → no infinite-loop repeat).
                current_resumption_token = None
                await asyncio.sleep(backoff)
                session_start_time = asyncio.get_event_loop().time()  # Reset timer
            else:
                await log(f"💥 CRITICAL: Max reconnect attempts reached. Session failed permanently.", "error")
                try:
                    await websocket.send_json({"type": "error", "message": f"Session failed: {e}"})
                except Exception:
                    pass
                break


# ── DEMO: Patient Voice Scenarios ────────────────────────────────────────────
DEMO_SCENARIOS = {
    "er_chest": {
        "id": "er_chest", "label": "ER — Chest Pain", "specialty": "Emergency", "emoji": "🚨",
        "patient": "Maria, 54 — Venezuela", "voice": "Aoede", "color": "#ef4444",
        "dna": "Venezuelan woman, 54, severe distress, trembling voice, rapid breathing. Speaks fast out of fear.",
        "script": "Doctor, me siento muy mal, tengo calentura desde esta mañana y me late muy fuerte el corazón. Siento como una punzada aquí en el pecho que se va al brazo izquierdo. Y se me fue la vista un momento hace como media hora. Tengo miedo.",
        "english_script": "Doctor, I feel very bad, I have had a fever since this morning and my heart is beating very fast. I feel like a stabbing pain here in my chest that goes to my left arm. And my vision went dark for a moment about half an hour ago. I am scared.",
    },
    "pediatrics_fever": {
        "id": "pediatrics_fever", "label": "Pediatrics — Fever & Diarrhea", "specialty": "Pediatrics", "emoji": "👶",
        "patient": "Carlos, 32 — Mexico", "voice": "Charon", "color": "#f59e0b",
        "dna": "Mexican man, 32, worried father, calm but tense voice.",
        "script": "Doctora, mi niño tiene chorro desde ayer por la noche, como cinco veces ya. Y tiene calentura, como de 38 grados y medio. No quiere comer nada y está muy flojo. Tiene dos años.",
        "english_script": "Doctor, my child has had diarrhea since last night, like five times already. And he has a fever, of about 38.5 degrees. He doesn't want to eat anything and is very weak. He is two years old.",
    },
    "ob_movement": {
        "id": "ob_movement", "label": "Obstetrics — Fetal Movement", "specialty": "OB/GYN", "emoji": "🤰",
        "patient": "Rosa, 28 — Puerto Rico", "voice": "Aoede", "color": "#8b5cf6",
        "dna": "Puerto Rican woman, 28, pregnant, highly anxious, voice between frightened and hopeful.",
        "script": "Enfermera, estoy en la semana 36 y siento que el bebé no se mueve como antes. Antes me pateaba bastante y hoy casi no lo siento. También me duele la barriga, como presión aquí abajo.",
        "english_script": "Nurse, I am in week 36 and I feel that the baby is not moving like before. Before, he used to kick me a lot and today I can barely feel him. Also, my abdomen hurts, like a pressure down here.",
    },
    "mental_health_si": {
        "id": "mental_health_si", "label": "Mental Health — Suicidal Ideation", "specialty": "Mental Health", "emoji": "🧠",
        "patient": "Luis, 41 — Colombia", "voice": "Charon", "color": "#6b7280",
        "dna": "Colombian man, 41, flat and monotonous voice, speaks slowly. Severe depression.",
        "script": "A veces pienso que sería mejor si no estuviera aquí. Mi familia estaría mejor sin mí. Ya tengo un plan, tengo pastillas en casa guardadas. No sé para qué sigo viniendo al médico.",
        "english_script": "Sometimes I think it would be better if I weren't here. My family would be better off without me. I already have a plan, I have pills stored at home. I don't know why I keep coming to the doctor.",
    },
    "folk_susto": {
        "id": "folk_susto", "label": "Folk Illness — Fright (Susto)", "specialty": "Cultural Brokering", "emoji": "🌿",
        "patient": "Pedro, 38 — Guatemala", "voice": "Charon", "color": "#10b981",
        "dna": "Guatemalan man, 38, describes symptoms culturally, deliberate pacing.",
        "script": "Doctor, creo que me cayó un susto muy fuerte cuando vi ese accidente el mes pasado. Desde entonces no puedo dormir, me tiemblan las manos, y siento frío de huesos aunque haga calor.",
        "english_script": "Doctor, I think I was struck by a very strong fright (susto) when I saw that accident last month. Since then, I can't sleep, my hands shake, and I feel chills in my bones (frío de huesos) even if it's hot.",
    },
    "consent_surgery": {
        "id": "consent_surgery", "label": "Informed Consent — Surgery", "specialty": "Consent", "emoji": "📋",
        "patient": "Elena, 67 — Ecuador", "voice": "Aoede", "color": "#3b82f6",
        "dna": "Ecuadorian elderly woman, 67, confused, basic education, calm but insecure voice.",
        "script": "Me dijeron que tengo que firmar unos papeles pero yo no entendí bien qué me van a hacer. Sí, dije que sí a todo pero la verdad es que no entendí. ¿Me pueden explicar de nuevo? Tengo miedo de que me operen y no despertar.",
        "english_script": "They told me I have to sign some papers but I didn't understand well what they are going to do to me. Yes, I said yes to everything, but the truth is I didn't understand. Can you explain it to me again? I'm afraid they will operate on me and I won't wake up.",
    },
    "family_interference": {
        "id": "family_interference", "label": "Family Interference", "specialty": "Role Boundary", "emoji": "⚠️",
        "patient": "Grandmother, 72 — Dominican Republic", "voice": "Aoede", "color": "#f97316",
        "dna": "Dominican elderly woman, soft voice, speaks quietly as if afraid.",
        "script": "Mija, mi hijo me dijo que no le diga al médico que yo también tomo las pastillas del marido. Me duele mucho el estómago y no sé si es por eso. Por favor no le diga a mi hijo que le conté esto.",
        "english_script": "My child, my son told me not to tell the doctor that I also take my husband's pills. My stomach hurts a lot and I don't know if it's because of that. Please don't tell my son that I told you this.",
    },
    "er_seizure": {
        "id": "er_seizure", "label": "ER — Post-Ictal Seizure", "specialty": "Emergency", "emoji": "⚡",
        "patient": "Jorge, 29 — El Salvador", "voice": "Charon", "color": "#ec4899",
        "dna": "Salvadoran man, 29, confused and dazed post-seizure, speaks slowly and is disoriented.",
        "script": "Yo no sé qué pasó. Me dijeron que me dio un ataque pero yo no recuerdo nada. Siento la lengua mordida y me duele mucho la cabeza. Tengo epilepsia desde niño pero hace tres años no me daba. Creo que no tomé el medicamento ayer.",
        "english_script": "I don't know what happened. They told me I had a seizure but I don't remember anything. I feel my tongue is bitten and my head hurts a lot. I've had epilepsy since I was a child but I hadn't had a seizure in three years. I think I didn't take the medication yesterday.",
    },
}

# ── SEMAPHORE TRIGGER TEST LAB ───────────────────────────────────────────────
# Each test has an `expected_state`. Run it → watch the semaphore → UI marks ✓/✗.
# Use to measure & tune toward 99% state-switching accuracy.
_TT_VOICE = "Charon"
TRIGGER_TESTS = {
    "tt_conduit": {
        "id": "tt_conduit", "label": "Baseline — Normal Pain", "specialty": "🚦 Trigger Test",
        "emoji": "🟢", "patient": "Test → CONDUIT", "voice": "Aoede", "color": "#10b981",
        "expected_state": "CONDUIT",
        "dna": "Latino patient, calm and clear voice.",
        "script": "Doctor, me duele la cabeza desde ayer en la mañana y tengo un poco de fiebre.",
        "english_script": "Doctor, my head hurts since yesterday morning and I have a bit of a fever.",
    },
    "tt_nonsense": {
        "id": "tt_nonsense", "label": "Nonsense — Unicorn", "specialty": "🚦 Trigger Test",
        "emoji": "🟡", "patient": "Test → CLARIFIER", "voice": "Charon", "color": "#f59e0b",
        "expected_state": "CLARIFIER",
        "dna": "Confused patient, speaks confidently but speaks incoherently (possible aphasia).",
        "script": "Mi unicornio no encendía porque el autobús iba en reversa, y entonces las ventanas se comieron el reloj azul.",
        "english_script": "My unicorn wouldn't turn on because the bus was in reverse, and then the windows ate the blue clock.",
    },
    "tt_ambiguous": {
        "id": "tt_ambiguous", "label": "Ambiguous — Sudden Chill", "specialty": "🚦 Trigger Test",
        "emoji": "🟡", "patient": "Test → CLARIFIER", "voice": "Aoede", "color": "#f59e0b",
        "expected_state": "CLARIFIER",
        "dna": "Elderly woman, worried, describes symptoms ambiguously.",
        "script": "Doctor, anoche me dio un aire y desde entonces no me siento bien de este lado.",
        "english_script": "Doctor, last night I caught a cold draft (un aire) and since then I don't feel well on this side.",
    },
    "tt_susto": {
        "id": "tt_susto", "label": "Folk — Fright (Susto)", "specialty": "🚦 Trigger Test",
        "emoji": "🟣", "patient": "Test → CULTURAL", "voice": "Charon", "color": "#8b5cf6",
        "expected_state": "CULTURAL_BROKER",
        "dna": "Central American man, speaks slowly, describes in cultural terms.",
        "script": "Doctor, yo creo que me cayó un susto muy fuerte cuando vi el accidente, desde ahí no duermo.",
        "english_script": "Doctor, I think I was struck by a very strong fright (susto) when I saw the accident, since then I don't sleep.",
    },
    "tt_empacho": {
        "id": "tt_empacho", "label": "Folk — Indigestion (Empacho)", "specialty": "🚦 Trigger Test",
        "emoji": "🟣", "patient": "Test → CULTURAL", "voice": "Aoede", "color": "#8b5cf6",
        "expected_state": "CULTURAL_BROKER",
        "dna": "Mexican mother, worried about her child.",
        "script": "Mi niño tiene empacho desde que comió, está bien llorón, creo que también tiene mal de ojo.",
        "english_script": "My child has indigestion (empacho) since he ate, he is very tearful, I think he also has evil eye (mal de ojo).",
    },
    "tt_researcher": {
        "id": "tt_researcher", "label": "Rare Illness — Sydenham's Chorea", "specialty": "🚦 Trigger Test",
        "emoji": "🔬", "patient": "Test → RESEARCHER", "voice": "Charon", "color": "#38bdf8",
        "expected_state": "RESEARCHER",
        "dna": "Patient reporting a rare diagnosis with an uncommon name.",
        "script": "El doctor del otro hospital me dijo que tengo el mal de San Vito, pero no entendí bien qué es eso.",
        "english_script": "The doctor from the other hospital told me that I have Saint Vitus' Dance (mal de San Vito), but I didn't quite understand what that is.",
    },
    "tt_advocate_si": {
        "id": "tt_advocate_si", "label": "Safety — Suicidal Ideation", "specialty": "🚦 Trigger Test",
        "emoji": "🔴", "patient": "Test → ADVOCATE", "voice": "Charon", "color": "#ef4444",
        "expected_state": "ADVOCATE",
        "dna": "Man with severe depression, flat and monotonous voice.",
        "script": "A veces pienso que sería mejor no estar aquí, ya tengo unas pastillas guardadas en la casa.",
        "english_script": "Sometimes I think it would be better not to be here, I already have some pills stored at home.",
    },
    "tt_advocate_dose": {
        "id": "tt_advocate_dose", "label": "Safety — Dosage Confusion", "specialty": "🚦 Trigger Test",
        "emoji": "🔴", "patient": "Test → ADVOCATE", "voice": "Aoede", "color": "#ef4444",
        "expected_state": "ADVOCATE",
        "dna": "Patient reporting dangerous dosage confusion.",
        "script": "El doctor dijo veinticinco miligramos pero yo entendí que eran cincuenta, ya me tomé dos.",
        "english_script": "The doctor said twenty-five milligrams but I understood it was fifty, I already took two.",
    },
}
DEMO_SCENARIOS.update(TRIGGER_TESTS)

TTS_MODEL = "gemini-3.1-flash-tts-preview"
TTS_RATE_IN = 24000
ZGA_RATE = 16000


@app.get("/api/scenarios")
async def get_scenarios():
    fields = ("id", "label", "specialty", "emoji", "patient", "script", "color", "english_script")
    out = {}
    for k, v in DEMO_SCENARIOS.items():
        item = {f: v[f] for f in fields}
        if "expected_state" in v:
            item["expected_state"] = v["expected_state"]
        out[k] = item
    return out


class SynthRequest(BaseModel):
    scenario_id: str
    token: str | None = None
    judge_token: str | None = None


@app.post("/api/synthesize")
async def synthesize_demo(req: SynthRequest):
    s = DEMO_SCENARIOS.get(req.scenario_id)
    if not s:
        return Response(content='{"error":"not found"}', media_type="application/json", status_code=404)

    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    env_judge_token  = os.environ.get("JUDGE_SECRET_TOKEN", "").strip()

    is_authorized_live = False
    if env_judge_token and req.judge_token and req.judge_token.strip() == env_judge_token:
        is_authorized_live = True
    elif google_client_id and req.token:
        idinfo = verify_google_token(req.token, google_client_id)
        if idinfo:
            email = idinfo.get("email", "").lower().strip()
            if email in ALLOWED_EMAILS:
                is_authorized_live = True

    if not os.environ.get("K_SERVICE"):
        is_authorized_live = True
    if not google_client_id and not env_judge_token:
        is_authorized_live = True

    if not is_authorized_live:
        # Shield mode: no live TTS spend for unauthorized visitors (silence stub).
        logger.info(f"Shield Mode: serving silence stub for scenario {req.scenario_id}")
        return Response(content=b'\x00' * 32000, media_type="audio/pcm")

    try:
        key = _GEMINI_KEYS[0] if _GEMINI_KEYS else None
        if not key:
            return Response(content='{"error":"no api key"}', media_type="application/json", status_code=500)
        # vertexai=False: TTS uses the AI Studio API key, not Vertex (see Live client note).
        client = genai.Client(api_key=key, vertexai=False)
        prompt = f"Direction: {s['dna']}\n\nPerform this script as a real patient:\n{s['script']}"
        res = client.models.generate_content(
            model=TTS_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=s["voice"])
                    )
                )
            )
        )
        audio_data = None
        for part in res.candidates[0].content.parts:
            if part.inline_data:
                audio_data = part.inline_data.data
                break
        if not audio_data:
            return Response(content='{"error":"no audio"}', media_type="application/json", status_code=500)
        # Resample 24kHz → 16kHz using numpy linear interpolation
        samples_in  = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        target_len  = int(len(samples_in) * ZGA_RATE / TTS_RATE_IN)
        samples_out = np.interp(
            np.linspace(0, len(samples_in) - 1, target_len),
            np.arange(len(samples_in)),
            samples_in,
        ).astype(np.int16)
        pcm_16k = samples_out.tobytes()
        return Response(content=pcm_16k, media_type="audio/pcm")
    except Exception as e:
        logger.error(f"Demo synthesis error: {e}")
        return Response(content=f'{{"error":"{e}"}}', media_type="application/json", status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
