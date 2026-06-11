"""
ZGA State Manager — Live Clinical Memory (the "scratchpad")
==========================================================
PLAIN ENGLISH: As the conversation happens, this file quietly writes down the
important clinical facts it hears — allergies, symptoms, medications, folk-illness
terms, and which interventions the agent has already made. The UI shows this as a
live "scratchpad" so both the human and the agent can see the session's memory.

WHY IT MATTERS: Keeping a factual record (instead of relying on the model's memory)
stops the agent from hallucinating or re-asking things it already clarified. It also
powers the ADVOCATE safety check (e.g. "the patient is allergic to penicillin").

PRIVACY: Everything lives only in RAM for the duration of one connection. Nothing is
written to disk — no Protected Health Information (PHI) is stored (HIPAA-aligned).

Each browser connection gets its own SessionState object. Detection is deterministic
(pattern-based), not AI — so it is predictable and cannot invent facts.
"""
import re
import uuid
from typing import Any


# ── Pattern Libraries ─────────────────────────────────────────────────────────

_ALLERGY_PATTERNS = [
    # Spanish — accented and unaccented variants
    re.compile(r"soy\s+al[eé]rgic[oa]\s+(?:al?|a\s+la\s+|a\s+los?\s+|a\s+las?\s+)?([a-záéíóúñü\s\-]+)", re.I),
    re.compile(r"tengo\s+alergia\s+(?:al?|a\s+la\s+|a\s+los?\s+|a\s+las?\s+)?([a-záéíóúñü\s\-]+)", re.I),
    re.compile(r"soy\s+al[eé]rgica?\s+al?\s+([a-záéíóúñü\s\-]+)", re.I),
    re.compile(r"me\s+da\s+(?:alergia|reacci[oó]n)\s+(?:el|la|los|las|al?|a\s+la)?\s*([a-záéíóúñü\s\-]+)", re.I),
    re.compile(r"no\s+puedo\s+tomar\s+(?:el|la|los|las)?\s*([a-záéíóúñü\s\-]+)", re.I),
    re.compile(r"me\s+hace\s+(?:da[nñ]o|mal)\s+(?:el|la|los|las)?\s*([a-záéíóúñü\s\-]+)", re.I),
    # English
    re.compile(r"allergic\s+to\s+([a-z\s\-]+)", re.I),
    re.compile(r"i(?:'m|\s+am)\s+allergic\s+to\s+([a-z\s\-]+)", re.I),
    re.compile(r"i\s+have\s+an?\s+allergy\s+to\s+([a-z\s\-]+)", re.I),
]

_FOLK_TERMS = [
    "susto", "mal de ojo", "empacho", "daño", "mal puesto", "sereno", "pasmo",
    "frío de huesos", "curandera", "curandero", "sobada", "limpia", "espanto",
    "bilis", "mollera caída", "caída de mollera", "nervios", "ataque de nervios",
    "brujería", "hechizo", "desbarajuste",
]

_CULTURAL_PATTERNS = [re.compile(r"\b" + re.escape(t) + r"\b", re.I) for t in _FOLK_TERMS]

_MEDICATION_NAMES = [
    # Common high-risk medications mentioned in clinical encounters
    "penicilina", "penicillin", "amoxicilina", "amoxicillin", "ampicilina",
    "aspirina", "aspirin", "ibuprofeno", "ibuprofen", "naproxeno",
    "sulfa", "sulfamida", "sulfamethoxazole", "trimetoprim",
    "morfina", "morphine", "codeína", "codeine", "tramadol",
    "warfarina", "warfarin", "coumadin", "heparina", "heparin",
    "metformina", "metformin", "insulina", "insulin",
    "lisinopril", "metoprolol", "atenolol", "amlodipina",
    "atorvastatina", "simvastatina", "statins",
    "diazepam", "lorazepam", "alprazolam",
    "cefalexina", "cephalexin", "ciprofloxacino", "ciprofloxacin",
    "metotrexato", "methotrexate", "prednisona", "prednisone",
]
_MED_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _MEDICATION_NAMES) + r")\b", re.I
)

# Symptom lexicon — deterministic capture of chief complaints for session memory.
# Stored as canonical English label so the scratchpad reads cleanly.
_SYMPTOM_MAP = {
    "fiebre": "fever", "calentura": "fever", "fever": "fever",
    "dolor de cabeza": "headache", "headache": "headache", "cefalea": "headache",
    "dolor de pecho": "chest pain", "chest pain": "chest pain",
    "punzada": "stabbing pain", "ardor": "burning sensation",
    "diarrea": "diarrhea", "chorro": "diarrhea", "diarrhea": "diarrhea",
    "vómito": "vomiting", "vomito": "vomiting", "náusea": "nausea", "nausea": "nausea",
    "mareo": "dizziness", "mareos": "dizziness", "dizziness": "dizziness",
    "tos": "cough", "cough": "cough", "falta de aire": "shortness of breath",
    "dificultad para respirar": "shortness of breath", "ahogo": "shortness of breath",
    "sangrado": "bleeding", "sangre": "bleeding", "bleeding": "bleeding",
    "convulsión": "seizure", "convulsion": "seizure", "ataque": "seizure",
    "presión alta": "high blood pressure", "presión baja": "low blood pressure",
    "hinchazón": "swelling", "hinchado": "swelling", "swelling": "swelling",
    "no puedo dormir": "insomnia", "insomnio": "insomnia",
    "se me fue la vista": "vision loss", "visión borrosa": "blurred vision",
    "temblor": "tremor", "temblores": "tremor", "tiemblan": "tremor",
    "dolor de barriga": "abdominal pain", "dolor de estómago": "abdominal pain",
    "dolor abdominal": "abdominal pain",
}
_SYMPTOM_PATTERNS = [
    (re.compile(r"\b" + re.escape(k) + r"\b", re.I), v)
    for k, v in _SYMPTOM_MAP.items()
]


def _clean_capture(raw: str) -> str:
    """Strip trailing noise from a regex capture group."""
    noise = re.compile(
        r"\b(?:que|me|y|pero|aunque|y también|también|es|la|el|los|las|un|una|porque|"
        r"cuando|desde|hace|más|mucho|muy|siempre|nunca)\b.*$",
        re.I,
    )
    cleaned = noise.sub("", raw).strip().rstrip(".,;:?!")
    return cleaned if len(cleaned) > 2 else ""


# ── Session State Class ───────────────────────────────────────────────────────

class SessionState:
    """
    One instance per WebSocket connection.
    Lives entirely in RAM — no persistence, no PHI storage.
    """

    def __init__(self):
        self.session_id: str = str(uuid.uuid4())[:8]
        self.allergies: list[str] = []
        self.cultural_terms: list[str] = []
        self.current_locked_state: str | None = None
        self.pending_clarification_phrase: str | None = None
        # ── Session memory (anti-hallucination scratchpad) ──
        self.symptoms: list[str] = []          # chief complaints captured deterministically
        self.medications: list[str] = []       # drugs mentioned (any context)
        self.clarified_terms: list[dict] = []   # [{term, meaning}] resolved via RESEARCHER/CLARIFIER
        self.interventions: list[dict] = []      # [{state, snippet}] log of non-CONDUIT actions
        self.turn_count: int = 0                 # completed interpretation turns

    # ── Mutation ──────────────────────────────────────────────────────────────

    def update_from_text(self, text: str) -> dict[str, list[str]]:
        """
        Extract clinical signals from a transcription chunk.
        Returns a dict of what was newly detected so callers can log it.
        """
        new_allergies: list[str] = []
        new_cultural: list[str] = []

        # Allergy extraction
        for pattern in _ALLERGY_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1)
                # Check if the captured substance is a known medication or generic noun
                candidate = _clean_capture(raw)
                if candidate and candidate.lower() not in [a.lower() for a in self.allergies]:
                    self.allergies.append(candidate)
                    new_allergies.append(candidate)

        # Also flag direct medication name mentions near allergy keywords
        allergy_context = re.compile(r"al[eé]rgic|alergia|allergic|reacci[oó]n|no\s+puedo\s+tomar", re.I)
        if allergy_context.search(text):
            for med_match in _MED_PATTERN.finditer(text):
                med = med_match.group(1).lower()
                if med not in [a.lower() for a in self.allergies]:
                    self.allergies.append(med)
                    new_allergies.append(med)

        # Cultural / folk term extraction
        for pattern in _CULTURAL_PATTERNS:
            for match in pattern.finditer(text):
                term = match.group(0).lower()
                if term not in [c.lower() for c in self.cultural_terms]:
                    self.cultural_terms.append(term)
                    new_cultural.append(term)

        # Symptom extraction (canonical English label, deduped)
        new_symptoms: list[str] = []
        for pattern, label in _SYMPTOM_PATTERNS:
            if pattern.search(text) and label not in self.symptoms:
                self.symptoms.append(label)
                new_symptoms.append(label)

        # Medication mentions (any context, not only allergy)
        new_meds: list[str] = []
        for med_match in _MED_PATTERN.finditer(text):
            med = med_match.group(1).lower()
            if med not in [m.lower() for m in self.medications]:
                self.medications.append(med)
                new_meds.append(med)

        return {
            "new_allergies": new_allergies,
            "new_cultural_terms": new_cultural,
            "new_symptoms": new_symptoms,
            "new_medications": new_meds,
        }

    # ── Session memory recorders ────────────────────────────────────────────

    def record_intervention(self, state: str, snippet: str = ""):
        """Log a non-CONDUIT action so the session has an audit trail."""
        if state and state != "CONDUIT":
            self.interventions.append({"state": state, "snippet": (snippet or "")[:90]})
            # keep memory bounded
            if len(self.interventions) > 25:
                self.interventions = self.interventions[-25:]

    def record_clarified_term(self, term: str, meaning: str = ""):
        """Store a term resolved via RESEARCHER/CLARIFIER so it isn't re-asked."""
        term = (term or "").strip()
        if not term:
            return
        if term.lower() not in [t["term"].lower() for t in self.clarified_terms]:
            self.clarified_terms.append({"term": term, "meaning": (meaning or "")[:160]})

    def increment_turn(self):
        self.turn_count += 1

    def unlock_clarifier(self):
        """Call when a clarification has been resolved or turn is complete."""
        self.current_locked_state = None
        self.pending_clarification_phrase = None

    def lock_state(self, state: str, phrase: str = None):
        """Lock into any non-CONDUIT agent state (CLARIFIER, ADVOCATE, CULTURAL_BROKER)."""
        self.current_locked_state = state
        self.pending_clarification_phrase = phrase

    def lock_clarifier(self, phrase: str):
        """Lock into CLARIFIER state and store the triggering phrase."""
        self.lock_state("CLARIFIER", phrase)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "current_locked_state": self.current_locked_state,
            "allergies": list(self.allergies),
            "symptoms": list(self.symptoms),
            "medications": list(self.medications),
            "cultural_terms": list(self.cultural_terms),
            "clarified_terms": list(self.clarified_terms),
            "interventions": list(self.interventions),
            "pending_clarification_phrase": self.pending_clarification_phrase,
        }
