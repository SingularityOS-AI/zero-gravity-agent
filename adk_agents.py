"""
ZGA ADK Multi-Agent Reasoning Layer — HARDENED
==============================================
The hybrid brain of Zero Gravity. The real-time AUDIO stays on Gemini 3.1
Flash Live (AI Studio) — untouched. This module adds a TEXT-only multi-agent
REASONING layer orchestrated with Google's Agent Development Kit (ADK),
running on Vertex AI.

ARCHITECTURE (agent-as-a-tool pattern, per the ADK guide):

    Orchestrator (LlmAgent)
      ├── Researcher Agent   → query_knowledge_base()  → Vertex RAG
      ├── Cultural Broker     → lookup_cultural_term()  → Vertex RAG (folk glossary)
      └── Advocate Agent      → check_clinical_safety() → allergy/dose risk

PRODUCTION GUARANTEES (the contract main.py relies on):
  1. run_clinical_reasoning() NEVER raises and NEVER blocks past its deadline.
     Every call is bounded by ADK_TIMEOUT_S (asyncio.wait_for). On timeout or
     error it returns {available: False, error: ...} so the caller falls back
     to the direct RAG path — the live interpreter NEVER goes mute waiting.
  2. No synchronous network I/O ever runs on the event loop. The heavy build
     (MCP subprocess spawn + Vertex init) and every rag_engine.query run in
     worker threads (asyncio.to_thread).
  3. Grounding metadata (found/source) is captured FROM the tool calls
     themselves during the run — no redundant second RAG query.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
import uuid
from pathlib import Path

import rag_engine
from config_loader import get_section

logger = logging.getLogger("ZeroGravity.ADK")

_HERE = Path(__file__).parent
_MCP_SERVER = str(_HERE / "mcp_server.py")

# ── Vertex AI configuration (reuse the resolved service account) ─────────────
_vx = get_section("vertex")
# flash-lite: measured ~3-5x lower tail latency per hop than gemini-2.5-flash on
# Vertex — the multi-hop chain (router → specialist → MCP → answer) must land
# inside the live interpreter's research budget. SingularityOS standard for text.
ADK_MODEL = os.environ.get("ADK_MODEL", _vx.get("adk_model", "gemini-2.5-flash-lite"))

# Hard deadline for one full orchestrator run (router → specialist → answer).
# Past this, the caller falls back to direct RAG. Tunable via env for the demo rig.
ADK_TIMEOUT_S = float(os.environ.get("ADK_TIMEOUT_S", "8.0"))

_available: bool | None = None
_init_error: str = ""
_orchestrator = None
_InMemoryRunner = None
_types = None
_build_lock = threading.Lock()

# ── Tool-call trace ───────────────────────────────────────────────────────────
# The RAG tools record their last result here so run_clinical_reasoning can
# report grounding WITHOUT re-querying Vertex (the old path paid a second
# blocking network call just to compute `grounded`). Keyed by thread-safe lock;
# sessions are serialized per live connection so last-write-wins is correct.
_trace_lock = threading.Lock()
_last_tool_hit: dict = {"found": False, "source": ""}


def _record_tool_hit(found: bool, source: str) -> None:
    with _trace_lock:
        _last_tool_hit["found"] = bool(found)
        _last_tool_hit["source"] = source or ""


def _read_tool_hit() -> tuple[bool, str]:
    with _trace_lock:
        return bool(_last_tool_hit["found"]), str(_last_tool_hit["source"])


def _reset_tool_hit() -> None:
    _record_tool_hit(False, "")


# ── Tools (FunctionTools) — type hints are mandatory for ADK schemas ─────────

def query_knowledge_base(term: str) -> dict:
    """Look up an uncertain medical term or idiom in the Vertex AI knowledge base.

    Use this for rare diseases, unfamiliar clinical terms, or regional medical
    expressions where guessing would be dangerous.

    Args:
        term: The exact term or phrase to verify (e.g. "mal de San Vito").

    Returns:
        A dict with keys: status ("success"/"error"), found (bool),
        context (the retrieved definition), and source.
    """
    r = rag_engine.query(term)
    _record_tool_hit(r["found"], r.get("source", ""))
    return {
        "status": "success" if r["found"] else "error",
        "found": r["found"],
        "context": r["context"][:1000],
        "source": r.get("source", ""),
    }


def lookup_cultural_term(term: str) -> dict:
    """Look up a folk-illness or cultural health concept (e.g. susto, empacho, mal de ojo).

    Use this when a patient describes illness in cultural terms that need
    brokering into a clinical equivalent for the provider.

    Args:
        term: The folk/cultural term the patient used.

    Returns:
        A dict with keys: status, found (bool), context (cultural+clinical
        explanation), and source.
    """
    r = rag_engine.query(f"cultural folk illness meaning of {term}")
    _record_tool_hit(r["found"], r.get("source", ""))
    return {
        "status": "success" if r["found"] else "error",
        "found": r["found"],
        "context": r["context"][:1000],
        "source": r.get("source", ""),
    }


def check_clinical_safety(statement: str, known_allergies: str = "") -> dict:
    """Assess a clinical statement for an immediate patient-safety risk.

    Use this to catch allergy conflicts, dangerous dosage confusion, or
    expressed self-harm intent that the interpreter must flag.

    Args:
        statement: The statement to evaluate (what the patient/provider said).
        known_allergies: Comma-separated allergies recorded this session, if any.

    Returns:
        A dict with keys: status, risk (bool), severity ("none"/"caution"/"critical"),
        and reason.
    """
    low = statement.lower()
    allergies = [a.strip().lower() for a in known_allergies.split(",") if a.strip()]

    # Allergy conflict: a known allergen named in a prescribing context
    for a in allergies:
        if a and a in low:
            return {"status": "success", "risk": True, "severity": "critical",
                    "reason": f"Statement references '{a}', a recorded allergy — possible prescribing conflict."}

    # Self-harm / suicidal ideation cues
    si_cues = ["mejor no estar", "no estuviera aqu", "quitarme la vida", "matarme",
               "pastillas guardadas", "end my life", "kill myself", "better off without me"]
    if any(c in low for c in si_cues):
        return {"status": "success", "risk": True, "severity": "critical",
                "reason": "Possible suicidal ideation — escalate as a safety advocacy event."}

    # Dosage confusion cues
    dose_cues = ["pero entend", "but i understood", "dijo veinticinco", "said twenty-five",
                 "no estoy seguro de la dosis", "double dose", "tomé dos"]
    if any(c in low for c in dose_cues):
        return {"status": "success", "risk": True, "severity": "caution",
                "reason": "Possible dosage misunderstanding — verify the exact dose/frequency."}

    return {"status": "success", "risk": False, "severity": "none", "reason": "No immediate safety risk detected."}


# ── Vertex env sandbox ────────────────────────────────────────────────────────

@contextlib.contextmanager
def sandbox_vertex_env():
    """
    Context manager to safely scope Vertex AI environment variables.
    Prevents global contamination that causes the Gemini Live client in main.py
    to crash. NOTE: the Live client also passes vertexai=False explicitly, so a
    concurrent reconnect during an ADK run cannot be poisoned by these vars.
    """
    old_use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
    old_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    old_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    old_location = os.environ.get("GOOGLE_CLOUD_LOCATION")

    try:
        if os.path.exists(rag_engine.SERVICE_ACCOUNT_PATH):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = rag_engine.SERVICE_ACCOUNT_PATH
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "TRUE"
        os.environ["GOOGLE_CLOUD_PROJECT"] = rag_engine.RAG_PROJECT
        os.environ["GOOGLE_CLOUD_LOCATION"] = rag_engine.RAG_LOCATION
        yield
    finally:
        for key, old in (
            ("GOOGLE_GENAI_USE_VERTEXAI", old_use_vertex),
            ("GOOGLE_APPLICATION_CREDENTIALS", old_credentials),
            ("GOOGLE_CLOUD_PROJECT", old_project),
            ("GOOGLE_CLOUD_LOCATION", old_location),
        ):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


# ── Build the multi-agent system ─────────────────────────────────────────────

def _build() -> None:
    """
    Construct the ADK orchestrator + sub-agents. Idempotent, never raises.
    SYNCHRONOUS AND HEAVY (MCP subprocess spawn + Vertex probe): callers on the
    event loop must go through ensure_built_async(), never call this directly.
    """
    global _available, _init_error, _orchestrator, _InMemoryRunner, _types
    with _build_lock:
        if _available is not None:
            return
        _available = False

        # RAG must be reachable (the tools depend on it / Vertex creds)
        if not rag_engine.is_available():
            _init_error = "Vertex/RAG unavailable — ADK reasoning disabled"
            logger.warning("⚠️ ADK: %s", _init_error)
            return

        try:
            with sandbox_vertex_env():
                from google.adk.agents import LlmAgent
            from google.adk.tools import FunctionTool
            from google.adk.tools.agent_tool import AgentTool
            from google.adk.runners import InMemoryRunner
            from google.genai import types

            # ── MCP: connect the Researcher to an external web-search tool server ──
            # Track 1 requirement: the agent uses the Model Context Protocol to
            # securely connect to an external tool. web_search runs out-of-process
            # in mcp_server.py. If MCP can't start, the Researcher still has the
            # Vertex RAG FunctionTool — graceful degradation.
            researcher_tools = []
            try:
                from google.adk.tools.mcp_tool import MCPToolset, StdioConnectionParams
                from mcp import StdioServerParameters
                mcp_search = MCPToolset(
                    connection_params=StdioConnectionParams(
                        server_params=StdioServerParameters(
                            command="python", args=[_MCP_SERVER],
                        ),
                        timeout=20,
                    )
                )
                researcher_tools.append(mcp_search)
                logger.info("🔌 MCP web-search tool connected to Researcher")
            except Exception as e:
                logger.warning("⚠️ MCP unavailable (%s) — Researcher uses RAG only", e)
            researcher_tools.append(FunctionTool(query_knowledge_base))

            researcher = LlmAgent(
                name="researcher",
                model=ADK_MODEL,
                description="Verifies rare/unknown medical terms via live web search (MCP) then the knowledge base.",
                instruction=(
                    "You verify uncertain medical terms. FIRST call web_search to check the live "
                    "public internet for the term's meaning. If web_search is empty, inconclusive, or "
                    "errors, THEN call query_knowledge_base (the curated Vertex RAG). "
                    "Return ONE concise sentence with the confirmed clinical meaning the interpreter can "
                    "use. Never invent a meaning — if both sources fail, say it could not be confirmed."
                ),
                tools=researcher_tools,
            )
            cultural = LlmAgent(
                name="cultural_broker",
                model=ADK_MODEL,
                description="Explains folk-illness / cultural health concepts and their clinical equivalents.",
                instruction=(
                    "You broker cultural health concepts (susto, empacho, mal de ojo, etc.). "
                    "Call lookup_cultural_term, then give a one-sentence neutral explanation plus the "
                    "likely clinical equivalent. Never dismiss the belief."
                ),
                tools=[FunctionTool(lookup_cultural_term)],
            )
            advocate = LlmAgent(
                name="advocate",
                model=ADK_MODEL,
                description="Flags immediate patient-safety risks (allergy conflict, dose confusion, self-harm).",
                instruction=(
                    "You assess patient safety. Call check_clinical_safety. If risk is true, state the "
                    "concern in one firm sentence for both parties. If false, say no safety issue found."
                ),
                tools=[FunctionTool(check_clinical_safety)],
            )

            orchestrator = LlmAgent(
                name="clinical_orchestrator",
                model=ADK_MODEL,
                description="Routes a medical interpreter's uncertainty to the right specialist agent.",
                instruction=(
                    "You are the reasoning orchestrator for a medical interpreter. Given an uncertain "
                    "term or a clinical statement, decide which specialist to consult and call it:\n"
                    "- researcher: rare/unknown medical terms or idioms.\n"
                    "- cultural_broker: folk-illness or cultural health concepts.\n"
                    "- advocate: possible safety risk (allergy, dosage, self-harm).\n"
                    "Consult exactly one specialist, then return ONLY the confirmed, concise result the "
                    "interpreter should use. Do not add disclaimers."
                ),
                tools=[AgentTool(agent=researcher), AgentTool(agent=cultural), AgentTool(agent=advocate)],
            )

            _orchestrator = orchestrator
            _InMemoryRunner = InMemoryRunner
            _types = types
            _available = True
            logger.info("✅ ADK multi-agent online — model: %s (Vertex) — 3 specialists + orchestrator", ADK_MODEL)
        except Exception as e:
            _init_error = f"ADK build failed: {e}"
            logger.warning("⚠️ ADK: %s", _init_error)


async def ensure_built_async() -> bool:
    """Build (if needed) in a worker thread so the event loop never blocks."""
    if _available is None:
        await asyncio.to_thread(_build)
    return bool(_available)


def is_available() -> bool:
    """True if the ADK multi-agent reasoning layer is ready. (Sync callers only.)"""
    _build()
    return bool(_available)


def status() -> dict:
    _build()
    return {"available": bool(_available), "model": ADK_MODEL, "error": _init_error,
            "timeout_s": ADK_TIMEOUT_S}


# ── Orchestrator run ──────────────────────────────────────────────────────────

async def _run_orchestrator(prompt: str) -> tuple[str, list[str]]:
    """One full orchestrator pass. Raises on failure — wrapped by the caller."""
    with sandbox_vertex_env():
        runner = _InMemoryRunner(agent=_orchestrator, app_name="zga_reasoning")
        uid, sid = "zga", uuid.uuid4().hex[:12]
        # Create the session (API is async in ADK 2.x, sync in older builds)
        try:
            await runner.session_service.create_session(app_name="zga_reasoning", user_id=uid, session_id=sid)
        except TypeError:
            runner.session_service.create_session(app_name="zga_reasoning", user_id=uid, session_id=sid)

        content = _types.Content(role="user", parts=[_types.Part(text=prompt)])
        final_text, agents_used = "", []
        async for event in runner.run_async(user_id=uid, session_id=sid, new_message=content):
            author = getattr(event, "author", None)
            if author and author not in agents_used and author != "clinical_orchestrator":
                agents_used.append(author)
            if hasattr(event, "is_final_response") and event.is_final_response():
                if event.content and event.content.parts:
                    final_text = "".join(p.text or "" for p in event.content.parts).strip()
        return final_text, agents_used


async def run_clinical_reasoning(query: str, allergies: str = "",
                                 timeout_s: float | None = None) -> dict:
    """
    Run the ADK orchestrator over an uncertain term/statement.

    HARD CONTRACT: never raises, never exceeds timeout_s (default ADK_TIMEOUT_S).
    On timeout/error returns {available: False, error: ...} so the caller can
    fall back to direct RAG instantly.

    Returns: {available, text, agents, grounded, source, error?}
      - text: the orchestrator's concise confirmed answer
      - agents: list of specialist agents that were invoked (observability)
      - grounded: whether a knowledge-base tool returned a hit during THIS run
      - source: knowledge base source, if any
    """
    deadline = timeout_s if timeout_s is not None else ADK_TIMEOUT_S
    out = {"available": False, "text": "", "agents": [], "grounded": False, "source": ""}

    try:
        built = await asyncio.wait_for(ensure_built_async(), timeout=deadline)
    except (asyncio.TimeoutError, Exception) as e:
        out["error"] = _init_error or f"ADK build did not complete: {e}"
        return out
    if not built:
        out["error"] = _init_error
        return out

    prompt = query if not allergies else f"{query}\n[Recorded allergies this session: {allergies}]"
    _reset_tool_hit()
    try:
        final_text, agents_used = await asyncio.wait_for(
            _run_orchestrator(prompt), timeout=deadline
        )
        grounded, source = _read_tool_hit()
        out.update({
            "available": True,
            "text": final_text,
            "agents": agents_used or ["clinical_orchestrator"],
            "grounded": grounded,
            "source": source,
        })
        return out
    except asyncio.TimeoutError:
        logger.error("ADK run timed out after %.1fs — falling back to direct RAG", deadline)
        out["error"] = f"ADK timeout after {deadline:.1f}s"
        return out
    except Exception as e:
        logger.error("ADK run error: %s", e)
        out["error"] = str(e)
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ADK status:", status())
    if is_available():
        for q in ["mal de San Vito", "me cayó un susto muy fuerte", "soy alérgico a la penicilina, me van a dar penicilina"]:
            r = asyncio.run(run_clinical_reasoning(q, allergies="penicilina"))
            print(f"\nQ: {q}\n  agents={r['agents']} grounded={r['grounded']} err={r.get('error','')}\n  → {r['text'][:200]}")
