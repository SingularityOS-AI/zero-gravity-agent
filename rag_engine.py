"""
ZGA RAG Engine — Vertex AI Knowledge Base Connector
====================================================
Powers the RESEARCHER state: when the agent is unsure of a term/idiom,
it queries the `zero-gravity-knowledge-base` corpus in Vertex AI live.

DESIGN PRINCIPLE — GRACEFUL DEGRADATION:
- If the service account JSON is missing, or Vertex is unreachable, or the
  SDK is not installed, the engine reports `available=False` and every query
  returns {found: False}. The agent then interprets normally (no crash).
- The first successful init resolves the corpus display-name → resource name
  and caches it, so subsequent queries are fast.

SETUP (one time):
1. GCP Console → IAM → Service Accounts → create `zero-gravity-rag`
2. Grant role: Vertex AI User  (roles/aiplatform.user)
3. Create JSON key → save as ./service-account.json  (next to this file)
   (or set GOOGLE_APPLICATION_CREDENTIALS to its path)
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("ZeroGravity.RAG")

try:
    from config_loader import get_section
    _vx = get_section("vertex")
except Exception:
    _vx = {}

# ── Configuration (config.json → env → defaults) ────────────────────────────
RAG_PROJECT        = os.environ.get("VERTEX_PROJECT", _vx.get("project", "singularityos-web-app"))
RAG_LOCATION       = os.environ.get("VERTEX_LOCATION", _vx.get("location", "us-central1"))
RAG_CORPUS_DISPLAY = os.environ.get("VERTEX_RAG_CORPUS_DISPLAY", _vx.get("corpus_display_name", "zero-gravity-knowledge-base"))
RAG_CORPUS_NAME    = os.environ.get("VERTEX_RAG_CORPUS", "")  # projects/.../ragCorpora/123
RAG_TOP_K          = int(os.environ.get("VERTEX_RAG_TOP_K", _vx.get("top_k", 3)))

_HERE = Path(__file__).parent


def _looks_like_service_account(path: str) -> bool:
    """True if the JSON file is a GCP service-account key."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("type") == "service_account" and "private_key" in data
    except Exception:
        return False


def _resolve_service_account() -> str:
    """
    Find the service-account JSON, in priority order:
      1. GOOGLE_APPLICATION_CREDENTIALS env var
      2. config.json → vertex.service_account_file (filename in this folder)
      3. ./service-account.json (the documented placeholder)
      4. AUTO-DETECT: any *.json in this folder that is a service_account key
    """
    env = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
    if env and os.path.exists(env):
        return env

    cfg_name = (_vx.get("service_account_file") or "").strip()
    if cfg_name:
        p = _HERE / cfg_name
        if p.exists():
            return str(p)

    placeholder = _HERE / "service-account.json"
    if placeholder.exists():
        return str(placeholder)

    # Auto-detect any service-account JSON dropped in the folder
    for jf in sorted(glob.glob(str(_HERE / "*.json"))):
        if _looks_like_service_account(jf):
            return jf

    return str(placeholder)  # non-existent → triggers graceful "not found"


SERVICE_ACCOUNT_PATH = _resolve_service_account()

# ── Lazy global state ──────────────────────────────────────────────────────
_initialized: bool = False
_available: bool = False
_init_error: str = ""
_rag = None                 # the vertexai.rag module once imported
_corpus_resource: str = ""  # resolved corpus resource name


def _try_init() -> None:
    """Initialize Vertex + resolve the corpus. Idempotent. Never raises."""
    global _initialized, _available, _init_error, _rag, _corpus_resource
    if _initialized:
        return
    _initialized = True

    # 1) Credentials: prefer a local key file (dev machine); otherwise fall back
    #    to Application Default Credentials (ADC) — the service account attached
    #    to the Cloud Run service in production. This makes the SAME code work
    #    locally (with key) and deployed (no key file, ADC from runtime SA).
    if os.path.exists(SERVICE_ACCOUNT_PATH):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_PATH
        logger.info("RAG: usando service account local (%s)", os.path.basename(SERVICE_ACCOUNT_PATH))
    else:
        logger.info("RAG: sin key file local — usando ADC (Cloud Run / gcloud auth)")

    # 2) Import SDK (new API first, fall back to preview)
    try:
        import vertexai
        try:
            from vertexai import rag as rag_mod          # newer SDK
        except Exception:
            from vertexai.preview import rag as rag_mod  # older SDK
        vertexai.init(project=RAG_PROJECT, location=RAG_LOCATION)
        _rag = rag_mod
    except Exception as e:  # SDK missing / init failed
        _init_error = f"Vertex SDK init failed: {e}"
        logger.warning("⚠️ RAG: %s", _init_error)
        return

    # 3) Resolve corpus resource name
    try:
        if RAG_CORPUS_NAME:
            _corpus_resource = RAG_CORPUS_NAME
        else:
            for c in _rag.list_corpora():
                if getattr(c, "display_name", "") == RAG_CORPUS_DISPLAY:
                    _corpus_resource = c.name
                    break
        if not _corpus_resource:
            _init_error = f"Corpus '{RAG_CORPUS_DISPLAY}' not found in {RAG_PROJECT}/{RAG_LOCATION}"
            logger.warning("⚠️ RAG: %s", _init_error)
            return
    except Exception as e:
        _init_error = f"Corpus lookup failed: {e}"
        logger.warning("⚠️ RAG: %s", _init_error)
        return

    _available = True
    logger.info("✅ RAG online — corpus: %s", _corpus_resource)


def is_available() -> bool:
    """True if Vertex RAG is ready to serve queries."""
    _try_init()
    return _available


def status() -> dict:
    """Diagnostic snapshot for the UI / logs."""
    _try_init()
    return {
        "available": _available,
        "corpus": _corpus_resource or RAG_CORPUS_DISPLAY,
        "project": RAG_PROJECT,
        "location": RAG_LOCATION,
        "error": _init_error,
        "sa_path": SERVICE_ACCOUNT_PATH,
    }


def query(term: str) -> dict:
    """
    Look up `term` in the knowledge base.

    Returns:
      {found: bool, context: str, source: str, term: str, error: str}

    Never raises — on any failure returns found=False so the agent can
    fall back to normal interpretation.
    """
    term = (term or "").strip()
    out = {"found": False, "context": "", "source": "", "term": term, "error": ""}
    if not term:
        out["error"] = "empty term"
        return out

    _try_init()
    if not _available:
        out["error"] = _init_error or "RAG unavailable"
        return out

    try:
        rag_resource = _rag.RagResource(rag_corpus=_corpus_resource)

        # The retrieval API signature shifted across SDK versions — try the
        # config-object form first, then the legacy keyword form.
        resp = None
        try:
            cfg = _rag.RagRetrievalConfig(top_k=RAG_TOP_K)
            resp = _rag.retrieval_query(rag_resources=[rag_resource], text=term, rag_retrieval_config=cfg)
        except Exception:
            resp = _rag.retrieval_query(rag_resources=[rag_resource], text=term, similarity_top_k=RAG_TOP_K)

        # Parse contexts (shape: resp.contexts.contexts[].text / .source_*)
        snippets: list[str] = []
        source = ""
        contexts = getattr(getattr(resp, "contexts", None), "contexts", None) or []
        for ctx in contexts:
            txt = (getattr(ctx, "text", "") or "").strip()
            if txt:
                snippets.append(txt)
            if not source:
                source = (getattr(ctx, "source_display_name", "")
                          or getattr(ctx, "source_uri", "") or "")

        if snippets:
            # Keep it tight so the injected prompt stays small
            joined = " … ".join(snippets)
            out["found"] = True
            out["context"] = joined[:1200]
            out["source"] = source or RAG_CORPUS_DISPLAY
        else:
            out["error"] = "no matching context"
        return out

    except Exception as e:
        logger.error("RAG query error for '%s': %s", term, e)
        out["error"] = str(e)
        return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("RAG status:", status())
    if is_available():
        for t in ["susto", "empacho", "mal de ojo"]:
            r = query(t)
            print(f"\n[{t}] found={r['found']} source={r['source']}")
            print((r["context"] or r["error"])[:300])
