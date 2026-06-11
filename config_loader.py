"""
ZGA Config Loader — Bulletproof Dynamic Configuration
=====================================================
Loads config.json and deep-merges it over hard-coded DEFAULTS, so any missing,
malformed, or out-of-range field silently falls back to a safe value.

DESIGN PRINCIPLE — YOU CANNOT BREAK THE AGENT BY EDITING config.json:
- File missing / invalid JSON   → full DEFAULTS are used.
- A field missing               → that field's default is used.
- A value out of sane range     → it is clamped to the nearest valid value.
- An unknown field              → ignored.

Hot-ish reload: get_config() re-reads the file when its mtime changes, so the
frontend gets fresh VAD/timeout values on every reconnect without a code change.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("ZeroGravity.Config")

CONFIG_PATH = Path(__file__).parent / "config.json"

# ── Safe defaults (the source of truth if config.json is absent/broken) ──────
DEFAULTS: dict = {
    "vad": {
        "silence_ms": 1500,
        "onset_ms": 180,
        "abs_floor": 0.012,
        "noise_mult": 2.8,
        "min_utter_ms": 250,
        "max_utterance_ms": 30000,
    },
    "timeouts": {
        "interpret_timeout_s": 5,
        "max_reconnects": 5,
        "reconnect_backoff_cap_s": 16,
        "session_warn_seconds": 870,
    },
    "recovery": {
        "notify_on_mute": True,
        "long_input_warning": True,
    },
    "interpretation": {
        "max_retries": 1,
        "incoherence_min_words": 6,
        "incoherence_max_stopword_ratio": 0.08,
    },
    "triggers": {
        "clarifier": ["interpreter speaking", "habla el intérprete"],
        "cultural_broker": ["interpreter note", "nota del intérprete"],
        "researcher": ["let me verify", "knowledge base"],
        "advocate": ["interpreter pause", "pausa del intérprete"],
    },
    "nonsense_handling": "hybrid",
    "vertex": {
        "service_account_file": "",
        "project": "singularityos-web-app",
        "location": "us-central1",
        "corpus_display_name": "zero-gravity-knowledge-base",
        "top_k": 3,
    },
}

# Valid numeric ranges → values outside are clamped (never crash).
_RANGES = {
    ("vad", "silence_ms"):           (300, 5000),
    ("vad", "onset_ms"):             (40, 1000),
    ("vad", "abs_floor"):            (0.001, 0.2),
    ("vad", "noise_mult"):           (1.2, 10.0),
    ("vad", "min_utter_ms"):         (50, 2000),
    ("vad", "max_utterance_ms"):     (4000, 40000),
    ("timeouts", "interpret_timeout_s"):     (5, 120),
    ("timeouts", "max_reconnects"):          (1, 20),
    ("timeouts", "reconnect_backoff_cap_s"): (2, 60),
    ("timeouts", "session_warn_seconds"):    (60, 1800),
    ("interpretation", "max_retries"):                 (0, 3),
    ("interpretation", "incoherence_min_words"):       (3, 30),
    ("interpretation", "incoherence_max_stopword_ratio"): (0.0, 0.5),
}

_NONSENSE_OPTIONS = {"hybrid", "clarify_first", "translate_and_flag"}

# ── Internal cache (mtime-based) ─────────────────────────────────────────────
_cache: dict | None = None
_cache_mtime: float = -1.0


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override onto a copy of base. Lists/scalars replace."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k.startswith("_"):  # skip _comment keys
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _clamp(cfg: dict) -> dict:
    """Clamp numeric fields to valid ranges; coerce bad types to defaults."""
    for (section, key), (lo, hi) in _RANGES.items():
        try:
            val = cfg[section][key]
            num = float(val)
            if num < lo:
                num = lo
            elif num > hi:
                num = hi
            # preserve int-ness for *_ms / counts
            cfg[section][key] = int(num) if float(DEFAULTS[section][key]).is_integer() else num
        except (KeyError, TypeError, ValueError):
            cfg.setdefault(section, {})[key] = DEFAULTS[section][key]

    # nonsense_handling must be one of the known modes
    if cfg.get("nonsense_handling") not in _NONSENSE_OPTIONS:
        cfg["nonsense_handling"] = DEFAULTS["nonsense_handling"]

    # triggers must be dict[str, list[str]]
    trg = cfg.get("triggers")
    if not isinstance(trg, dict):
        cfg["triggers"] = dict(DEFAULTS["triggers"])
    else:
        for role in DEFAULTS["triggers"]:
            phrases = trg.get(role)
            if not isinstance(phrases, list) or not all(isinstance(p, str) for p in phrases):
                trg[role] = list(DEFAULTS["triggers"][role])
    return cfg


def get_config(force: bool = False) -> dict:
    """
    Return the merged, validated config. Re-reads config.json when its mtime
    changes (or on force=True). Never raises.
    """
    global _cache, _cache_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else -1.0
    except OSError:
        mtime = -1.0

    if not force and _cache is not None and mtime == _cache_mtime:
        return _cache

    user_cfg: dict = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if not isinstance(user_cfg, dict):
                raise ValueError("config.json root is not an object")
        except Exception as e:
            logger.warning("⚠️ config.json invalid (%s) — using safe defaults", e)
            user_cfg = {}
    else:
        logger.info("config.json not found — using built-in defaults")

    merged = _clamp(_deep_merge(DEFAULTS, user_cfg))
    _cache = merged
    _cache_mtime = mtime
    return merged


def get_section(name: str) -> dict:
    """Convenience accessor for a top-level section (always a dict)."""
    val = get_config().get(name, {})
    return val if isinstance(val, dict) else {}


def save_config(updates: dict) -> dict:
    """
    Deep-merge `updates` into config.json on disk and return the freshly
    validated/clamped config. Used by the Settings panel to tune live.
    Never raises — on write failure it logs and returns the current config.
    """
    # Start from whatever is on disk (raw), so we preserve _comment keys.
    raw: dict = {}
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raw = {}
        except Exception:
            raw = {}

    merged_raw = _deep_merge(raw if raw else DEFAULTS, updates or {})

    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(merged_raw, f, indent=2, ensure_ascii=False)
        logger.info("⚙️ config.json actualizado desde la UI")
    except Exception as e:
        logger.error("No se pudo escribir config.json: %s", e)

    # Force a clean re-read (also re-clamps anything out of range)
    return get_config(force=True)


if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO)
    pprint.pprint(get_config(force=True))
