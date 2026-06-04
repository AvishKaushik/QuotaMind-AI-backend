#!/usr/bin/env python3
"""
QuotaMind AI — Configuration & connectivity checker.

Verifies that every external service referenced in `.env` is reachable with the
credentials provided. Run this right after copying `.env.example` -> `.env` and
filling in your values. It is safe to run repeatedly and makes only tiny,
read-only / minimal calls.

Usage:
    cd quotamind-backend
    pip install -r requirements.txt        # if you haven't already
    python tests/check_config.py

Exit code is 0 only if every required check passes — handy for CI.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ── Locate backend root (parent of this tests/ folder) ─────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

# ── Tiny ANSI helpers (no dependency) ──────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s
PASS = _c("92", "PASS")
FAIL = _c("91", "FAIL")
WARN = _c("93", "WARN")
SKIP = _c("90", "SKIP")

results: list[tuple[str, str]] = []  # (status, name)

def report(status: str, name: str, detail: str = "") -> None:
    line = f"  [{status}] {name}"
    if detail:
        line += f"  {_c('90', '— ' + detail)}"
    print(line)
    results.append((status, name))


# ── Load .env (minimal parser, no dependency needed) ───────────────────────────
def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_PATH.exists():
        return env
    for raw in ENV_PATH.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def require(env: dict[str, str], *keys: str) -> bool:
    """Return True if all keys are present and non-empty."""
    missing = [k for k in keys if not env.get(k)]
    if missing:
        report(FAIL, "env vars present", f"missing/empty: {', '.join(missing)}")
        return False
    return True


# ── Individual checks ──────────────────────────────────────────────────────────
def check_env_file(env: dict[str, str]) -> None:
    print("\n• Environment file")
    if not ENV_PATH.exists():
        report(FAIL, ".env exists", f"not found at {ENV_PATH} (copy .env.example -> .env)")
        return
    report(PASS, ".env exists", str(ENV_PATH.relative_to(ROOT)))
    required = [
        "MONGODB_URI", "MONGODB_DB_NAME", "REDIS_URL",
        "GEMINI_API_KEY", "GCP_PROJECT_ID", "GCP_LOCATION",
        "AGENT_BUILDER_AGENT_ID", "GOOGLE_APPLICATION_CREDENTIALS",
        "DYNATRACE_ENV_URL", "DYNATRACE_API_TOKEN",
    ]
    empty = [k for k in required if not env.get(k)]
    if empty:
        report(WARN, "all required keys filled", f"empty: {', '.join(empty)}")
    else:
        report(PASS, "all required keys filled")

    # Note: Google issues Gemini keys in multiple valid formats (e.g. 'AIza…'
    # and 'AQ.…'). We don't warn on format — the live Gemini check below is the
    # real test of whether the key works.


def check_mongodb(env: dict[str, str]) -> None:
    print("\n• MongoDB Atlas")
    if not require(env, "MONGODB_URI"):
        return
    try:
        from pymongo import MongoClient
    except ImportError:
        report(SKIP, "MongoDB connect", "pymongo not installed (pip install -r requirements.txt)")
        return
    try:
        client = MongoClient(env["MONGODB_URI"], serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db_name = env.get("MONGODB_DB_NAME", "quotamind")
        report(PASS, "MongoDB connect", f"ping ok, db '{db_name}'")
        client.close()
    except Exception as e:  # noqa: BLE001
        report(FAIL, "MongoDB connect", f"{type(e).__name__}: {str(e)[:120]}")


def check_redis(env: dict[str, str]) -> None:
    print("\n• Redis")
    if not require(env, "REDIS_URL"):
        return
    try:
        import redis
    except ImportError:
        report(SKIP, "Redis connect", "redis not installed (pip install -r requirements.txt)")
        return
    try:
        r = redis.from_url(env["REDIS_URL"], socket_connect_timeout=5)
        if r.ping():
            report(PASS, "Redis connect", "PONG")
        else:
            report(FAIL, "Redis connect", "no PONG")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "Redis connect", f"{type(e).__name__}: {str(e)[:120]}")


def check_dynatrace(env: dict[str, str]) -> None:
    print("\n• Dynatrace")
    if not require(env, "DYNATRACE_ENV_URL", "DYNATRACE_API_TOKEN"):
        return
    try:
        import httpx
    except ImportError:
        report(SKIP, "Dynatrace API", "httpx not installed (pip install -r requirements.txt)")
        return
    url = env["DYNATRACE_ENV_URL"].rstrip("/") + "/api/v2/problems"
    headers = {"Authorization": f"Api-Token {env['DYNATRACE_API_TOKEN']}"}
    try:
        resp = httpx.get(url, headers=headers, params={"pageSize": 1}, timeout=10)
        if resp.status_code == 200:
            total = resp.json().get("totalCount", "?")
            report(PASS, "Dynatrace API", f"200 OK (problems totalCount={total})")
        elif resp.status_code in (401, 403):
            report(FAIL, "Dynatrace API",
                   f"{resp.status_code} — token invalid or missing scopes (need problems.read)")
        else:
            report(FAIL, "Dynatrace API", f"HTTP {resp.status_code}: {resp.text[:100]}")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "Dynatrace API", f"{type(e).__name__}: {str(e)[:120]} (check DYNATRACE_ENV_URL)")


def check_gemini(env: dict[str, str]) -> None:
    print("\n• Gemini (Google AI Studio)")
    if not require(env, "GEMINI_API_KEY"):
        return
    try:
        import google.generativeai as genai
    except ImportError:
        report(SKIP, "Gemini generate", "google-generativeai not installed")
        return
    try:
        genai.configure(api_key=env["GEMINI_API_KEY"])
        model_name = env.get("GEMINI_FAST_MODEL", "gemini-2.5-flash")
        model = genai.GenerativeModel(model_name)
        resp = model.generate_content("Reply with the single word: ok")
        text = (resp.text or "").strip()
        report(PASS, "Gemini generate", f"{model_name} replied: '{text[:30]}'")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "Gemini generate", f"{type(e).__name__}: {str(e)[:140]}")


def check_agent_builder(env: dict[str, str]) -> None:
    print("\n• Google Cloud Agent Builder (Dialogflow CX)")
    if not require(env, "GCP_PROJECT_ID", "GCP_LOCATION",
                   "AGENT_BUILDER_AGENT_ID", "GOOGLE_APPLICATION_CREDENTIALS"):
        return

    # Resolve credential path relative to backend root if not absolute
    cred = env["GOOGLE_APPLICATION_CREDENTIALS"]
    cred_path = Path(cred)
    if not cred_path.is_absolute():
        cred_path = (ROOT / cred).resolve()
    if not cred_path.exists():
        report(FAIL, "service account file", f"not found: {cred_path}")
        return
    report(PASS, "service account file", str(cred_path.name))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_path)

    try:
        from google.cloud import dialogflowcx_v3 as dialogflow
    except ImportError:
        report(SKIP, "Agent detect_intent", "google-cloud-dialogflow-cx not installed")
        return

    project = env["GCP_PROJECT_ID"]
    location = env["GCP_LOCATION"]
    agent_id = env["AGENT_BUILDER_AGENT_ID"]
    try:
        # Regional endpoint required for non-global locations
        client_opts = None
        if location and location != "global":
            client_opts = {"api_endpoint": f"{location}-dialogflow.googleapis.com"}
        client = dialogflow.SessionsClient(client_options=client_opts)
        session_path = client.session_path(project, location, agent_id, "config-check-session")
        text_input = dialogflow.TextInput(text="ping")
        query_input = dialogflow.QueryInput(text=text_input, language_code="en")
        client.detect_intent(request={"session": session_path, "query_input": query_input})
        report(PASS, "Agent detect_intent", "auth + agent reachable")
    except Exception as e:  # noqa: BLE001
        report(FAIL, "Agent detect_intent",
               f"{type(e).__name__}: {str(e)[:160]}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 64)
    print(" QuotaMind AI — configuration & connectivity check")
    print("=" * 64)

    env = load_env()
    check_env_file(env)
    check_mongodb(env)
    check_redis(env)
    check_dynatrace(env)
    check_gemini(env)
    check_agent_builder(env)

    # Summary
    n_pass = sum(1 for s, _ in results if s == PASS)
    n_fail = sum(1 for s, _ in results if s == FAIL)
    n_warn = sum(1 for s, _ in results if s == WARN)
    n_skip = sum(1 for s, _ in results if s == SKIP)
    print("\n" + "=" * 64)
    print(f" Summary: {n_pass} pass · {n_fail} fail · {n_warn} warn · {n_skip} skip")
    print("=" * 64)

    if n_fail:
        print(" Some checks FAILED — see details above before running the app.")
    elif n_skip:
        print(" All reachable checks passed. Install deps to run the skipped ones:")
        print("   pip install -r requirements.txt")
    else:
        print(" 🎉 Everything is configured and reachable. You're good to go!")

    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
