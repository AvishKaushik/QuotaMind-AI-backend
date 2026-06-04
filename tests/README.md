# Backend tests

## `check_config.py` — setup verifier

Run this **first** when setting up the backend locally. It connects to every
external service in your `.env` and reports `PASS` / `FAIL` for each, so you know
your credentials actually work before writing or running any app code.

It checks:

| Service | What it does |
|---|---|
| `.env` | Confirms the file exists and all required keys are filled (warns on odd `GEMINI_API_KEY` format) |
| MongoDB Atlas | `ping` the cluster with `MONGODB_URI` |
| Redis | `PING` → `PONG` using `REDIS_URL` |
| Dynatrace | `GET /api/v2/problems` with your API token (checks token + scopes) |
| Gemini | A minimal `generate_content` call with `GEMINI_API_KEY` |
| Agent Builder | `detect_intent` against your Dialogflow CX agent using the service account |

### How to run

```bash
cd quotamind-backend

# 1. Set up your environment (one time)
cp .env.example .env          # then fill in your own values
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Run the checker
python tests/check_config.py
```

### Reading the output

- **All `PASS`** → you're fully configured, start building.
- **`FAIL`** → the detail message tells you what's wrong (bad credential, service
  not running, missing scope, wrong URL).
- **`SKIP`** → a Python dependency isn't installed yet → run
  `pip install -r requirements.txt`.
- **`WARN`** → not fatal, but worth a look (e.g. an unusual API-key format).

### Notes for setup

- You need your **own** `.env` — secrets are never committed. Copy `.env.example`
  and fill in your personal keys (each teammate uses their own, or shares a set).
- `service-account.json` (Google Cloud) must sit in the backend root and is
  git-ignored. Each person downloads their own from the GCP service account.
- For **local Redis**, start one with: `docker run -d -p 6379:6379 redis:7`
  and use `REDIS_URL=redis://localhost:6379`.
- A brand-new Dynatrace trial has no data yet, so the problems count may be `0` —
  that's still a `PASS` (it means the token works).
