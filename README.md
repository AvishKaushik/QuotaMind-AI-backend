# QuotaMind AI — Backend

> Autonomous AI Quota & Inference Optimization Agent — FastAPI service.

This is the backend for **QuotaMind AI**, built for the Google Cloud Rapid Agent
Hackathon (Dynatrace partner track). It ingests AI workload traffic, runs a suite
of optimization engines, and drives an autonomous agent (Google Cloud Agent Builder
+ Gemini 2.5) that reroutes, compresses, throttles, and forecasts — all streamed to
the dashboard in real time over SSE.

> ⚠️ **Status:** This folder currently contains only the **project skeleton** —
> the directory layout, dependency manifest, env template, and Dockerfile. All
> Python files are empty placeholders. The sections below describe exactly **what
> each file must implement**.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI (Python 3.12), Uvicorn |
| Validation | Pydantic v2 |
| Database | MongoDB Atlas (async via Motor) |
| Cache / event bus | Redis (counters + pub/sub) |
| AI core | Gemini 2.5 Pro / Flash (`google-generativeai`) |
| Agent orchestration | Google Cloud Agent Builder (Vertex AI / Dialogflow CX) |
| Observability MCP | Dynatrace MCP Server (`httpx` / `mcp`) |
| Real-time | Server-Sent Events (SSE) over Redis pub/sub |
| Deploy | Google Cloud Run (Docker) |

---

## Project Structure

```
quotamind-backend/
├── main.py                       # FastAPI app: CORS, router registration, /health, startup tasks
│
├── api/                          # HTTP routers (one file per concern)
│   ├── router_requests.py        # POST /api/requests/ingest, GET /api/requests/summary, GET /api/requests/live
│   ├── router_metrics.py         # GET /api/metrics/summary, /api/metrics/cache-stats, /api/metrics/savings
│   ├── router_agent.py           # GET /api/agents/status, /api/agents/{id}/health, POST /api/agents/{id}/pause, GET /api/forecast, GET /api/recommendations, GET /api/budget/status, POST /api/config/budget
│   ├── router_events.py          # GET /api/events/stream (SSE bridge to the frontend)
│   └── router_simulator.py       # POST /api/simulator/{spike,runaway,budget-overflow,dynatrace-anomaly,reset}
│
├── agent/                        # Autonomous agent layer (the "brain")
│   ├── agent_runner.py           # Wraps Agent Builder client; sends context, parses tool calls into AgentDecision
│   └── orchestrator.py           # AgentOrchestrator.run_cycle() — 30s loop wiring all engines + agent together
│
├── engines/                      # Core optimization engines (pure logic)
│   ├── prompt_compressor.py      # PromptCompressor — Gemini Flash prompt compression + savings logging
│   ├── duplicate_detector.py     # DuplicateDetector — SHA-256 hashing, Redis/Mongo cache, (optional) semantic dedupe
│   ├── traffic_router.py         # TrafficRouter — model-tier routing by quota/cost/priority
│   ├── runaway_detector.py       # RunawayDetector — rate/recursion/token-spike detection + throttle/pause
│   ├── budget_allocator.py       # BudgetAllocator — per-category spend tracking + enforcement
│   ├── quota_forecaster.py       # QuotaForecaster — burn-rate projection + exhaustion ETA
│   └── recommendations_engine.py # RecommendationsEngine — Gemini-generated daily optimization insights
│
├── integrations/                 # External service clients
│   ├── db.py                     # Async Motor client + collection getters + index setup
│   ├── cache.py                  # Redis client + helpers: get/set_cached, increment_counter, publish/subscribe
│   ├── gemini_client.py          # GeminiClient — reason / compress_prompt / classify_priority / generate_recommendation
│   └── dynatrace_mcp.py          # DynatraceMCPClient — get_metrics / get_active_problems / push_optimization_event
│
├── models/                       # Pydantic schemas (request bodies + DB documents)
│   ├── ai_request.py             # AIRequest — prompt/model/tokens/latency/cost/priority/agent_id
│   ├── optimization_event.py     # OptimizationEvent — event_type/trigger/action/savings_usd/timestamp
│   ├── agent_log.py              # AgentLog — decision/reasoning/anomaly_type/severity/timestamp
│   └── cached_output.py          # CachedOutput — request_hash/response/hit_count/created_at
│
├── config/                       # Static configuration
│   ├── settings.py               # Pydantic Settings — loads env vars (URIs, keys, limits, intervals)
│   └── models.py                 # MODEL_TIERS, COST_PER_1K_TOKENS, PRIORITY_TO_MIN_TIER pricing/tier maps
│
├── simulator/                    # AI workload traffic generator (the demo "traffic source")
│   └── workload_simulator.py     # Mimics 4 agents POSTing to /api/requests/ingest; drives crisis scenarios
│
├── requirements.txt              # Pinned Python dependencies
├── .env.example                  # All required env var names (no secrets) — copy to .env
├── .gitignore                    # Python / env / secret ignores
├── Dockerfile                    # Cloud Run container image (python:3.12-slim, uvicorn on :8080)
└── README.md                     # This file
```

---

## Files To Build (Implementation Checklist)

### Bootstrap
- [ ] **`main.py`** — create the `FastAPI` app, add CORS middleware (origins from `config.settings`),
  register all five routers from `api/`, expose `GET /health` → `{"status": "ok"}`, and register the
  `AgentOrchestrator` loop as a startup background task.

### `config/`
- [ ] **`settings.py`** — `Settings(BaseSettings)` reading every var in `.env.example`.
- [ ] **`models.py`** — `MODEL_TIERS`, `COST_PER_1K_TOKENS`, and `PRIORITY_TO_MIN_TIER` dicts.

### `models/` (Pydantic v2)
- [ ] **`ai_request.py`** → `AIRequest`
- [ ] **`optimization_event.py`** → `OptimizationEvent`
- [ ] **`agent_log.py`** → `AgentLog`
- [ ] **`cached_output.py`** → `CachedOutput`

### `integrations/`
- [ ] **`db.py`** — async Motor client; getters for `ai_requests`, `optimization_events`,
  `agent_logs`, `cached_outputs`; index creation on `timestamp` / `model` / `priority`.
- [ ] **`cache.py`** — Redis client; `get_cached`, `set_cached`, `increment_counter`,
  and pub/sub helpers for the `quotamind:events` channel.
- [ ] **`gemini_client.py`** — `GeminiClient` with `reason`, `compress_prompt`,
  `classify_priority`, `generate_recommendation`; retry w/ exponential backoff; token logging.
- [ ] **`dynatrace_mcp.py`** — `DynatraceMCPClient` with `get_metrics`, `get_active_problems`,
  `get_latency_spike_events`, `push_optimization_event`.

### `engines/`
- [ ] **`prompt_compressor.py`** → `PromptCompressor.compress(...)`
- [ ] **`duplicate_detector.py`** → `DuplicateDetector` (hash, check, cache, get)
- [ ] **`traffic_router.py`** → `TrafficRouter.get_optimal_model / reroute_agent`
- [ ] **`runaway_detector.py`** → `RunawayDetector.analyze_agent`
- [ ] **`budget_allocator.py`** → `BudgetAllocator.check_budget_status / enforce_budget_limits`
- [ ] **`quota_forecaster.py`** → `QuotaForecaster.forecast_exhaustion`
- [ ] **`recommendations_engine.py`** → `RecommendationsEngine.generate_daily_report`

### `agent/`
- [ ] **`agent_runner.py`** — init Agent Builder client, run a session with context, parse tool calls
  into structured `AgentDecision` objects.
- [ ] **`orchestrator.py`** — `AgentOrchestrator.run_cycle()`: collect metrics → pull Dynatrace signals →
  run runaway/forecast/budget checks → build context → call Agent Builder → execute tool calls →
  log to Mongo → publish events to Redis. Runs every `ORCHESTRATOR_INTERVAL_SECONDS`.

### `api/`
- [ ] **`router_requests.py`** — ingestion endpoint (cost calc, dup check, Mongo write, Redis counters, publish).
- [ ] **`router_metrics.py`** — aggregated metrics, cache stats, savings.
- [ ] **`router_agent.py`** — agent status/health, forecast, recommendations, budget, manual pause/config.
- [ ] **`router_events.py`** — SSE endpoint subscribing to `quotamind:events`.
- [ ] **`router_simulator.py`** — the five demo crisis-trigger endpoints.

### `simulator/`
- [ ] **`workload_simulator.py`** — generate realistic traffic for 4 agents (support-bot-1, summarizer-1,
  analytics-agent-1, workflow-agent-1), inject ~15% duplicate hashes, and execute crisis scenarios.

---

## Data Model (MongoDB `quotamind`)

| Collection | Holds | Key indexes |
|---|---|---|
| `ai_requests` | Every AI call (prompt, model, tokens, latency, cost, priority, hash, ts) | `timestamp`, `model`, `priority` |
| `optimization_events` | Reroute / throttle / compression / cache decisions + savings | `timestamp` |
| `agent_logs` | Agent planning decisions, reasoning traces, anomaly detections | `timestamp` |
| `cached_outputs` | Duplicate request → reusable response mappings | `request_hash` |

**Redis keys:** `quota:tokens_used_today`, `quota:cost_usd_today`, `quota:current_pct`,
`quota:daily_limit`, `ratewindow:{agent_id}`, `cache:{prompt_hash}`, `cache:hits`,
`cache:misses`, `compress_next:{agent_id}` · **pub/sub channel:** `quotamind:events`.

---

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET  | `/health` | Liveness check → `{"status": "ok"}` |
| POST | `/api/requests/ingest` | Ingest one AI request (from simulator) |
| GET  | `/api/requests/summary` | Aggregated request stats |
| GET  | `/api/metrics/summary` | Tokens / cost / quota |
| GET  | `/api/metrics/cache-stats` | Cache hit rate + savings |
| GET  | `/api/forecast` | Quota exhaustion forecast |
| GET  | `/api/recommendations` | Gemini optimization insights |
| GET  | `/api/agents/status` | All agent statuses + models |
| GET  | `/api/agents/{id}/health` | Runaway risk score |
| POST | `/api/agents/{id}/pause` | Manual pause override |
| GET  | `/api/budget/status` | Budget breakdown + projections |
| POST | `/api/config/budget` | Update budget config |
| GET  | `/api/events/stream` | SSE event stream (frontend) |
| POST | `/api/simulator/spike` | Demo: token spike |
| POST | `/api/simulator/runaway` | Demo: runaway agent |
| POST | `/api/simulator/budget-overflow` | Demo: budget overflow |
| POST | `/api/simulator/dynatrace-anomaly` | Demo: Dynatrace anomaly |
| POST | `/api/simulator/reset` | Demo: reset to baseline |

---

## Local Development

```bash
# 1. Create & activate a virtualenv (Python 3.12+)
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env        # then fill in your keys/URIs

# 4. Run the server (hot reload)
uvicorn main:app --reload --port 8080

# 5. Verify
curl http://localhost:8080/health   # → {"status": "ok"}
```

You will also need: a MongoDB Atlas cluster, a Redis instance
(`docker run -p 6379:6379 redis`), a Gemini API key, a Google Cloud project with
Agent Builder enabled, and a Dynatrace trial token.

---

## Deployment (Google Cloud Run)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/quotamind-backend

gcloud run deploy quotamind-backend \
  --image gcr.io/PROJECT_ID/quotamind-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=...,MONGODB_URI=...,REDIS_URL=...
```

The resulting Cloud Run URL is the backend host the frontend's `NEXT_PUBLIC_API_URL` points to.
