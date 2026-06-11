# QuotaMind AI — Backend

> Autonomous AI Quota & Inference Optimization Agent — FastAPI service.

This is the backend for **QuotaMind AI**, built for the Google Cloud Rapid Agent
Hackathon (Dynatrace partner track). It ingests AI workload traffic, runs a suite
of optimization engines, and drives an autonomous agent (Google Cloud Agent Builder
+ Gemini 2.5) that reroutes, compresses, throttles, and forecasts — all streamed to
the dashboard in real time over SSE.

**Status: feature-complete.** 23 API routes, autonomous 30-second agent loop,
live-verified against MongoDB Atlas, Redis, Gemini, Agent Builder, and a real
Dynatrace MCP server. The React dashboard lives in the sibling
`quotamind-frontend` repo/folder.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI (Python 3.12), Uvicorn |
| Validation | Pydantic v2 |
| Database | MongoDB Atlas (async via Motor) |
| Cache / event bus | Redis (counters + pub/sub) |
| AI core | Gemini 2.5 Pro / Flash (`google-genai`) |
| Agent orchestration | Google Cloud Agent Builder (Vertex AI / Dialogflow CX) |
| Observability MCP | Real `@dynatrace-oss/dynatrace-mcp-server` over stdio (reads) + Dynatrace REST API (writes/fallback) |
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

## How It Works (the autonomous loop)

Every `ORCHESTRATOR_INTERVAL_SECONDS` (default 30s), `agent/orchestrator.py` runs one cycle:

1. **Collect signals** — quota forecast (burn rate + exhaustion ETA), budget status,
   per-agent runaway analysis, and a live **Dynatrace observability briefing fetched
   through the real Dynatrace MCP server** (`list_problems`), with REST fallback.
2. **Reason** — the context bundle is sent to a **Google Cloud Agent Builder** playbook
   (Gemini 2.5), which replies with structured `REASONING:` / `ACTIONS:` output.
3. **Act** — `agent/agent_runner.py` parses the response into an `AgentDecision` and
   executes the chosen tools: `reroute` (downgrade model tier), `compress` (prompt
   compression), `throttle` / `pause` (runaway containment), `cache`, and `alert`.
4. **Close the loop** — every mutating action is logged to Mongo as an
   `OptimizationEvent`, streamed to the dashboard over SSE, and **pushed back into the
   Dynatrace timeline** via event ingest, so QuotaMind's actions appear alongside the
   anomalies that triggered them.

If Agent Builder is unreachable, a deterministic fallback policy still protects
critical agents — the loop never dies on a single failed cycle.

The **workload simulator** (`simulator/workload_simulator.py`) provides demo traffic:
four synthetic AI agents with distinct personalities (a critical support bot, a
Pro-overusing summarizer, a low-priority analytics agent, a bursty workflow agent)
plus one-click crisis scenarios — token spike, runaway loop, budget overflow, and a
real Dynatrace anomaly event.

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
| GET  | `/health` | Per-dependency status → `{"app","mongodb","redis","dynatrace": "mcp\|rest\|unconfigured"}` |
| POST | `/api/requests/ingest` | Ingest one AI request (from simulator) |
| GET  | `/api/requests/summary` | Aggregated request stats |
| GET  | `/api/requests/live` | Most recent requests feed |
| GET  | `/api/metrics/summary` | Tokens / cost / quota |
| GET  | `/api/metrics/cache-stats` | Cache hit rate + savings |
| GET  | `/api/metrics/savings` | Savings by optimization type |
| GET  | `/api/forecast` | Quota exhaustion forecast |
| GET  | `/api/recommendations` | Gemini optimization insights |
| GET  | `/api/agents/status` | All agent statuses + models |
| GET  | `/api/agents/{id}/health` | Runaway risk score |
| POST | `/api/agents/{id}/pause` | Manual pause override |
| GET  | `/api/budget/status` | Budget breakdown + projections |
| POST | `/api/config/budget` | Update budget config |
| GET  | `/api/events/stream` | SSE event stream (frontend) |
| GET  | `/api/simulator/status` | Simulator state |
| POST | `/api/simulator/start` | Start continuous demo traffic |
| POST | `/api/simulator/stop` | Stop demo traffic |
| POST | `/api/simulator/spike` | Demo: token spike |
| POST | `/api/simulator/runaway` | Demo: runaway agent |
| POST | `/api/simulator/budget-overflow` | Demo: budget overflow |
| POST | `/api/simulator/dynatrace-anomaly` | Demo: Dynatrace anomaly (real event push) |
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

# 4. Run the server (hot reload). Tip: ORCHESTRATOR_AUTOSTART=false
#    boots the API without the autonomous agent loop.
uvicorn main:app --reload --port 8080

# 5. Verify
curl http://localhost:8080/health
# → {"app":"ok","mongodb":"ok","redis":"ok","dynatrace":"mcp"}
```

You will also need: a MongoDB Atlas cluster, a Redis instance
(`docker run -p 6379:6379 redis`), a Gemini API key, a Google Cloud project with
Agent Builder enabled, a Dynatrace API token (REST), and a Dynatrace platform
token (real MCP server — launched automatically via `npx`, requires Node.js).

---

## Deployment (Google Cloud Run)

The image bundles **Node.js + the pinned Dynatrace MCP server package**, so the
real MCP transport works in Cloud Run (no silent REST fallback).

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/quotamind-backend

gcloud run deploy quotamind-backend \
  --image gcr.io/PROJECT_ID/quotamind-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars GEMINI_API_KEY=...,MONGODB_URI=...,REDIS_URL=...,DT_PLATFORM_TOKEN=...,DT_ENVIRONMENT=...,DYNATRACE_ENV_URL=...,DYNATRACE_API_TOKEN=...
```

The resulting Cloud Run URL is the backend host the frontend's `VITE_API_BASE_URL`
points to.

---

## License

MIT — see [LICENSE](LICENSE).
