# AI Agent with Tool Calling

[![GitHub repo](https://img.shields.io/badge/GitHub-ai--agent--tool--calling-181717?logo=github)](https://github.com/CodeCrafter-creator/ai-agent-tool-calling)

A small, production-style FastAPI service that demonstrates an AI agent
using Anthropic Claude's **tool-calling (function-calling)** feature to
decide, per request, which of three tools to run — then actually runs it
and returns a structured result.

## The problem this demonstrates

A lot of "AI features" are really just a single prompt → single response.
The harder, more realistic problem is: a user sends one free-form request,
and the system has to (a) figure out *what kind* of request it is, (b)
route it to the right piece of backend logic, (c) execute that logic
safely, and (d) return something predictable to the caller — even when
the underlying AI call or the chosen tool fails.

This project is a minimal but real implementation of that pattern:

- **Intent classification** is delegated to Claude's tool-calling, instead
  of hand-written keyword/if-else routing (see the comment at the top of
  `app/agent.py` for why).
- **Execution** is isolated per tool, so one tool failing (bad input, a
  missing key, an API hiccup) never takes down the whole request.
- **Resilience** is built in at the one place that actually talks to an
  external API (`app/retry_utils.py`): transient failures get retried with
  exponential backoff; permanent failures (bad API key, bad request) fail
  immediately instead of retrying something that can never succeed.

## Architecture

```
                      ┌─────────────────────────┐
  POST /query         │        FastAPI           │
  {"query": "..."} ───▶   app/main.py             │
                      │   - validates request    │
                      │   - calls run_agent()     │
                      │   - 502 on agent failure  │
                      └────────────┬──────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │   app/agent.py            │
                      │   run_agent(query)        │
                      │                           │
                      │   1. _select_tool(query)  │──────▶  Claude API
                      │      tool_choice="any"    │◀──────  (tool_use block:
                      │      (retried on failure) │          name + input)
                      │                           │
                      │   2. dispatch chosen tool │
                      │      inside its own       │
                      │      try/except           │
                      └────────────┬──────────────┘
                                   │
                 ┌─────────────────┼─────────────────┐
                 ▼                 ▼                 ▼
        ┌────────────────┐ ┌───────────────┐ ┌───────────────────┐
        │ web_search_     │ │  data_lookup   │ │   draft_email      │
        │ simulation      │ │                │ │                    │
        │                 │ │ reads          │ │ calls Claude API    │
        │ pure local mock │ │ data/          │ │ again to generate   │
        │ (no network     │ │ database.json  │ │ the email body       │
        │ call)           │ │                │ │ (retried on failure) │
        └────────────────┘ └───────────────┘ └───────────────────┘
                 │                 │                 │
                 └─────────────────┴─────────────────┘
                                   │
                                   ▼
                 {"tool_used", "input", "output", "status"}
```

**Key design decisions** (explained in-line in the code, summarized here):

| Decision | Why |
|---|---|
| `tool_choice={"type": "any"}` not `"auto"` | Spec requires exactly one of 3 tools be chosen every time — `"any"` forces a tool call, `"auto"` would let Claude skip tools and reply with text instead. |
| Tool selection left to Claude, not keyword routing | Keyword/if-else routing is brittle against phrasing the developer didn't anticipate, and Claude already does this as part of tool-calling. |
| Exponential backoff, not fixed delay, on retries | Transient issues (rate limits, brief 5xx) often clear in 1-2s; a fixed delay either wastes time waiting too long or hammers the API too fast if the issue persists. |
| Auth/bad-request errors are NOT retried | No amount of retrying fixes an invalid API key or malformed request — failing fast avoids wasting ~3 attempts and several seconds on a call that can never succeed. |
| Each tool's execution wrapped in its own `try/except` in `agent.py` | A single tool misbehaving (bad lookup key, empty query) must return a clean `status: "error"` JSON response, not a 500 or stack trace. |
| Agent-level failures (can't reach Claude at all) → HTTP 502, not 500 | A 500 implies a bug in this code. If Claude is unreachable after retries, that's an upstream dependency failure — 502 (Bad Gateway) communicates that more accurately. |
| `web_search_simulation` is fully mocked, no real HTTP call | Keeps the project scoped to demonstrating tool-calling/orchestration, not search-API integration, and avoids needing a second paid API key. |

## Setup

Requires Python 3.11+.

```bash
# 1. Create and activate a virtual environment
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
# venv\Scripts\activate.bat    # Windows (cmd)
# source venv/bin/activate     # macOS/Linux

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and paste in your real ANTHROPIC_API_KEY

# 4. Run the server
uvicorn app.main:app --reload --port 8000
```

The server starts at `http://localhost:8000`. Interactive API docs
(Swagger UI) are available at `http://localhost:8000/docs`.

## Example requests

All three examples below assume the server is running locally with a
valid `ANTHROPIC_API_KEY` in `.env`. Exact wording of `output` will vary
slightly between runs since Claude generates it live — the `tool_used`
and overall shape will not.

### 1. Data lookup

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is your refund policy?"}'
```

Expected response:

```json
{
  "tool_used": "data_lookup",
  "input": { "key": "refund_policy_days" },
  "output": {
    "key": "refund_policy_days",
    "value": "14-day money-back guarantee from the date of purchase."
  },
  "status": "success"
}
```

### 2. Web search simulation

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Search for best practices in cold email outreach"}'
```

Expected response:

```json
{
  "tool_used": "web_search_simulation",
  "input": { "query": "best practices in cold email outreach" },
  "output": {
    "query": "best practices in cold email outreach",
    "result_count": 2,
    "results": [
      {
        "title": "Best Practices In Cold Email Outreach — Overview and Best Practices",
        "url": "https://example.com/articles/best-practices-in-cold-email-outreach",
        "snippet": "A practical overview of best practices in cold email outreach, covering common approaches and pitfalls."
      },
      {
        "title": "Top 5 Tools for Best Practices In Cold Email Outreach",
        "url": "https://example.com/guides/top-best-practices-in-cold-email-outreach",
        "snippet": "A comparison of the most popular tools and services related to best practices in cold email outreach."
      }
    ],
    "simulated": true
  },
  "status": "success"
}
```

### 3. Draft an email

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Draft an email to a prospect named Alex following up after a demo call"}'
```

Expected response:

```json
{
  "tool_used": "draft_email",
  "input": {
    "recipient": "Alex",
    "subject": "Following up on our demo call",
    "context": "Following up after a product demo call"
  },
  "output": {
    "recipient": "Alex",
    "subject": "Following up on our demo call",
    "body": "Hi Alex, thanks for taking the time to join the demo earlier. I wanted to follow up and see if you had any questions, or if there's anything I can clarify about what we covered. Happy to jump on another call if that's useful. Looking forward to hearing your thoughts."
  },
  "status": "success"
}
```

### Error example (bad data_lookup key)

If Claude picks `data_lookup` with a key that isn't in `data/database.json`,
the tool fails cleanly instead of crashing the request:

```json
{
  "tool_used": "data_lookup",
  "input": { "key": "warehouse_location" },
  "output": { "error": "No entry for key 'warehouse_location' in local database" },
  "status": "error"
}
```

## Running tests

Tests mock every Claude API call — no real network request or API key is
needed to run the suite.

```bash
pytest -v
```

```
tests/test_agent.py::test_run_agent_dispatches_to_selected_tool PASSED
tests/test_agent.py::test_run_agent_returns_clean_error_when_tool_raises PASSED
tests/test_data_lookup.py::test_data_lookup_returns_known_key PASSED
tests/test_data_lookup.py::test_data_lookup_raises_on_unknown_key PASSED
tests/test_data_lookup.py::test_data_lookup_rejects_empty_key PASSED
tests/test_draft_email.py::test_draft_email_returns_generated_body PASSED
tests/test_draft_email.py::test_draft_email_rejects_missing_fields PASSED
tests/test_draft_email.py::test_draft_email_requires_api_key PASSED
tests/test_web_search.py::test_web_search_simulation_returns_expected_shape PASSED
tests/test_web_search.py::test_web_search_simulation_rejects_empty_query PASSED

10 passed
```

## Project structure

```
ai-agent-tool-calling/
├── app/
│   ├── main.py          FastAPI app, /query and /health endpoints
│   ├── agent.py          Orchestration: ask Claude which tool, dispatch it
│   ├── tools.py           The 3 tools + their Claude tool-schema definitions
│   ├── retry_utils.py     Shared exponential-backoff retry wrapper
│   └── config.py          Env vars, paths, logging setup
├── data/
│   └── database.json      Local JSON "database" used by data_lookup
├── tests/
│   ├── conftest.py         Shared fixtures for building fake Claude responses
│   ├── test_web_search.py
│   ├── test_data_lookup.py
│   ├── test_draft_email.py
│   └── test_agent.py
├── requirements.txt
├── .env.example
└── README.md
```

## Scope / what this intentionally does NOT include

- No authentication, no database server, no Docker, no frontend — scoped
  to the agent + API only, by design.
- `web_search_simulation` does not call a real search API — it returns
  clearly-labeled mocked data (`"simulated": true` in every response).
- No persistence of request history — each `/query` call is stateless.
