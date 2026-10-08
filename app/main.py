"""
FastAPI entry point. Exposes a single endpoint: POST /query.

Why this file stays thin (no tool logic, no Claude calls here): main.py's
only job is HTTP concerns — request validation, routing, and turning
agent.py's output into an HTTP response. Keeping orchestration logic in
agent.py means agent.py can be unit-tested directly (see tests/) without
spinning up an HTTP server at all.
"""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent import run_agent
from app.config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Agent with Tool Calling",
    description=(
        "Accepts a natural-language query, uses Claude's tool-calling to pick "
        "one of three tools (web search simulation, local data lookup, or "
        "email drafting), executes it, and returns a structured result."
    ),
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The natural language request to route to a tool.")


@app.post("/query")
def query(request: QueryRequest) -> dict:
    """
    Run one query through the agent and return its structured result.

    Why failures here become a 502 rather than a 500: a 500 implies a bug
    in OUR code. If this fails, it's because the upstream Claude API could
    not be reached or returned something unusable after all retries were
    exhausted — that's an upstream dependency failure, which 502 (Bad
    Gateway) communicates more accurately to an API consumer than a bare
    500 would. Failures INSIDE a tool (bad lookup key, etc.) do NOT reach
    this except block — those are already caught in agent.py and returned
    as a normal 200 with status="error".
    """
    logger.info("Received /query request: %r", request.query)
    try:
        return run_agent(request.query)
    except Exception as exc:
        logger.error("Agent-level failure (not a tool failure): %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Agent could not process the request: {exc}",
        ) from exc


@app.get("/health")
def health() -> dict:
    """Trivial liveness check — useful for confirming the server is up before testing /query."""
    return {"status": "ok"}
