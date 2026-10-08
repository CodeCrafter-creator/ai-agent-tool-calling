"""
The three tools the agent can choose between, plus the JSON schema
descriptions that get sent to Claude so it knows these tools exist and
what arguments each one takes.

Why each tool raises a plain exception on failure instead of returning
some kind of {"error": ...} dict: it keeps the "did this succeed" logic
in exactly one place (agent.py's try/except around each dispatch), rather
than every tool having to invent its own error shape that agent.py then
has to special-case.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, DATABASE_PATH, MAX_TOKENS, MAX_RETRY_ATTEMPTS, RETRY_BASE_DELAY_SECONDS
from app.retry_utils import call_with_retries

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1: web_search_simulation
# ---------------------------------------------------------------------------
# This is intentionally FAKE. Building a real search integration (SerpAPI,
# Bing, Google Custom Search, etc.) would add a paid dependency and an
# external API key unrelated to the point of this portfolio project, which
# is demonstrating tool-calling/agent orchestration, not search quality.
# The mocked results are shaped like a real search API's response so the
# rest of the pipeline (agent dispatch, JSON response shape) is identical
# to what it would be with a real provider swapped in later.
def web_search_simulation(query: str) -> dict[str, Any]:
    """Return mocked but realistic search results for `query`. No network call is made."""
    if not query or not query.strip():
        raise ValueError("web_search_simulation requires a non-empty query")

    logger.info("Running web_search_simulation (mocked) for query=%r", query)

    # A handful of canned "result" templates so repeated calls don't all
    # look identical, without needing any real search backend.
    templates = [
        {
            "title": f"{query.title()} — Overview and Best Practices",
            "url": f"https://example.com/articles/{query.lower().replace(' ', '-')}",
            "snippet": f"A practical overview of {query}, covering common approaches and pitfalls.",
        },
        {
            "title": f"Top 5 Tools for {query.title()}",
            "url": f"https://example.com/guides/top-{query.lower().replace(' ', '-')}",
            "snippet": f"A comparison of the most popular tools and services related to {query}.",
        },
        {
            "title": f"{query.title()}: A 2025 Guide",
            "url": f"https://example.com/blog/{query.lower().replace(' ', '-')}-guide",
            "snippet": f"Updated guidance on {query}, including recent changes and recommendations.",
        },
    ]

    # random.sample without a fixed seed means results vary call to call,
    # which is closer to how a real search API behaves (rankings shift).
    results = random.sample(templates, k=min(2, len(templates)))

    return {
        "query": query,
        "result_count": len(results),
        "results": results,
        "simulated": True,  # explicit flag so nobody mistakes this for live data
    }


# ---------------------------------------------------------------------------
# Tool 2: data_lookup
# ---------------------------------------------------------------------------
def _load_database() -> dict[str, str]:
    """
    Load the local JSON "database" from disk on every call rather than
    caching it in memory. For a file this small, re-reading costs
    microseconds, but it means an edit to database.json takes effect
    immediately without restarting the server — useful for a client demo
    where you might tweak a value live.
    """
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(f"Local database file not found at {DATABASE_PATH}")

    with open(DATABASE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def data_lookup(key: str) -> dict[str, Any]:
    """Look up `key` in the local JSON database and return its value."""
    if not key or not key.strip():
        raise ValueError("data_lookup requires a non-empty key")

    logger.info("Running data_lookup for key=%r", key)
    database = _load_database()

    if key not in database:
        # Raising KeyError (rather than returning None) makes "not found"
        # loud and explicit — it gets caught in agent.py and reported as a
        # clean tool error, instead of silently returning a null value that
        # looks like a successful lookup.
        raise KeyError(f"No entry for key '{key}' in local database")

    return {"key": key, "value": database[key]}


# ---------------------------------------------------------------------------
# Tool 3: draft_email
# ---------------------------------------------------------------------------
def draft_email(recipient: str, subject: str, context: str) -> dict[str, Any]:
    """
    Generate a draft email body using the Claude API.

    Unlike the other two tools, this one makes its own (separate) call to
    Claude — it does not reuse the tool-selection response. The first call
    to Claude only decides WHICH tool to use; it does not write the email
    itself, because the email body depends on inputs (recipient/subject/
    context) that are only finalized once this tool is actually invoked.
    """
    if not recipient or not subject or not context:
        raise ValueError("draft_email requires recipient, subject, and context")

    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot generate a draft email")

    logger.info("Running draft_email for recipient=%r subject=%r", recipient, subject)

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = (
        f"Write a short, professional email body.\n"
        f"Recipient: {recipient}\n"
        f"Subject: {subject}\n"
        f"Context: {context}\n\n"
        "Return only the email body text, no subject line, no markdown formatting."
    )

    def _make_call() -> anthropic.types.Message:
        return client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

    # Same retry wrapper used for the tool-selection call in agent.py — see
    # retry_utils.py for why exponential backoff over a fixed delay.
    response = call_with_retries(
        _make_call,
        max_attempts=MAX_RETRY_ATTEMPTS,
        base_delay_seconds=RETRY_BASE_DELAY_SECONDS,
    )

    # Claude's response content is a list of blocks; for a plain text
    # generation with no tools involved, it is a single text block.
    body = "".join(block.text for block in response.content if block.type == "text").strip()

    return {
        "recipient": recipient,
        "subject": subject,
        "body": body,
    }


# ---------------------------------------------------------------------------
# Tool schemas sent to Claude's tool-calling API.
# ---------------------------------------------------------------------------
# These descriptions are what Claude actually reads to decide which tool
# fits the user's query — vague descriptions lead to wrong tool choices,
# so each one states plainly what the tool is for and when to use it.
TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "web_search_simulation",
        "description": (
            "Search the web for general information, news, or background on a topic. "
            "Use this when the user asks a question that requires looking something up "
            "that is not a known internal data field and is not a request to write an email. "
            "Returns mocked/simulated search results, not live internet data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query describing what to look up.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "data_lookup",
        "description": (
            "Look up a specific internal fact by key from a small local database. "
            "Use this when the user asks about a specific known internal field such as "
            "support email, business hours, pricing, refund policy, or onboarding time. "
            "Available keys: support_email, refund_policy_days, business_hours, "
            "pricing_plan_basic, company_address, onboarding_time_estimate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The exact database key to look up.",
                }
            },
            "required": ["key"],
        },
    },
    {
        "name": "draft_email",
        "description": (
            "Draft an email body. Use this when the user asks to write, draft, or compose "
            "an email to someone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Who the email is addressed to (name or role).",
                },
                "subject": {
                    "type": "string",
                    "description": "The subject line of the email.",
                },
                "context": {
                    "type": "string",
                    "description": "Background/context the email body should be based on.",
                },
            },
            "required": ["recipient", "subject", "context"],
        },
    },
]

# Maps a tool name (as chosen by Claude) to the actual Python function that
# executes it. A dict dispatch here is simpler and more explicit than
# getattr()-based reflection or a class hierarchy — there are only 3 tools,
# and this makes the full set of what the agent can do visible at a glance.
TOOL_DISPATCH: dict[str, Any] = {
    "web_search_simulation": lambda tool_input: web_search_simulation(**tool_input),
    "data_lookup": lambda tool_input: data_lookup(**tool_input),
    "draft_email": lambda tool_input: draft_email(**tool_input),
}
