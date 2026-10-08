"""
Agent orchestration: ask Claude which tool fits the user's query, execute
that tool, and return a structured result.

Why tool SELECTION is left to the model instead of hardcoded keyword
routing (e.g. "if 'email' in query: use draft_email"): keyword matching is
brittle and breaks on phrasing the developer didn't anticipate ("write
something to send to..." vs "draft an email to..."). Claude's tool-calling
already does intent classification as part of what it's trained for, so
re-implementing a worse version of it with if/else string matching would
be redundant and lower quality. Hardcoded routing would also need to be
updated by hand every time a tool's trigger phrasing changes.
"""

import logging
from typing import Any

import anthropic

from app.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, MAX_TOKENS, MAX_RETRY_ATTEMPTS, RETRY_BASE_DELAY_SECONDS
from app.retry_utils import call_with_retries
from app.tools import TOOL_DEFINITIONS, TOOL_DISPATCH

logger = logging.getLogger(__name__)


def _select_tool(query: str) -> anthropic.types.Message:
    """
    Make the Claude API call that decides which tool to use for `query`.

    Why tool_choice is set to {"type": "any"} instead of "auto": "auto"
    lets Claude reply with plain text and skip tools entirely, which fits
    a general chatbot. This project's spec is narrower — every query must
    resolve to exactly one of the 3 tools — so "any" is used instead: it
    forces Claude to call *some* tool, while still letting Claude (not our
    code) decide *which* one. This matches "decide which ONE of 3 tools to
    invoke" precisely, rather than allowing a fourth, no-tool outcome.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set — cannot contact Claude API")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def _make_call() -> anthropic.types.Message:
        return client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            tools=TOOL_DEFINITIONS,
            tool_choice={"type": "any"},
            messages=[{"role": "user", "content": query}],
        )

    # Retrying here (tool selection) and separately inside draft_email
    # (body generation) rather than wrapping the whole /query request in
    # one giant retry: these are two independent Claude API calls that can
    # fail independently, and retrying the first one again after the
    # second one already failed would needlessly repeat work and API spend.
    return call_with_retries(
        _make_call,
        max_attempts=MAX_RETRY_ATTEMPTS,
        base_delay_seconds=RETRY_BASE_DELAY_SECONDS,
    )


def run_agent(query: str) -> dict[str, Any]:
    """
    Run the full agent flow for one user query:
      1. Ask Claude which tool applies and with what arguments.
      2. Execute that tool.
      3. Return a structured result dict matching the API's response shape.

    Errors are handled in two separate layers on purpose:
      - Failure to even reach Claude / parse its tool choice is a problem
        with the AGENT itself, so it's allowed to raise and becomes a 502
        in main.py (the request as a whole could not be processed).
      - Failure INSIDE a chosen tool (bad key, bad recipient, etc.) is a
        problem with that one TOOL, so it's caught here and returned as a
        normal 200 response with status="error" — one tool misbehaving
        should never look like our API is down.
    """
    logger.info("Agent received query: %r", query)

    response = _select_tool(query)

    # Find the tool_use block in Claude's response. With tool_choice="any"
    # there is exactly one, but we scan defensively rather than assuming
    # content[0] is it, since Claude may also emit preceding text blocks.
    tool_use_block = next(
        (block for block in response.content if block.type == "tool_use"),
        None,
    )

    if tool_use_block is None:
        # Should not happen given tool_choice="any", but guards against an
        # unexpected API response shape instead of raising a confusing
        # AttributeError deeper in the code.
        raise RuntimeError("Claude did not return a tool_use block despite tool_choice='any'")

    tool_name: str = tool_use_block.name
    tool_input: dict[str, Any] = tool_use_block.input

    logger.info("Claude selected tool=%r with input=%r", tool_name, tool_input)

    if tool_name not in TOOL_DISPATCH:
        # Defensive check: Claude is constrained to TOOL_DEFINITIONS, so
        # this should be unreachable, but a future tool rename/typo
        # mismatch between TOOL_DEFINITIONS and TOOL_DISPATCH would surface
        # here clearly instead of as a KeyError with no context.
        raise RuntimeError(f"Claude selected unknown tool: {tool_name!r}")

    try:
        output = TOOL_DISPATCH[tool_name](tool_input)
        logger.info("Tool %r succeeded", tool_name)
        return {
            "tool_used": tool_name,
            "input": tool_input,
            "output": output,
            "status": "success",
        }
    except Exception as exc:
        # Caught broadly and per-tool (not globally around the whole
        # request) because each tool fails for different, expected
        # reasons (bad DB key, empty query, Claude API error during email
        # generation) and all of them should degrade the same way: a clean
        # error status in the response, never a stack trace to the client.
        logger.error("Tool %r failed: %s", tool_name, exc)
        return {
            "tool_used": tool_name,
            "input": tool_input,
            "output": {"error": str(exc)},
            "status": "error",
        }
