"""
Retry wrapper for Anthropic API calls.

Why this exists as a separate, reusable function instead of inline
try/except in every place we call Claude: there are two call sites
(tool-selection in agent.py, and email generation in tools.py). Both need
identical retry behavior, so this is extracted once rather than copied
twice and risking the two copies drifting apart.
"""

import logging
import time
from typing import Callable, TypeVar

import anthropic

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Errors where retrying is pointless: the request itself is invalid, or the
# credentials are wrong. No amount of waiting fixes a bad API key or a
# malformed request, so we fail fast instead of burning 3 attempts and
# several seconds on a call that will never succeed.
NON_RETRYABLE_EXCEPTIONS = (
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.BadRequestError,
    anthropic.NotFoundError,
    anthropic.UnprocessableEntityError,
)


def call_with_retries(
    api_call: Callable[[], T],
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
) -> T:
    """
    Run `api_call` (a zero-argument function wrapping a single Anthropic
    API request) with up to `max_attempts` tries.

    Why exponential backoff instead of a fixed delay between retries:
    transient failures (rate limits, brief network blips, momentary 5xx
    errors from Anthropic) often clear up within a second or two, so the
    first retry fires quickly. But if the issue persists, hammering the API
    again at the same fixed interval just repeats the failure and can make
    rate limiting worse. Backing off (1s, 2s, 4s, ...) gives the upstream
    service increasing room to recover before the next attempt.
    """
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return api_call()
        except NON_RETRYABLE_EXCEPTIONS as exc:
            # Fail immediately — retrying would waste time and still fail.
            logger.error("Non-retryable Anthropic API error: %s", exc)
            raise
        except anthropic.APIError as exc:
            last_exception = exc
            logger.error(
                "Anthropic API call failed (attempt %d/%d): %s",
                attempt,
                max_attempts,
                exc,
            )
            if attempt < max_attempts:
                delay = base_delay_seconds * (2 ** (attempt - 1))
                time.sleep(delay)

    # All attempts exhausted — re-raise the last error so the caller
    # (agent.py or tools.py) can decide how to surface it to the client.
    assert last_exception is not None
    raise last_exception
