"""
Central place for environment variables and logging setup.

Why one shared config module instead of reading os.environ in every file:
if a client ever wants to change the model name, log level, or database
path, there is exactly one place to look. Scattering os.getenv() calls
across the codebase makes it easy to miss one during a config change.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a local .env file into the process environment.
# This must run before we read any ANTHROPIC_API_KEY/CLAUDE_MODEL values below.
load_dotenv()

# --- Paths -------------------------------------------------------------
# Resolved relative to this file (not the current working directory) so the
# app behaves the same whether it's started from this folder or elsewhere.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATABASE_PATH: Path = PROJECT_ROOT / "data" / "database.json"
LOG_FILE_PATH: Path = PROJECT_ROOT / "app.log"

# --- Anthropic API settings ---------------------------------------------
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")

# The model is read from the environment (with a fallback default) rather
# than hardcoded, because Anthropic periodically retires old model names.
# Hardcoding one risks the whole project breaking the moment that model is
# deprecated. Pointing CLAUDE_MODEL at whatever the account currently has
# access to keeps this working without a code change.
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")

# Shared tuning values for the Claude API calls (kept in one place so both
# the tool-routing call and the draft_email generation call stay consistent).
MAX_TOKENS: int = 1024
MAX_RETRY_ATTEMPTS: int = 3
RETRY_BASE_DELAY_SECONDS: float = 1.0


def configure_logging() -> None:
    """
    Set up structured logging to BOTH stdout and a local app.log file.

    Why both: stdout is what you see live while developing or in a
    container's log stream; the file survives after the process exits, so
    you can look back at what happened during a client demo without having
    had a terminal open and scrolled up far enough.

    This is called once, at app startup (see main.py), not per-request —
    calling basicConfig repeatedly would duplicate log handlers.
    """
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(),  # stdout
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),  # app.log
        ],
    )
