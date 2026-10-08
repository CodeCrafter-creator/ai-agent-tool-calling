"""
Tests for the agent orchestration layer (app/agent.py).

These mock the Claude "tool selection" call (not the tools themselves) to
verify two things the individual tool tests can't: that agent.py correctly
dispatches to whichever tool Claude picked, and that a tool raising an
exception is caught and turned into a status="error" response rather than
propagating up as an unhandled exception.
"""

from unittest.mock import MagicMock

import pytest

from app import agent
from tests.conftest import make_tool_use_message


@pytest.fixture
def mock_tool_selection(monkeypatch):
    """Patch app.agent's Anthropic client so no real API call is made."""
    fake_client = MagicMock()
    monkeypatch.setattr(agent.anthropic, "Anthropic", MagicMock(return_value=fake_client))
    monkeypatch.setattr(agent, "ANTHROPIC_API_KEY", "fake-test-key")
    return fake_client


def test_run_agent_dispatches_to_selected_tool(mock_tool_selection, monkeypatch):
    mock_tool_selection.messages.create.return_value = make_tool_use_message(
        "data_lookup", {"key": "support_email"}
    )
    monkeypatch.setattr(
        agent, "TOOL_DISPATCH", {"data_lookup": lambda tool_input: {"key": tool_input["key"], "value": "ok@example.com"}}
    )

    result = agent.run_agent("What's your support email?")

    assert result["status"] == "success"
    assert result["tool_used"] == "data_lookup"
    assert result["input"] == {"key": "support_email"}
    assert result["output"] == {"key": "support_email", "value": "ok@example.com"}


def test_run_agent_returns_clean_error_when_tool_raises(mock_tool_selection, monkeypatch):
    mock_tool_selection.messages.create.return_value = make_tool_use_message(
        "data_lookup", {"key": "unknown_key"}
    )

    def _raise(_tool_input):
        raise KeyError("No entry for key 'unknown_key' in local database")

    monkeypatch.setattr(agent, "TOOL_DISPATCH", {"data_lookup": _raise})

    result = agent.run_agent("What's the value of unknown_key?")

    assert result["status"] == "error"
    assert result["tool_used"] == "data_lookup"
    assert "unknown_key" in result["output"]["error"]
