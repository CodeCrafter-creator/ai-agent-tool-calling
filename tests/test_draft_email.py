"""
Tests for the draft_email tool.

This is the only tool that itself calls the Claude API, so the real
network call is mocked out by replacing app.tools.anthropic.Anthropic with
a stand-in whose .messages.create(...) returns a canned Message object
(see tests/conftest.py). The real ANTHROPIC_API_KEY is never needed or
used to run this test suite.
"""

from unittest.mock import MagicMock

import pytest

from app import tools
from tests.conftest import make_text_message


@pytest.fixture
def mock_anthropic_client(monkeypatch):
    fake_response = make_text_message("Hi there, just following up on our conversation.")

    fake_client = MagicMock()
    fake_client.messages.create.return_value = fake_response

    # Patch both the client constructor and the API key check: tests must
    # not require a real key, and must not make a real network call.
    monkeypatch.setattr(tools.anthropic, "Anthropic", MagicMock(return_value=fake_client))
    monkeypatch.setattr(tools, "ANTHROPIC_API_KEY", "fake-test-key")

    return fake_client


def test_draft_email_returns_generated_body(mock_anthropic_client):
    result = tools.draft_email(
        recipient="Jane Doe",
        subject="Following up",
        context="She requested a status update on the onboarding call.",
    )

    assert result["recipient"] == "Jane Doe"
    assert result["subject"] == "Following up"
    assert result["body"] == "Hi there, just following up on our conversation."
    mock_anthropic_client.messages.create.assert_called_once()


def test_draft_email_rejects_missing_fields(mock_anthropic_client):
    with pytest.raises(ValueError):
        tools.draft_email(recipient="", subject="Subj", context="Ctx")


def test_draft_email_requires_api_key(monkeypatch):
    monkeypatch.setattr(tools, "ANTHROPIC_API_KEY", None)

    with pytest.raises(RuntimeError):
        tools.draft_email(recipient="Jane", subject="Subj", context="Ctx")
