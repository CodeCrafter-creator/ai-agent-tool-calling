"""
Tests for the data_lookup tool.

Why a temporary JSON file instead of reading the real data/database.json:
tests should not depend on the real database file's contents staying the
same forever — if someone edits database.json for a demo, these tests
should not silently start failing (or silently start passing for the
wrong reason). Pointing DATABASE_PATH at a throwaway file keeps the test
isolated from that.
"""

import json

import pytest

from app import tools


@pytest.fixture
def fake_database(tmp_path, monkeypatch):
    db_file = tmp_path / "database.json"
    db_file.write_text(json.dumps({"support_email": "test@example.com"}), encoding="utf-8")
    monkeypatch.setattr(tools, "DATABASE_PATH", db_file)
    return db_file


def test_data_lookup_returns_known_key(fake_database):
    result = tools.data_lookup("support_email")

    assert result == {"key": "support_email", "value": "test@example.com"}


def test_data_lookup_raises_on_unknown_key(fake_database):
    with pytest.raises(KeyError):
        tools.data_lookup("nonexistent_key")


def test_data_lookup_rejects_empty_key(fake_database):
    with pytest.raises(ValueError):
        tools.data_lookup("")
