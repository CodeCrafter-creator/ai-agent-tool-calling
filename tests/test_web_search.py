"""
Tests for the web_search_simulation tool.

No mocking of an external API is needed here because the tool itself
never makes a network call — it's a local mock by design (see the comment
block at the top of app/tools.py). These tests just verify the shape and
basic correctness of the fake data, and that invalid input is rejected.
"""

import pytest

from app.tools import web_search_simulation


def test_web_search_simulation_returns_expected_shape():
    result = web_search_simulation("outbound email automation")

    assert result["query"] == "outbound email automation"
    assert result["simulated"] is True
    assert isinstance(result["results"], list)
    assert result["result_count"] == len(result["results"])
    assert len(result["results"]) > 0

    for item in result["results"]:
        assert "title" in item
        assert "url" in item
        assert "snippet" in item


def test_web_search_simulation_rejects_empty_query():
    with pytest.raises(ValueError):
        web_search_simulation("")

    with pytest.raises(ValueError):
        web_search_simulation("   ")
