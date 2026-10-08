"""
Shared pytest fixtures.

Why a shared helper for building fake Anthropic responses instead of
constructing them inline in each test: every test that mocks a Claude call
needs an object shaped like anthropic.types.Message (with a `.content`
list of blocks). Building that by hand in each test file would repeat the
same boilerplate three times; a single builder here keeps the tests
focused on what's actually being verified.
"""

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage


def make_tool_use_message(tool_name: str, tool_input: dict) -> Message:
    """Build a fake Anthropic Message whose content is a single tool_use block."""
    return Message(
        id="msg_test_001",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[
            ToolUseBlock(type="tool_use", id="toolu_test_001", name=tool_name, input=tool_input)
        ],
        stop_reason="tool_use",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=10),
    )


def make_text_message(text: str) -> Message:
    """Build a fake Anthropic Message whose content is a single plain text block."""
    return Message(
        id="msg_test_002",
        type="message",
        role="assistant",
        model="claude-sonnet-4-6",
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=10),
    )
