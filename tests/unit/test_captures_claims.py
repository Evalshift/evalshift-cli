"""The wording check behind unbacked-tool-claim refusals."""

from __future__ import annotations

import pytest

from evalshift_cli.captures.claims import find_tool_claim


@pytest.mark.parametrize(
    ("text", "phrase"),
    [
        (
            "I don't have that information—my notes search didn't find anything specific",
            "my notes search didn't",
        ),
        ("I searched your notes and found nothing.", "I searched"),
        ("I've checked your calendar for Friday.", "I've checked"),
        ("I’ve already looked through your tasks.", "I’ve already looked through"),  # noqa: RUF001 — curly apostrophe
        ("Let me check your notes for that.", "Let me check"),
        ("I'll look into it.", "I'll look"),
        ("The notes search returned no results.", "The notes search returned"),
        ("My search didn’t turn up anything.", "My search didn’t"),  # noqa: RUF001 — curly apostrophe
    ],
)
def test_claims_are_found_and_quoted(text: str, phrase: str) -> None:
    assert find_tool_claim(text) == phrase


@pytest.mark.parametrize(
    "text",
    [
        "Per your notes from Aug 20, the dentist told you to floss daily.",
        "Your passport renewal is due by September 15, 2026, and it's still pending.",
        "I can't search contract documents.",
        "Would you like me to check?",
        "Use the search bar at the top.",
        "Let me know if you need anything else.",
        "I looked at your question again: Friday is clear.",
        "Checked items are done.",
        "",
    ],
)
def test_non_claims_are_ignored(text: str) -> None:
    assert find_tool_claim(text) is None
