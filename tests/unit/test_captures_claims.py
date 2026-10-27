"""The wording check behind unbacked-tool-claim refusals."""

from __future__ import annotations

import pytest

from evalshift_cli.captures.claims import find_tool_claim, unbacked_history_claims
from evalshift_cli.suite.models import ChatMessage, HistoryToolCall


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
        ("My search didn’t turn up anything.", "My search didn’t"),  # noqa: RUF001 — curly apostrophe
        # Known, accepted false positive: past-tense review of content the user pasted reads
        # exactly like a claimed lookup. Wording alone can't tell them apart; the turn is
        # rescuable with --allow-unbacked-claims, and the refusal quotes the phrase.
        ("I checked the code you pasted and it looks fine.", "I checked"),
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
        # A promise is not a claim that an action happened -- and a clarifying question
        # before acting is exactly the turn where calling no tool is correct.
        "Let me check your notes for that.",
        "I'll look into it.",
        "Let me know the date and I'll check your calendar.",
        "Once you confirm the project name, I'll search your notes.",
        "Sure — let me find a time that works. What duration do you need?",
        "Let me check I understand: you want the Friday slot moved?",
        "I'll look forward to it!",
        "Let me check: 2+2=4.",
        # Only *my* search reports a result the reply claims for itself; "the" search/query
        # is as often the user's or a piece of code under discussion.
        "The notes search returned no results.",
        "The query didn't include a WHERE clause.",
        "The search didn't match because the term is misspelled.",
        "",
    ],
)
def test_non_claims_are_ignored(text: str) -> None:
    assert find_tool_claim(text) is None


def _user(text: str) -> ChatMessage:
    return ChatMessage(role="user", content=text)


def _assistant(text: str, *, calls: list[HistoryToolCall] | None = None) -> ChatMessage:
    return ChatMessage(role="assistant", content=text, tool_calls=calls)


def _tool(call_id: str, result: str) -> ChatMessage:
    return ChatMessage(role="tool", content=result, tool_call_id=call_id)


_SEARCH = HistoryToolCall(id="c1", name="search_notes", arguments={"q": "PO"})


def test_history_claim_without_a_tool_call_is_reported_by_position() -> None:
    history = [_user("PO number?"), _assistant("Sorry, my notes search didn't find it.")]
    assert unbacked_history_claims(history) == [(2, "my notes search didn't")]


def test_history_claim_with_tool_call_in_its_turn_is_backed() -> None:
    """The native-shapes shape: "Let me look." beside a real call, then the answer."""
    history = [
        _user("PO number?"),
        _assistant("Let me look.", calls=[_SEARCH]),
        _tool("c1", '{"po": "PO-7"}'),
        _assistant("I searched and found PO-7."),
    ]
    assert unbacked_history_claims(history) == []


def test_history_reply_that_claims_nothing_is_ignored() -> None:
    history = [
        ChatMessage(role="system", content="You are Kaila."),
        _user("dentist?"),
        _assistant("Per your notes, floss daily."),
    ]
    assert unbacked_history_claims(history) == []


def test_unbacked_claims_in_different_turns_are_all_reported_in_order() -> None:
    history = [
        _user("PO number?"),
        _assistant("I searched your notes and found nothing."),
        _user("Friday?"),
        _assistant("I've checked your calendar: it's clear."),
    ]
    assert unbacked_history_claims(history) == [(2, "I searched"), (4, "I've checked")]


def test_a_new_user_turn_resets_backing() -> None:
    """A call in an earlier turn does not back a claim made after the user spoke again."""
    history = [
        _user("PO number?"),
        _assistant("", calls=[_SEARCH]),
        _tool("c1", '{"po": "PO-7"}'),
        _assistant("It is PO-7."),
        _user("And the Globex one?"),
        _assistant("I checked and there is none."),
    ]
    assert unbacked_history_claims(history) == [(6, "I checked")]


def test_user_text_riding_on_a_tool_result_turn_keeps_backing() -> None:
    """Claude delivers tool results in a user turn; text in that same turn normalises to
    tool rows followed by a user row. It is still the answer to the call, not a new turn."""
    history = [
        _user("PO number?"),
        _assistant("", calls=[_SEARCH]),
        _tool("c1", '{"po": "PO-7"}'),
        _user("(the user is waiting)"),
        _assistant("I searched your notes: it is PO-7."),
    ]
    assert unbacked_history_claims(history) == []
