"""Spot a reply that claims a tool action, for refusing hallucinated turns at promotion.

A captured turn that was offered tools, called none, and still says "my notes search didn't
find anything" records an action that never happened. Promoting it would make that
hallucination ground truth (``expected_no_tools: true`` plus the fabricated reply). The check is
wording only — deterministic, offline, English — and deliberately narrow: a first-person claim
that an action already *happened* ("I searched", "I've checked", "my notes search didn't …"),
never a bare mention of a tool or a noun like "search bar".

Promises are not claims: "let me check" or "I'll search your notes once you confirm" is how a
model asks a clarifying question before acting -- exactly the turn where calling no tool is
right. A reported result counts only as *my* search/lookup/query: "the query didn't include a
WHERE clause" is about the user's query as often as the model's.

Known false positive, accepted: past-tense review of content the user pasted ("I checked the
code you pasted") reads exactly like a claimed lookup. The refusal quotes the phrase and
``--allow-unbacked-claims`` rescues the turn.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from evalshift_cli.suite.models import ChatMessage

_APOS = "['\u2019]"  # straight or curly apostrophe

_CLAIM = re.compile(
    rf"""\b(?:
        I(?:{_APOS}ve|\s+have)?\s+(?:just\s+|already\s+)?
            (?:searched|checked|looked\s+(?:up|through|in(?:to)?)|queried|pulled\s+up
              |went\s+through|scanned)
      | my\s+(?:\w+\s+)?(?:search|lookup|query)\s+
            (?:didn{_APOS}t|did\s+not|found|returned|came\s+back|turned\s+up|shows?|showed)
    )\b""",
    re.IGNORECASE | re.VERBOSE,
)


def find_tool_claim(text: str) -> str | None:
    """Return the first phrase in ``text`` that claims a tool action.

    Args:
        text: A model reply.

    Returns:
        The matched phrase exactly as it appears in ``text`` (for quoting in a message), or
        ``None`` when the reply makes no such claim.
    """
    match = _CLAIM.search(text)
    return match.group(0) if match else None


def unbacked_history_claims(history: Sequence[ChatMessage]) -> list[tuple[int, str]]:
    """Find the assistant replies in a recorded history that claim a tool action never taken.

    Backing is decided per turn: a turn runs from one ``user`` message to the next, and every
    reply in a turn that holds a ``tool`` message or an ``assistant`` message with
    ``tool_calls`` -- before *or after* the reply -- is backed ("I searched your calendar."
    then the call is still that turn's action). A ``user`` message directly after a ``tool``
    row does not start a new turn: Claude delivers tool results inside a user turn, and text
    riding in that same turn normalises to ``tool`` rows followed by a ``user`` row -- still
    the answer to the call, not a fresh question.

    Only history recorded with its tool rows intact can be judged. One with no ``tool`` row
    and no ``tool_calls`` anywhere either never called a tool or had its tool rows stripped
    when the app resent it (or was rebuilt from earlier turns' text) -- undecidable, and
    judging it would flag every past claim -- so it yields nothing.

    Args:
        history: A case's recorded conversation prefix.

    Returns:
        ``(position, phrase)`` per unbacked claiming reply, in order. ``position`` is
        1-based within ``history``; ``phrase`` is quoted from the reply by
        :func:`find_tool_claim`.
    """
    # Pass 1: number each message's turn and note which turns acted.
    turns: list[int] = []
    backed_turns: set[int] = set()
    turn = 0
    previous_role: str | None = None
    for message in history:
        if message.role == "user" and previous_role != "tool":
            turn += 1
        elif message.role == "tool" or (message.role == "assistant" and message.tool_calls):
            backed_turns.add(turn)
        turns.append(turn)
        previous_role = message.role
    if not backed_turns:
        return []
    # Pass 2: a claiming reply in a turn that never acted is unbacked.
    claims: list[tuple[int, str]] = []
    for position, (message, message_turn) in enumerate(zip(history, turns, strict=True), start=1):
        if message.role == "assistant" and message.content and message_turn not in backed_turns:
            phrase = find_tool_claim(message.content)
            if phrase is not None:
                claims.append((position, phrase))
    return claims
