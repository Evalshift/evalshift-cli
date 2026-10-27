"""Spot a reply that claims a tool action, for refusing hallucinated turns at promotion.

A captured turn that was offered tools, called none, and still says "my notes search didn't
find anything" records an action that never happened. Promoting it would make that
hallucination ground truth (``expected_no_tools: true`` plus the fabricated reply). The check is
wording only — deterministic, offline, English — and deliberately narrow: first-person action
phrasing, never a bare mention of a tool or a noun like "search bar".
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
      | (?:my|the)\s+(?:\w+\s+)?(?:search|lookup|query)\s+
            (?:didn{_APOS}t|did\s+not|found|returned|came\s+back|turned\s+up|shows?|showed)
      | let\s+me\s+(?:check|search|look|pull\s+up|find)
      | I{_APOS}ll\s+(?:check|search|look)
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

    A reply is *backed* when a ``tool`` message or an ``assistant`` message with
    ``tool_calls`` sits between it and the user turn it answers. A ``user`` message starts a
    new turn, except one directly after a ``tool`` row: Claude delivers tool results inside
    a user turn, and text riding in that same turn normalises to ``tool`` rows followed by a
    ``user`` row -- still the answer to the call, not a fresh question.

    Only meaningful for history recorded with its tool rows intact. History rebuilt from
    earlier turns' text carries no tool rows, so every past reply would look unbacked.

    Args:
        history: A case's recorded conversation prefix.

    Returns:
        ``(position, phrase)`` per unbacked claiming reply, in order. ``position`` is
        1-based within ``history``; ``phrase`` is quoted from the reply by
        :func:`find_tool_claim`.
    """
    claims: list[tuple[int, str]] = []
    backed = False
    previous_role: str | None = None
    for position, message in enumerate(history, start=1):
        if message.role == "user" and previous_role != "tool":
            backed = False
        elif message.role == "tool" or (message.role == "assistant" and message.tool_calls):
            backed = True
        elif message.role == "assistant" and message.content and not backed:
            phrase = find_tool_claim(message.content)
            if phrase is not None:
                claims.append((position, phrase))
        previous_role = message.role
    return claims
