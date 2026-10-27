"""Spot a reply that claims a tool action, for refusing hallucinated turns at promotion.

A captured turn that was offered tools, called none, and still says "my notes search didn't
find anything" records an action that never happened. Promoting it would make that
hallucination ground truth (``expected_no_tools: true`` plus the fabricated reply). The check is
wording only — deterministic, offline, English — and deliberately narrow: first-person action
phrasing, never a bare mention of a tool or a noun like "search bar".
"""

from __future__ import annotations

import re

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
