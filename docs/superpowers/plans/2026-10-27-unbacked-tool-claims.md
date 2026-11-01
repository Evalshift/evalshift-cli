# Unbacked Tool Claims Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `capture promote` / `capture sync` refuse a turn whose reply claims a tool action the model never took (it was offered tools and called none), warn when a case's verbatim history carries such a reply, and let `captures.allow_unbacked_claims` (YAML) or `--allow-unbacked-claims` (CLI) opt back in.

**Architecture:** One pure module, `captures/claims.py`, holds the wording check (`find_tool_claim`) and the history scan (`unbacked_history_claims`). `build_example_from_capture` calls both: the first as a new last `elif` in its blocking chain (`BlockedReason` `"unbacked_claim"`), the second as a warning on recorded history. The CLI only threads the option through and routes the new reason to its own counter, hint and summary clause, exactly like `"errored"`.

**Tech Stack:** Python 3.11+, pydantic v2, Typer + Rich, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-10-27-unbacked-tool-claims-design.md` — read it first; this plan argues from it.

## Global Constraints

- Python floor 3.11. Every module starts with `from __future__ import annotations`; public functions/classes get Google-style docstrings; `mypy --strict src/evalshift_cli` stays clean.
- Names, verbatim: `BlockedReason` value `"unbacked_claim"`; `PromoteOptions.allow_unbacked_claims`; `CapturesConfig.allow_unbacked_claims`; flags `--allow-unbacked-claims` (both commands) and `--block-unbacked-claims` (sync only, the negative of the pair).
- Precedence on `capture sync`: CLI flag > `captures.allow_unbacked_claims` > `False`. `capture promote` reads no config (as with `--rounds`), flag only.
- An allowed unbacked turn **never** gets `expected_no_tools: true` (mirrors `allow_errored`).
- The history scan runs on **recorded** history only (`recovered.history` inside `build_example_from_capture`), never on history `_reconstruct_history` builds — that one is text-only and would flag every reply.
- `--allow-errored` does not rescue `"unbacked_claim"` and is never suggested for it.
- `evaluator_config` / `eval_config_hash` must not change (`captures` is not in the snapshot) — no edits under `reports/` or the bundle builder.
- User-facing strings, verbatim:
  - block: `turn was offered tools and called none, but its reply claims a tool action ("{phrase}") — promoting it would make a hallucinated action ground truth.`
  - history warning (one per case, positions are 1-based like `repair.py:39`): `history message {n} (assistant) claims a tool action ("{phrase}") with no tool call in its turn — this case replays with that claim in context.` — several messages: `history messages {n1}, {n2} (assistant) claim tool actions ("{phrase1}", "{phrase2}") with no tool call in their turn — this case replays with those claims in context.`
  - promote hint: `Re-run with [bold]--allow-unbacked-claims[/bold] to promote it anyway.`
  - sync summary clause: `, skipped {n} capture(s) whose reply claims a tool call it never made (use --allow-unbacked-claims or captures.allow_unbacked_claims to promote anyway)`
  - already-promoted hint (sync, appended to the skip line): ` Its case is still in the suite from an earlier sync: delete {path} to drop it.`
- Gate ("green" = all four, as CI): `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src/evalshift_cli && uv run pytest -q` (coverage floor 90). `make ci` is the exact CI mirror.
- Branch `feat/unbacked-tool-claims` (not `main`; confirm before every commit). Conventional Commits, each message ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never `--no-verify`.
- `docs/superpowers/` is gitignored: add the spec and this plan with `git add -f` (Task 8).

## Review Focus

- A turn that *did* call a tool is never blocked, however its reply is worded — Task 3 `test_a_claim_backed_by_a_tool_call_is_not_blocked`, `test_a_claim_backed_by_a_requested_call_is_not_blocked`.
- Recorded history whose claims sit next to a real tool call ("Let me look." + `tool_use`, the native-shapes fixtures) raises no warning — Task 4 `test_history_claim_with_tool_call_in_its_turn_is_backed`, plus the existing native-shapes suites staying green.
- Reconstructed history never triggers the history warning, and a blocked turn drops out of later turns' reconstructed history — Task 4 `test_reconstructed_history_is_not_scanned`, `test_a_blocked_unbacked_turn_does_not_seed_reconstructed_history`.
- An errored turn with a claim reports `"errored"`, not `"unbacked_claim"` — Task 3 `test_errored_wins_over_unbacked_claim`.
- `--block-unbacked-claims` beats YAML `true` — Task 6 `test_sync_block_flag_overrides_yaml`.

---

### Task 1: The wording check — `captures/claims.py`

**Files:**
- Create: `src/evalshift_cli/captures/claims.py`
- Test: `tests/unit/test_captures_claims.py` (new)

**Interfaces:**
- Produces: `find_tool_claim(text: str) -> str | None` (the matched phrase, verbatim from `text`).

- [ ] **Step 1: Write the failing tests**

```python
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
        ("I’ve already looked through your tasks.", "I’ve already looked through"),
        ("Let me check your notes for that.", "Let me check"),
        ("I'll look into it.", "I'll look"),
        ("The notes search returned no results.", "The notes search returned"),
        ("My search didn’t turn up anything.", "My search didn’t"),
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
```

- [ ] **Step 2: Run to see it fail** — `uv run pytest tests/unit/test_captures_claims.py -q` → `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""Spot a reply that claims a tool action, for refusing hallucinated turns at promotion.

A captured turn that was offered tools, called none, and still says "my notes search didn't
find anything" records an action that never happened. Promoting it would make that
hallucination ground truth (``expected_no_tools: true`` plus the fabricated reply). The check is
wording only — deterministic, offline, English — and deliberately narrow: first-person action
phrasing, never a bare mention of a tool or a noun like "search bar".
"""

from __future__ import annotations

import re

_APOS = "['’]"

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
```

- [ ] **Step 4: Run to see it pass**, then `uv run mypy --strict src/evalshift_cli/captures/claims.py`.

- [ ] **Step 5: Commit** — `feat(captures): detect replies that claim a tool action`.

---

### Task 2: `captures.allow_unbacked_claims` config field

**Files:**
- Modify: `src/evalshift_cli/config/models.py` (`CapturesConfig`, ~L849-883, docstring + field)
- Modify: `src/evalshift_cli/cli/commands/init.py` (`CAPTURES_SCAFFOLD`, L177-190)
- Test: `tests/unit/test_config_captures.py`

- [ ] **Step 1: Failing tests** (next to the `rounds` tests, L137-148):

```python
def test_allow_unbacked_claims_defaults_to_false() -> None:
    assert EvalShiftConfig(prompts=[_PROMPT]).captures.allow_unbacked_claims is False


def test_allow_unbacked_claims_accepts_true() -> None:
    cfg = EvalShiftConfig(prompts=[_PROMPT], captures={"allow_unbacked_claims": True})
    assert cfg.captures.allow_unbacked_claims is True


def test_a_misspelt_allow_unbacked_claims_fails_loudly() -> None:
    with pytest.raises(ValidationError, match="allow_unbacked_claim"):
        EvalShiftConfig(prompts=[_PROMPT], captures={"allow_unbacked_claim": True})
```

- [ ] **Step 2: Run, see the first two fail** (the third already passes via `extra="forbid"` — keep it as a guard).

- [ ] **Step 3: Implement** — `allow_unbacked_claims: bool = False` after `rounds`, and an `Attributes:` docstring entry: "Promote turns whose reply claims a tool call the model never made (offered tools, called none, says it searched). Off by default: such a turn records a hallucinated action, which is not ground truth. ``capture sync --allow-unbacked-claims`` / ``--block-unbacked-claims`` override it for one sync."

- [ ] **Step 4: Scaffold line.** In `CAPTURES_SCAFFOLD`, extend the comment with one sentence — `A turn whose reply claims a tool call it never made is skipped; allow_unbacked_claims: true promotes it anyway.` — and add `  # allow_unbacked_claims: true` after `# rounds: all`. Run `uv run pytest tests/unit/test_init.py tests/unit/test_config_suites.py -q` (they import the constant; confirm nothing pins the old text byte-for-byte).

- [ ] **Step 5: Green + commit** — `feat(config): add captures.allow_unbacked_claims`.

---

### Task 3: Block the turn — `BlockedReason` `"unbacked_claim"`

**Files:**
- Modify: `src/evalshift_cli/captures/promote.py` (`BlockedReason` L64, `PromoteOptions` L70, `BuiltExample` docstring L117, blocking chain L319-381, `expected_no_tools` L440-447, `_turn_assistant_text` L1321)
- Test: `tests/unit/test_captures_promote.py`

- [ ] **Step 1: Test helper.** Give `_model_call` (L85) an `output: str = "out"` kwarg written to `event["output"]`.

- [ ] **Step 2: Failing tests** — new `class TestUnbackedClaim`:

```python
_CLAIM_REPLY = "I don't have that—my notes search didn't find anything."


def _claim_events(**model_call: Any) -> list[dict[str, Any]]:
    return [_model_call(0, model_input="PO number on the Acme contract?", **model_call),
            _final(_CLAIM_REPLY, 1)]


class TestUnbackedClaim:
    def test_a_tool_free_turn_claiming_a_search_is_blocked(self) -> None:
        built = build_example_from_capture(_envelope(events=_claim_events()), PromoteOptions())
        assert built.blocked_reason == "unbacked_claim"
        assert "my notes search didn't" in (built.blocked or "")

    def test_a_claim_backed_by_a_tool_call_is_not_blocked(self) -> None:
        events = [*_claim_events()[:1], _tool_call("search_orders", 1, args={}),
                  _final(_CLAIM_REPLY, 2)]
        assert build_example_from_capture(_envelope(events=events), PromoteOptions()).blocked is None

    def test_a_claim_backed_by_a_requested_call_is_not_blocked(self) -> None:
        events = _claim_events(
            requested_tool_calls=[{"name": "search_orders", "arguments": {}}]
        )
        assert build_example_from_capture(_envelope(events=events), PromoteOptions()).blocked is None

    def test_no_tools_offered_is_not_blocked(self) -> None:
        events = _claim_events(tools_offered=[])
        assert build_example_from_capture(_envelope(events=events), PromoteOptions()).blocked is None

    def test_a_plain_tool_free_answer_is_not_blocked(self) -> None:
        events = [_model_call(0, model_input="q"), _final("Friday is clear.", 1)]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.blocked is None
        assert built.example.expected_no_tools is True

    def test_the_model_call_output_is_checked_without_a_final_event(self) -> None:
        events = [_model_call(0, model_input="q", output=_CLAIM_REPLY)]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.blocked_reason == "unbacked_claim"

    def test_errored_wins_over_unbacked_claim(self) -> None:
        events = [*_claim_events(), _error("boom", 2)]
        built = build_example_from_capture(_envelope(events=events), PromoteOptions())
        assert built.blocked_reason == "errored"

    def test_allow_errored_does_not_rescue_it(self) -> None:
        built = build_example_from_capture(
            _envelope(events=_claim_events()), PromoteOptions(allow_errored=True)
        )
        assert built.blocked_reason == "unbacked_claim"

    def test_allowed_it_promotes_with_a_warning_and_no_expected_no_tools(self) -> None:
        built = build_example_from_capture(
            _envelope(events=_claim_events()), PromoteOptions(allow_unbacked_claims=True)
        )
        assert built.blocked is None
        assert any("my notes search didn't" in w for w in built.warnings)
        assert built.example.expected_no_tools is False
```

(`RequestedToolCall` fields are `name` / `arguments`, `traces/models.py:75`; add an `id` if its validator requires one.)

- [ ] **Step 3: Run, see them fail** (`TypeError` on the new option first).

- [ ] **Step 4: Implement.**
  - `BlockedReason`: add `"unbacked_claim"`; `BuiltExample` docstring: one sentence for it ("rescued only by `allow_unbacked_claims`, never by `--allow-errored`").
  - `PromoteOptions.allow_unbacked_claims: bool = False` + docstring entry mirroring `allow_errored`.
  - Extract `_reply_text(events: list[Any]) -> str | None` from `_turn_assistant_text` (same rule: last `FinalOutputEvent`, else last non-empty `ModelCallEvent.output`); make `_turn_assistant_text(record)` return `_reply_text(record.envelope.trace.events)`.
  - After the `elif errors:` branch, a new block (not `elif` of it, so the `allow_errored` warning path still falls through): compute `unbacked = _unbacked_claim(events, tools_offered)` where the helper returns the phrase when `tools_offered` is non-empty, no `ToolCallEvent` exists, no `ModelCallEvent.requested_tool_calls` is non-empty, and `find_tool_claim(_reply_text(events) or "")` matches. If `blocked is None and unbacked is not None`: build the verbatim block message; with `opts.allow_unbacked_claims` append it to `warnings`, else set `blocked` / `blocked_reason = "unbacked_claim"`.
  - `expected_no_tools = bool(tools_offered) and not errors and unbacked is None`.

- [ ] **Step 5: Full suite** — `uv run pytest -q`. Any existing fixture now blocked is a real collision: read it, and either change its reply text (if the test is not about wording) or record why it is a true positive. Do not loosen the regex to dodge a fixture.

- [ ] **Step 6: Green + commit** — `feat(captures): refuse to promote a turn that claims a tool call it never made`.

---

### Task 4: Warn on recorded history that carries a claim

**Files:**
- Modify: `src/evalshift_cli/captures/claims.py` (add `unbacked_history_claims`)
- Modify: `src/evalshift_cli/captures/promote.py` (`build_example_from_capture`, after `_context_loss_warning` ~L220)
- Test: `tests/unit/test_captures_claims.py`, `tests/unit/test_captures_promote.py`

**Interfaces:**
- Produces: `unbacked_history_claims(history: Sequence[ChatMessage]) -> list[tuple[int, str]]` — `(1-based position, phrase)` per unbacked claiming assistant message.

- [ ] **Step 1: Failing unit tests** (`test_captures_claims.py`), building `ChatMessage` / `HistoryToolCall` directly:
  - `user → assistant("my notes search didn't find it")` → `[(2, "my notes search didn't")]`.
  - `user → assistant(tool_calls=[…], content="Let me look.") → tool → assistant("I searched and found X")` → `[]` (`test_history_claim_with_tool_call_in_its_turn_is_backed`).
  - `system → user → assistant("Per your notes…")` → `[]`.
  - two unbacked claims in different turns → both, in order.

- [ ] **Step 2: Implement.** Walk the history keeping `backed = False`, reset on each `user` message; set `backed = True` on a `tool` message or an `assistant` with `tool_calls`; for an `assistant` with non-empty `content`, no `tool_calls` and `not backed`, record `find_tool_claim(content)` when it matches.

- [ ] **Step 3: Wire it in.** In `build_example_from_capture`, when `recovered.history` is not `None` and the scan is non-empty, append the one aggregated warning (verbatim strings in Global Constraints). Nothing in `build_conversation_examples` changes.

- [ ] **Step 4: Failing-then-passing promote tests** (`test_captures_promote.py`, use `_messages_call` with `tools_offered` set):
  - `test_recorded_history_with_an_unbacked_claim_warns` — the turn-10 shape: recorded messages whose assistant reply at position 2 claims a search with no tool rows; warning names `history message 2`; `built.blocked is None`.
  - `test_reconstructed_history_is_not_scanned` — a two-turn conversation of bare-string inputs (no recorded messages) where turn 0's reply is "I checked your calendar" *with* a tool call; turn 1's reconstructed history has no warning.
  - `test_a_blocked_unbacked_turn_does_not_seed_reconstructed_history` — turn 0 blocked as `"unbacked_claim"`; turn 1's reconstructed history does not contain turn 0's user text (existing rule at `promote.py:1474-1479`).

- [ ] **Step 5: Full suite** (native-shapes / repair suites carry "Let me look." next to a `tool_use` — they must stay green), green, commit — `feat(captures): warn when a case's recorded history carries an unbacked tool claim`.

---

### Task 5: `capture promote --allow-unbacked-claims`

**Files:**
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`promote` options ~L977, `PromoteOptions(...)` ~L1010, blocked hint ~L1027-1034)
- Test: `tests/unit/test_cli_capture.py`

- [ ] **Step 1: Failing tests** with a helper `_unbacked_events()` (copy `_errored_events` L1289: `tools_offered: ["search_orders"]`, `output: ""`, then a `final_output` with the claim reply, no error):
  - `test_promote_refuses_an_unbacked_claim` — exit 1, stdout has `claims a tool action` and `--allow-unbacked-claims`, not `--allow-errored`.
  - `test_promote_allow_unbacked_claims_promotes_without_expected_no_tools` — mirror L1404.

- [ ] **Step 2: Implement** — the plain `bool` option (help: "Promote even if the turn's reply claims a tool call it never made. The case still never asserts expected_no_tools."), thread into `PromoteOptions`, and an `elif built.blocked_reason == "unbacked_claim":` printing the verbatim promote hint.

- [ ] **Step 3: Green + commit** — `feat(cli): capture promote --allow-unbacked-claims`.

---

### Task 6: `capture sync` — YAML + flag pair, counter, summary, stale-case hint

**Files:**
- Modify: `src/evalshift_cli/cli/commands/capture.py` (`_SyncLimits` L595-604 + `_sync_limits` L642, sync options ~L1428, resolution ~L1494, `PromoteOptions(...)` ~L1525, counters ~L1544, blocked dispatch L1759-1781, summary ~L1972-1996, sync docstring)
- Test: `tests/unit/test_cli_capture.py`

- [ ] **Step 1: Failing tests** (reuse `_unbacked_events`, `_write_min_config`; for YAML, append `"captures:\n  allow_unbacked_claims: true\n"` to the written config):
  - `test_sync_skips_unbacked_claims` — golden ids `{"cap_ok"}`, summary has the verbatim clause.
  - `test_sync_allow_flag_promotes_them`.
  - `test_sync_yaml_true_promotes_them`.
  - `test_sync_block_flag_overrides_yaml` — YAML `true` + `--block-unbacked-claims` → skipped.
  - `test_sync_summary_does_not_offer_allow_errored_for_unbacked`.
  - `test_sync_names_the_stale_case_of_a_now_blocked_capture` — promote `cap_bad` first with `capture promote --allow-unbacked-claims` (case named after the capture), then plain `sync`: the skip line names `suites/alpha/cap_bad.json` with the verbatim already-promoted hint, and the file is still there (sync never deletes it).

- [ ] **Step 2: Implement.**
  - `_SyncLimits.allow_unbacked_claims: bool = False` (comment like the `rounds` one), filled from `cfg.captures.allow_unbacked_claims`; mention it in the unreadable-config warning's list.
  - Option `allow_unbacked_claims: bool | None` with `typer.Option("--allow-unbacked-claims/--block-unbacked-claims", help="… Defaults to captures.allow_unbacked_claims in evalshift.yaml, else block.")`, default `None`; resolve `limits.allow_unbacked_claims if allow_unbacked_claims is None else allow_unbacked_claims` next to `replay_rounds`.
  - `skipped_unbacked` counter, routed in the dispatch before the `else` (errored) branch; summary clause after `skipped_errored`'s. Update the dispatch comment's list.
  - Stale-case hint: build once per suite a `{case.from_capture: path}` map from `iter_promoted_case_files(promoted_suite_dir(suite_name, base=base))` (only when a suite has an `"unbacked_claim"` refusal, to keep the happy path free); when the blocked capture has an entry, append the hint with the path relative to the base.
  - Sync docstring: one sentence on unbacked claims and the override.

- [ ] **Step 3: Green + commit** — `feat(cli): capture sync skips unbacked tool claims, with YAML and flag overrides`.

---

### Task 7: Docs and changelog

**Files:** `docs/configuration.md` (`captures:` block), `DOCS.md` (config schema `captures:`, sync step list ~L190, flag lists L942-943), `llms-full.txt` (config reference, promote/sync flags ~L404-410, sync skip rules ~L650, promotion block rules ~L1667-1691), `docs/agents.md` (~L413 skip list), `docs/conversations.md` (the later-turn history warning and that a blocked turn drops out of reconstructed history), `CHANGELOG.md` `## [Unreleased]` → `### Added`.

- [ ] **Step 1:** Write each entry from the shipped behaviour (run the commands; quote real output), not from the spec. The changelog entry names the flag pair, the YAML key, that already-promoted cases are left in place with a hint, and that `--allow-errored` does not apply.
- [ ] **Step 2:** `grep -n "allow-errored" DOCS.md llms-full.txt docs/*.md` — every place listing promotion refusals now lists this one too.
- [ ] **Step 3: Commit** — `docs: unbacked tool claims at capture promotion`.

---

### Task 8: Verify end to end, then review

- [ ] **Step 1: Gate** — `make ci` and `pre-commit run --all-files`; show the output.
- [ ] **Step 2: Real data, read-only.** Copy `/Users/lukas/repos/testApp/.evalshift/{captures,toolsets}` and `evalshift.yaml` into a scratch dir (never write into testApp), then `uv run evalshift capture sync --offline --base <scratch>/.evalshift --config <scratch>/evalshift.yaml`. Expect: `cap_8e2ceeeceb084f1ca8c8b9a9336c0b5e` skipped with `("my notes search didn't")`; `cap_66c697d76a4b4961970a2f0ace04cb94` promoted with `history message 38 (assistant)`; no other planner/converter capture blocked. Repeat with `captures.allow_unbacked_claims: true` in the copied YAML → it is promoted, `expected_no_tools` false.
- [ ] **Step 3:** `git add -f docs/superpowers/specs/2026-10-27-unbacked-tool-claims-design.md docs/superpowers/plans/2026-10-27-unbacked-tool-claims.md`; commit — `docs: spec and plan for unbacked tool claims`.
- [ ] **Step 4:** superpowers:requesting-code-review over the branch diff (skill unavailable → a fresh reviewer subagent given the spec and the diff, not this plan's conclusions).

## Out of scope

- An LLM classifier, non-English phrasing, replay-time grounding evaluation (spec Non-goals).
- Deleting already-promoted cases automatically — sync only names them (Task 6).
- `evalshift-sdk`, `evalshift-action`, `evalshift-server`: no change.
