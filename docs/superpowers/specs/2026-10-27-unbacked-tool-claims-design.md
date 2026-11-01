# Unbacked tool claims: block hallucinated turns at promotion

**Date:** 2026-10-27 · **Status:** decisions approved in chat, spec for maintainer review · **Repos:** evalshift-cli (behaviour + docs); evalshift-sdk, evalshift-action, evalshift-server (nothing)

> **Amended after review (2026-10-28).** Review found false positives on exactly the turns
> where calling no tool is correct. Changes, folded into the sections below: promises (`let me
> …`, `I'll …`) and `the <x> search/query …` no longer count as claims (§1); a tool result, a
> retrieval event or a second model call also backs a turn (§2); history backing is decided per
> turn, forward as well as backward, and a recorded history with no tool rows at all is not
> scanned (§3); a blank `final_output` falls back to the model-call text (§2); `--repair` names
> `--allow-unbacked-claims` when it can't repair such a case, and the stale-case hint skips
> reviewed cases (§5).

## Problem

A captured turn can record a reply that *claims* a tool action the model never took. Found in
testApp's planner suite (`r_20261027_planner_47f8ef`), conversation
`planner:dee6c8bd-…`, turn 9:

- user: "What's the PO number on the Acme contract?"
- production `claude-sonnet-5`, offered `get_schedule` / `get_tasks` / `search_notes`, made **one**
  model call with `requested_tool_calls: []` and replied "I don't have that information—**my notes
  search didn't find anything** specific…"

`capture sync` promoted it as-is, so today:

1. **The hallucination becomes ground truth.** The case gets `expected_no_tools: true`
   (`captures/promote.py:440-447`) and the fabricated reply as `expected.final_output`. A
   candidate that correctly calls `search_notes` fails `tool_selection` conformance. On replay
   both models *did* search — the production turn was a one-off.
2. **It poisons later turns silently.** Turn 10's verbatim history carries the fabricated reply
   as message 38. The target model, seeing an unanswered question and an unverified "I searched",
   re-ran `search_notes("Acme PO")` on turn 10, which surfaced in the report as a puzzling
   off-topic call and a 0.5 routing score — with nothing pointing at the cause.

No evaluator catches this either: `llm_judge` never sees the turn's own tool calls
(`evaluators/llm_judge.py:113`), and no `trace_invariants` rule type can condition on reply text.

## Decisions (approved in chat)

| Question | Decision |
| --- | --- |
| Flagged turn | **Block** like an errored turn, with an opt-in `--allow-unbacked-claims` override. |
| Persistent setting | **`captures.allow_unbacked_claims`** in `evalshift.yaml`, overridable per sync. |
| Later turns carrying it in history | **Warn**, naming the history position; still promote. |
| Detection | **Deterministic wording check only.** No LLM, no cost, offline. |

## Design

### 1. Detector — new `captures/claims.py`

```python
def find_tool_claim(text: str) -> str | None:
    """Return the first phrase in ``text`` that claims a tool action, else ``None``."""
```

One compiled, case-insensitive regex over first-person claims that an action already happened.
Validated against testApp's three tool-free captures (1 true positive, 0 false positives):

- past action: `I('ve| have)? (just |already )?(searched|checked|looked (up|through|in|into)|queried|pulled up|went through|scanned)`
- reported result: `my (<word> )?(search|lookup|query) (didn't|did not|found|returned|came back|turned up|shows?|showed)`

*Amended:* the original set also matched promises (`let me (check|search|look|pull up|find)`,
`I'll (check|search|look)`) and `the <x> search/query …`. Both were dropped: a promise is not a
claim that an action happened, and "let me know the date and I'll check your calendar" is a
clarifying question — a correct no-tool turn; "the query didn't include a WHERE clause" is about
the user's query. Known residual false positive, accepted: past-tense review of pasted content
("I checked the code you pasted") — the flag rescues it.

Returns the matched phrase so messages can quote it. English only (non-goal below).

### 2. Current turn — new `BlockedReason` `"unbacked_claim"`

In `build_example_from_capture`, as the **last** `elif` of the blocking chain (after `errors`, so
an errored turn keeps reporting as errored). It fires when all hold:

- the first model call offered tools (`tools_offered` non-empty);
- the turn recorded no `tool_call` events and no model call has `requested_tool_calls`;
- *(amended)* the turn recorded no `tool_result` and no `retrieval` event (a RAG answer did
  search), and only one model call — a second one happens only after something was fed back,
  the only trace of an agent loop in a pre-2.1.0 capture with no `requested_tool_calls`;
- `find_tool_claim(<turn's reply>)` matches, where the reply is the text
  `_turn_assistant_text` already selects (last `FinalOutputEvent`, else last non-empty
  `ModelCallEvent.output`). *(Amended:* a blank final text falls back to the model-call text;
  with nothing to fall back to, the blank final still counts — reconstruction is unchanged
  there.)

Message (exact wording settled in implementation):

> turn was offered tools and called none, but its reply claims one ("my notes search didn't") —
> promoting it would make a hallucinated action ground truth.

`PromoteOptions.allow_unbacked_claims: bool = False`. With it set, the note becomes a warning,
and — mirroring `allow_errored` — the case **never gets `expected_no_tools: true`**: a turn that
claimed a search is not evidence that calling nothing was right.

### 3. Later turns — warning on verbatim history

Also in `build_example_from_capture`, when `recovered.history` is not `None` (the capture's own
recorded messages, which keep `tool_calls` / `tool` rows): for each assistant message with text
and no `tool_calls`, it is *backed* if any `tool` message or assistant-with-`tool_calls` sits
between it and the preceding user message. An unbacked one whose text matches `find_tool_claim`
adds one aggregated warning:

*Amended:* backing is per turn (one user message to the next, a user row right after a tool row
not starting one), forward as well as backward — "I searched your calendar." followed by the
call in the same turn is backed. Two passes: mark turns that hold a tool row or `tool_calls`,
then scan. A recorded history with no tool row and no `tool_calls` anywhere is not scanned at
all: the app may strip tool rows when it resends history, making backing undecidable.


> history message 38 (assistant) claims a tool action ("my notes search didn't") with no tool
> call in its turn — this case replays with that claim in context.

**Reconstructed history needs nothing new**: it is text-only, so the structural scan would flag
every past reply — it must *not* run there. It is already safe: a blocked turn never seeds later
turns' reconstructed history (`promote.py:1474-1479`), so blocking turn 9 removes it.

### 4. Config — `captures.allow_unbacked_claims`

New field on `CapturesConfig` (`config/models.py:849`), following the `captures.rounds`
precedent (a sync knob that lives in YAML with a per-run CLI override):

```yaml
captures:
  # Promote turns whose reply claims a tool call the model never made. Off by default:
  # such a turn records a hallucinated action, which is not ground truth.
  allow_unbacked_claims: false
```

- `allow_unbacked_claims: bool = False`; `extra="forbid"` already rejects typos.
- Read by `capture sync` only, alongside `rounds` in the limits read (`capture.py:603-642`).
  `capture promote` does not read `evalshift.yaml` today (same as `--rounds`), so it stays
  flag-only.
- **Hosted / hash impact: none.** `captures` is not part of the `evaluator_config` snapshot
  (`../evalshift-server/BUNDLE_SPEC.md#evaluator-config`), so bundles and `eval_config_hash`
  are unchanged.

### 5. CLI

- `capture sync`: `--allow-unbacked-claims/--block-unbacked-claims`, default `None` = use
  `captures.allow_unbacked_claims` (else `false`); either flag overrides it for one sync — the
  `--rounds` resolution pattern (`capture.py:1494`). The pair lets a sync re-check a project that
  set `true` in YAML.
- `capture promote`: `--allow-unbacked-claims` (plain bool, help text parallel to
  `--allow-errored`). On `blocked_reason == "unbacked_claim"`, print "Re-run with
  `--allow-unbacked-claims` to promote it anyway." (`capture.py:1032` pattern).
- *(Amended)* `capture sync --repair`: a repair candidate whose rebuild is refused as
  `unbacked_claim` prints `can't repair <case>: … Re-run with --allow-unbacked-claims to repair
  it anyway.` (the flag unblocks the rebuild, so the repair runs). Without `--repair` it is
  still counted in `left N case(s) for --repair`, as an errored rebuild is. The stale-case hint
  below skips cases marked `provenance: reviewed`.
- `capture sync` summary: new `skipped_unbacked` counter in the `blocked_reason` dispatch
  (`capture.py:1771`) and clause `skipped N capture(s) whose reply claims a tool call it never
  made (use --allow-unbacked-claims or captures.allow_unbacked_claims to promote anyway)`.
  `--allow-errored` does not apply.

### 6. Docs (same change)

`docs/configuration.md` (`captures:` block — required for a new config field), `DOCS.md`
(config schema `captures:` entry, sync step list ~L190, flag lists L942-943), `llms-full.txt`
(config reference, promote/sync flags
~L404-410, sync skip rules ~L650, promotion block rules ~L1667-1691), `docs/agents.md` (~L413
skip list), `docs/conversations.md` (later-turn history warning), `CHANGELOG.md` `[Unreleased]`.

## Tests (TDD, `tests/unit/`)

- `test_captures_claims.py` (new): positives — "my notes search didn't find anything", "I
  searched your notes", "I've checked your calendar"; negatives — *(amended)* "Let me check
  your notes for that.", "I'll look into it.", "The notes search returned no results.", "Per
  your notes from Aug 20…", "Your passport renewal is due by September 15",
  "I can't search contract documents", "Would you like me to check?", "the search bar",
  empty string.
- `test_captures_promote.py`: blocks with `"unbacked_claim"` on the turn-9 shape; not blocked
  when the turn called a tool, when no tools were offered, or when the reply makes no claim;
  errored + claim reports `"errored"`; `allow_unbacked_claims` promotes with a warning and
  `expected_no_tools` false; verbatim-history warning names the right position and does not fire
  for a claim backed by a tool call in its turn; reconstructed history drops the blocked turn and
  raises no history warning.
- CLI tests for both commands: exit 1 + hint on `promote`, summary clause on `sync`, flag rescues.
- Config (`tests/unit/test_config*.py`): `captures.allow_unbacked_claims` defaults to `false`,
  accepts `true`, a misspelt key fails loudly.
- Sync resolution: YAML `true` promotes; YAML `true` + `--block-unbacked-claims` blocks; YAML
  absent + `--allow-unbacked-claims` promotes.

## Non-goals / risks

- **False positives** — "As I checked earlier, …" referring to a real call in a prior turn would
  block. Accepted: the flag rescues it and the message quotes the phrase. Revisit if reports land.
- **Misses** — unusual phrasings and non-English replies pass. An opt-in LLM classifier was
  considered and declined for now.
- **Existing goldens stay.** Resolved while planning: sync's blocked check (`capture.py:1759`)
  runs before its already-promoted check (`:1794`), so a newly blocked capture is reported as
  skipped but its earlier case file is left in the suite, `--force` or not. Sync never deletes
  it; the skip line names the file to delete (plan Task 6).
- Replay-time grounding evaluation (judge given the turn's own calls/results) is a separate idea.
