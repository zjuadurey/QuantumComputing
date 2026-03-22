# Codex Review Contract

This repository uses a split workflow:

- Claude (or another coding agent) is the primary implementer.
- Codex is the reviewer and acceptance gate.
- The current task context lives in `docs/review_spec.md` attached to the review thread.

## Review Mode

When running `/review`, Codex should treat this repository as a code review task by default.
Focus on correctness, regressions, missing tests, and spec mismatches before style or refactoring advice.
If `docs/review_spec.md` is attached in the current thread, treat it as the task contract for this review.

## Context Priority

Read context in this order when needed:

1. The diff / changed files
2. `docs/review_spec.md` attached to the thread
3. Project specs and logs under `openqasm-rtir/docs/`
4. `openqasm-rtir/CLAUDE.md` for project background

Do not expand scope beyond the attached spec unless the diff clearly introduces new risk.

## Acceptance Priorities

Review against the following priorities, highest first:

1. Semantic correctness
2. Conformance to the task spec
3. Regression risk
4. Test adequacy
5. Simplicity and scope control

## Project-Specific Acceptance Rules

For `openqasm-rtir/`, pay extra attention to:

- Formal definitions are the source of truth; code must not silently contradict `openqasm-rtir/docs/formal_definitions_v0.md`
- Checker and oracle must stay meaningfully independent
- Timing semantics must be explicit and monotonic
- Port occupancy, feedback readiness, and frame/phase evolution must be auditable from code and tests
- Scope should stay MVP-first; avoid speculative abstractions unless the spec explicitly asks for them

## Pulse-Level Review Checklist

If a change touches pulse code, verify:

- `Play`, `Acquire`, `ShiftPhase`, `Delay`, and conditional behavior match the documented semantics
- Port conflicts are checked independently from the reference semantics path
- Feedback checks use the right readiness time for classical bits
- Phase accumulation and elapsed-time handling are correct
- Positive and negative examples both exist when behavior changes

## Testing Expectations

A review should flag missing validation when a change:

- modifies semantics without updating tests
- adds a new case without a focused unit or regression test
- changes a spec assumption without updating docs or examples

Passing tests do not override a clear semantic or specification bug.

## Review Output

Findings come first.
Each finding should be concrete, severity-ordered, and tied to file references when possible.
Prefer reporting:

- incorrect behavior
- spec mismatch
- edge-case breakage
- missing regression coverage

Avoid low-value style nits unless they hide a real maintenance or correctness risk.

## Communication

- Respond in Chinese unless the user asks otherwise.
- Keep code identifiers and file names in English.
- Be direct about blockers and uncertainty.
