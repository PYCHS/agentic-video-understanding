# Counting model calls and recording failures

`DecisionCaller` wraps the backend interface, JSON parser and controller checks. Create one
caller for each question. Call `decide(messages, controller)` once after each observation.
It returns a checked decision without changing the real controller or reading more frames.
It validates against a copy, so rejected decisions cannot spend the observation budget.

The current policy is deliberately simple: one call per round, at most four calls, and no
retries. A backend exception, invalid JSON or rejected controller action ends the question.
An accepted answer also ends calls. Calling twice in the same round is blocked before the
backend runs. At the frame or round limit, the controller requires an answer; a malformed
answer is a failure, not a reason to make a fifth call. This zero-retry policy is an initial
implementation choice to freeze before evaluation, not an extra rule stated in the proposal.

Each attempted backend call has a record containing its index, observation round, whether
an answer was forced, status, error reason, raw response, reported token counts, backend
latency and measured wall time. Missing backend responses and measurements remain null.
The wall timer includes generation and checking, but not video decoding. Records live in
memory until `write_trace(Path(...))` writes a question-specific JSONL file. Existing files
are never overwritten. This is not crash-safe logging yet; the future runner needs run and
question IDs, prompt/config hashes and incremental persistence.

This wrapper does not build prompts, execute inspections, reconcile requested/decoded
timestamps, or load Qwen. The future agent loop must decode and validate a batch before
committing it to the controller. Stop the question if that downstream step fails; do not
create another caller to bypass the limit. Backend adapters must not hide internal retries.
Timeouts and cancellation also belong to backend/runner integration; a call-count limit
does not bound the duration of one backend call.

Tests use scripted responses only. They check normal answers, backend failures, invalid JSON,
low-confidence early answers, duplicate observations, same-round calls and terminal failures.
No test output is a model prediction or benchmark result.
