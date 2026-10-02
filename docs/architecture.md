# Architecture and implementation boundary

## Present foundation

`sampling.py` implements midpoint sampling and the fixed baseline's selection from supplied
embeddings. `controller.py` enforces timestamps, batches, budget, rounds and stopping.
`backend.py` defines a future adapter interface; `tracing.py` writes JSONL records.
`video.py` reads selected frames from local videos and checks decoded-frame collisions.
`actions.py` checks model JSON, action fields and observed evidence without mutating state.
`calls.py` limits backend attempts and records responses/rejections under a zero-retry policy.
`runner.py` connects observations and decisions with an injected message builder/backend.
There is no live model adapter, frozen multimodal prompt or benchmark runner yet.
The [observation session](observation_session.md) connects the reader/controller with batch
rollback and actual-time evidence. See [runner](runner.md) for the tested loop and its limits.
See [model calls](model_calls.md) for the current limits and logging boundary.
See [action format](actions.md) for the parser contract and remaining integration work.
See [video reader](video_reader.md) for its timestamp convention and limits.

## Planned execution

1. Return only requested frames. Map them to stable decoded frame IDs; reject batches
   that collide after decoding. Never silently duplicate frames or reduce batch size.
2. Resize to a 448-pixel long side preserving aspect ratio. The long side is divisible by 32;
   padding/processor alignment of the short side must be frozen and logged before evaluation.
3. Supply explicit `[t = ... s]` labels with image-text pairs, fixed checkpoint/processor,
   pixel budget, quantization mode and greedy decoding.
4. Parse strict JSON: candidate option, supporting timestamps, missing evidence, confidence,
   and action. Inspect actions specify start/end/k; answer actions supply option/evidence.
   Treat model output as untrusted and validate it before executing requests.
5. The initial survey is round 1. Each successful inspection adds one observation/decision
   round. At round 4 or 24 unique frames, the current decision must answer. No fifth round
   or hidden repair call is allowed; invalid terminal JSON is a logged failure.
6. Reject invalid intervals, duplicates and over-budget requests without mutating state.
   The call wrapper currently stops on errors with no retries and records every attempted
   backend call, including failures. Freeze this choice before evaluation.

## Engineering interpretations requiring preregistration

Midpoint timestamps and a 1-microsecond duplicate tolerance are initial choices, not explicit
proposal requirements. The reader additionally compares actual frame IDs. Too-short clips
need a predeclared exclusion/failure rule. The session now handles reader/controller integration.
Call accounting currently covers the backend interface, not a live model adapter.

Evidence sufficiency is structural: membership does not establish that an image truly supports
an answer. Forced answers retain low confidence or unresolved evidence; empty evidence is
allowed at a forced limit rather than manufactured.

The always-use-24 ablation disables early stopping. Batch choices that make 24 frames
unreachable within four rounds are rejected. This feasibility rule is an implementation
interpretation to review before preregistration.

## Fixed coarse-to-fine baseline

Encode 8 coarse frames using frozen CLIP ViT-B/32, normalize embeddings, compute adjacent
cosine distances, choose the largest-distance interval, and add 16 interior uniform frames.
Ties select the earliest interval. No question-conditioned CLIP ranking or adaptive replanning.
The 16-frame batch belongs to this baseline, not the agent tool.

## Logging contract

Per call: run/question/video ID, method, backbone/revision, raw response/action, validation
outcome, round, requested and resolved timestamps/frame IDs, cumulative unique frames,
input/output/visual tokens, VLM calls, synchronized latency, and failure status.
Per run: wall latency, decode/CLIP time, peak memory, hardware/software versions,
prompt/config/manifest hashes, and cache policy. Extend the initial RoundTrace at integration.

Repeated context images do not add unique observations, but their processed visual tokens
and latency count on every call. Missing measurements remain null with a reason. CLIP
compute belongs in baseline cost accounting. Separate model from total pipeline latency;
frame savings do not establish wall-clock speedups.
