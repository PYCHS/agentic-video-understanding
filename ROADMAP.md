# Roadmap / TODO

Each item is a meaningful increment, not a daily quota. Work may take multiple days.

- [x] Proposal-grounded docs, configs, CPU sampler/controller, tests and CI configuration.
- [x] Reference decoder and 448px preprocessing: frame IDs, boundaries, short clips and collisions.
  Broader source-format checks remain pending.
- [x] Observation session: decode before committing budget; use actual times for evidence.
- [x] Strict JSON parser: reject malformed fields, invalid intervals and unseen evidence.
- [x] Model-call wrapper: record rejections, count attempts, enforce no retries and terminal failures.
  Timeouts and crash-safe trace persistence remain pending.
- [x] Question runner: survey, inspect/answer loop and failure handling with scripted-backend tests.
  Real backend and durable run metadata remain pending.
- [x] Draft image/text prompt builder: actual timestamps, budget state and decision history.
  Qwen processor compatibility and frozen baseline-matched prompts remain pending.
- [x] Qwen3-VL adapter interface: explicit revisions/settings, greedy decoding and token accounting.
- [x] Call-cost summary validation: reject invalid measurements and preserve missing-value coverage.
- [ ] Real Qwen smoke test: compatible environment, pinned weights/processor and GPU validation.
- [ ] Uniform/CLIP integrations: reproducible 8/16/24 and fixed 8+16 frame sets.
- [ ] Adaptive loop and always-24 ablation: replayable traces, caps and terminal failures.
- [ ] Dual-reviewer screening: labels, exclusions, kappa, target 100 questions and 20 pilot IDs.
- [x] Prediction pairing: manifest-order ID alignment, missing/duplicate rejection, failures retained.
- [ ] Paired statistics: trusted-reference validation and frozen-run provenance checks.
- [ ] GPU environment lock and preregistration: fresh rerun, no unknown required run fields.
- [ ] Main evaluation and failure analysis: raw traces support all reported outcomes and costs.
- [ ] Qwen2.5-VL pilot: frozen 20 questions, matched protocol, no pooled effects.
- [ ] Research report: supported/unsupported hypotheses, limitations, contributor roles.

Future automated work should select one ready item, validate it and commit only meaningful
changes. If blocked on hardware, data or human screening, record the blocker instead of
inventing progress. Scheduling is managed outside this repository.
