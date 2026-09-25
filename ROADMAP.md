# Roadmap / TODO

Each item is a meaningful increment, not a daily quota. Work may take multiple days.

- [x] Proposal-grounded docs, configs, CPU sampler/controller, tests and CI configuration.
- [x] Reference decoder and 448px preprocessing: frame IDs, boundaries, short clips and collisions.
  Reader-to-controller integration and broader source-format checks remain part of the adaptive loop.
- [ ] Strict JSON parser and bounded error handling: reject invalid actions, count all calls.
- [ ] Qwen3-VL adapter: real local smoke trace, frozen revisions and token/latency accounting.
- [ ] Uniform/CLIP integrations: reproducible 8/16/24 and fixed 8+16 frame sets.
- [ ] Adaptive loop and always-24 ablation: replayable traces, caps and terminal failures.
- [ ] Dual-reviewer screening: labels, exclusions, kappa, target 100 questions and 20 pilot IDs.
- [ ] Paired statistics: ID alignment, missing/duplicate rejection, trusted-reference validation.
- [ ] GPU environment lock and preregistration: fresh rerun, no unknown required run fields.
- [ ] Main evaluation and failure analysis: raw traces support all reported outcomes and costs.
- [ ] Qwen2.5-VL pilot: frozen 20 questions, matched protocol, no pooled effects.
- [ ] Research report: supported/unsupported hypotheses, limitations, contributor roles.

Future automated work should select one ready item, validate it and commit only meaningful
changes. If blocked on hardware, data or human screening, record the blocker instead of
inventing progress. Scheduling is managed outside this repository.
