# Reproducibility checklist

The CPU foundation has no runtime dependencies. Inference requirements are intentionally
unlocked until hardware compatibility is tested. They are not a validated experimental
environment. Consult the official model resources linked in README.

Before scoring, commit a preregistration record with:

- Immutable model/CLIP/processor revisions, dataset revision and subset/pilot hashes.
- Python, torch, transformers, decoder, CUDA/driver versions, GPU, precision, quantization,
  deterministic settings and exact environment lock with wheel/index provenance.
- Processor pixel budget, resize/pad policy, timestamp-to-frame convention and duplicate rule.
- Prompts/hashes, action JSON schema, greedy parameters and generation limits.
- Call/retry bounds, failure handling, warmups, cache policy and latency boundaries.
- Bootstrap settings, pairing unit, exclusions, pilot overlap and primary/secondary tests.

Initial configs contain nulls for unknown values. A future scored runner must reject incomplete
preregistration. The current config checker checks proposal constants only.

Save immutable per-question traces and metadata under ignored runs/. Connect reports to
predictions and source commit by hashes. Publish derived results only after verifying
provenance and redistribution terms. Record failed runs and rerun reasons. Never substitute
zero for unavailable measurements. Both backbones require real compatibility smoke tests;
no GPU requirement, latency, compatibility or accuracy is asserted by this foundation.
