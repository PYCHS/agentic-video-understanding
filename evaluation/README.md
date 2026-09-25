# Evaluation protocol (planned; not yet frozen)

## Data and controls

Target 100 visually answerable Video-MME questions: 50 medium and 50 long when available.
Two reviewers independently assess silent-visual answerability and brief-event dependence
(decisive evidence within a contiguous interval <=10 seconds), reconcile disagreements,
and report Cohen's kappa before reconciliation. Freeze IDs and exclusion reasons before
comparing methods. See [screening](../docs/data_screening.md).

No audio/subtitles. Match checkpoint, processor, resolution, image-text format, answer prompt,
options and greedy decoding. Version agent-specific tool instructions separately and document
that necessary policy difference. Pair every method on the same frozen IDs. Count invalid
predictions as incorrect and report failure counts; never silently drop failed samples.
Infrastructure reruns require a documented, uniformly applied policy fixed before scoring.

## Comparisons and metrics

Primary: Adaptive-24 minus Uniform-24 accuracy. Secondary: Uniform-8, Uniform-16, fixed
CLIP 8+16 and adaptive always-use-24. Descriptive strata: duration, brief-event label,
stop reason and failure mode. Report denominators and limitations of small subgroups.

Report accuracy; mean/median frames; calls; input/output tokens; processed visual tokens
summed across calls; mean/median/p95 model and end-to-end latency. Include CLIP/decoding
overhead, warmups, hardware and peak memory. Audit missed events, wrong intervals,
unsupported evidence, premature stopping, malformed JSON and decoder collisions.

## Paired statistical inference

For each frozen question i, d_i = correct_adaptive_i - correct_uniform24_i. Bootstrap paired
rows with replacement 10,000 times (initial default seed 128), computing the mean difference.
Use 2.5th/97.5th percentiles for a paired 95% CI; report percentage points. Replicate count
and seed are engineering defaults to freeze. When questions share videos, additionally
report a paired video-cluster bootstrap sensitivity CI and the number of independent videos.

Exact two-sided McNemar: b = adaptive correct / baseline wrong; c = adaptive wrong / baseline
correct. Under the null b ~ Binomial(b+c, 0.5). Use an exact two-sided binomial test; no
discordant pairs means p=1. Report b, c, n, p, effect and CI. Label secondary comparisons
exploratory or preregister multiplicity correction. Statistical implementation and validation
against trusted-library results remain pending.

Gain: CI lower bound >0. Comparable accuracy: lower bound >=-2 percentage points AND fewer
mean frames. The study may be inconclusive; lack of significance does not prove equivalence.

## Backbone sensitivity pilot

Freeze 20 question IDs independently of outcomes. Run Uniform-24 and Adaptive-24 using
Qwen2.5-VL-7B-Instruct. Match sampling conventions, prompts, decoding, preprocessing and
stopping rules; archive all realized frame sets. Adaptive selections can differ when the
policy backbone changes. The proposal's matched-frame-set wording needs clarification:
preregister whether it means matched sampling protocol or literal frame replay. A literal
replay experiment must be labeled separately from a fresh adaptive-policy run.
Compare effects within each backbone only; never pool predictions across models.

## Results

All results are pending. No scored manifest, predictions, statistical output or GPU benchmark
exists in the initial commit. Synthetic smoke output is not an experiment.
