# Running and reviewing an experiment

The batch runner connects the existing decoder, adaptive controller, Qwen adapter,
strict response parser and saved-result evaluation. It supports Adaptive-24,
Adaptive-always-24 and Uniform-8/16/24. Real Qwen weights and Video-MME have **not**
been run as part of this implementation. CLIP and the alternate pilot backbone
are not supported by this command yet.

## 1. Prepare inputs

Start from `examples/questions.json`. Each row needs `question_id`, `video_id`,
`video_path`, `question`, and an `options` object with A/B/C/D text. Video paths
resolve relative to the manifest. IDs must be unique per question; one video ID
must identify one local file. The template video is a placeholder, not bundled data.

Keep gold answers in a separate scoring manifest:

```json
[{"question_id":"your-question-001","video_id":"your-video-001","answer":"A"}]
```

If a source manifest also contains labels, the loader projects it onto the input
fields above; gold and other extra fields never enter the question object or
model prompts. Screen visual-only questions before freezing the input set.
The runner does not implement Video-MME downloading, screening or license checks.

## 2. Pin and validate settings

Copy `examples/qwen-settings.json` and replace **both** placeholder revisions with
real, full Hugging Face commit hashes. These placeholders intentionally fail
validation. Model identity is fixed to `Qwen/Qwen3-VL-8B-Instruct`.
The pixel and token settings are an initial configuration, not a validated GPU lock.

```sh
python -m pip install -e ".[video,test,stats-test]"
# Install an appropriate CUDA torch build and requirements/inference.txt separately.
python -m watch_wisely.cli --manifest /data/questions.json \
  --settings /data/settings.json --output runs/pilot-01 --run-id pilot-01 --validate-only
```

Preflight validates JSON fields, local file paths and settings. It does not decode
videos, check CUDA memory or assert compatibility with a live processor. It writes
nothing and works without torch/transformers installed.

## 3. Execute

```sh
python -m watch_wisely.cli --manifest /data/questions.json \
  --settings /data/settings.json --output runs/pilot-01 --run-id pilot-01 \
  --methods Uniform-8 Uniform-16 Uniform-24 Adaptive-24 Adaptive-always-24
```

Weights must already be cached unless `--allow-download` is explicitly supplied.
One model stays loaded across questions. Each method gets its own reader and frame
selection state. Uniform methods make exactly one answer call; all methods use the
same JSON/evidence contract, with a forced-answer override for fixed baselines.
Their prompt contexts differ by method, so this is not a token-matched experiment.

- `run.json`: method list, question/video identities, input hashes, model/processor
  revisions, settings, installed versions and source hashes. These are audit data,
  not independent certification of a run.
- `<method>/000000.json`: each full result, observations, generated responses,
  token counts and failures. Numeric names cannot be influenced by input IDs.
- `summary.json`: answered/failed counts across all question-method pairs, written
  only when the batch completes.

The command returns 1 if any question fails, 0 otherwise. Invalid inputs or model
loading failures abort execution. A question decode/generation/validation failure
is recorded and the next question still runs; failed questions receive no retries.
Existing output directories are refused. Interrupted runs remain incomplete and
should be rerun into a new directory. Results publish per question without
replacement; an in-flight GPU call is not crash-recoverable. There is no inference
timeout or automatic OOM recovery yet.

Whole-question latency includes reader initialization and decoding, prompt building,
generation and validation. It excludes input hashing, model loading and result
publication. Source hashes include all package modules; package versions are recorded,
but GPU driver/CUDA versions and hardware still need a separate environment record.

## 4. Compare, retaining every question

From the repository root:

```sh
python -m evaluation.run_report --run runs/pilot-01 \
  --manifest /data/scoring.json > runs/pilot-01/comparison.json
```

The report retains failed predictions as incorrect; rejects synthetic runs, mixed
identities, missing records and gold/run ID mismatches; and includes paired bootstrap
intervals, McNemar statistics, video-cluster sensitivity, observed frames, calls,
latency and token measurement coverage. The existing `evaluation.compare` command
remains available for comparisons between different runs. Missing measurements
remain unknown, never zero. Costs include failures; observed-frame counts omit a
batch that failed before committing its observations.

## Application wording

An implementation-only CV bullet (use after reviewing and understanding the code):

> Developed a budget-constrained video question-answering pipeline integrating
> Qwen3-VL, adaptive temporal inspection, timestamp-grounded evidence validation,
> and reproducible evaluation against fixed-frame baselines.

Until real inference is validated, label the project **in progress** and do not
claim improved accuracy, frame savings, a completed benchmark or a publication.
The original four-person course proposal and this repository's implementation
work must retain their separate attribution in `docs/proposal.md`.

To finish the empirical study: validate one real GPU question first; freeze the
checkpoint, environment and prompts; run the screened pilot and main question
sets; inspect failures; report the measured paired effects even if negative.
