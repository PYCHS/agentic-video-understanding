# Pair predictions before scoring

paired_correctness in pair_predictions.py aligns both methods to a frozen manifest by
question_id. It returns manifest-ordered IDs and baseline/adaptive boolean vectors for
summarize_paired_results. Input row order does not affect pairing.

Manifest rows contain question_id and answer (A/B/C/D). Prediction rows contain question_id,
status and option. Use status answered with an A/B/C/D option, or failed with a null option.
Do not drop failed runs: they count as incorrect and stay in the denominator. Convert runner
failures to failed rows explicitly, retaining their reasons in the original trace.

Empty inputs, duplicate IDs, missing/extra questions, unknown statuses and invalid options
raise errors. A malformed stored row is rejected; a failed model run is a valid scoring row.
Never guess which row a partial or ambiguous record belongs to.

Gold labels belong only in this offline scoring step, never in prompts. The future evaluation
runner must also verify run provenance, backbone/config hashes and frozen-manifest identity.
This helper checks IDs and scoring fields; it does not certify matched experimental settings.
Tests use artificial examples. No dataset has been scored.
