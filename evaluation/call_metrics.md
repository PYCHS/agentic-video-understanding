# Checking call-cost traces

Run `python evaluation/summarize_calls.py runs/question.jsonl` on a call trace written by
DecisionCaller.write_trace. The loader reports the file and line for invalid measurements.
Direct calls to summarize_call_records use the same checks.

Negative or nonfinite costs, booleans used as numbers, fractional token counts, unknown
statuses and incomplete response fields are rejected. Missing measurements must be null.
A backend failure can have no response; an accepted or rejected model decision must retain
its response. Empty inputs are rejected rather than producing an empty report.

Totals cover only measured calls. Always read measurement_coverage alongside them: a token
total with coverage 1/2 is a partial sum, not the cost of both calls. No measured values means
null, not zero. Wall latency statistics are per-call, not per-question or full-pipeline latency.
Group traces by method and backbone before summarizing; the utility does not infer groups.

Tests use fabricated numeric fixtures only to check validation and arithmetic. They are not
experiment measurements. Unique frames, per-question results and paired accuracy statistics
still require the evaluation runner and frozen dataset manifest.
