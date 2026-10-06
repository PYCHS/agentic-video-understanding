"""Summarize measured VLM call traces without inventing missing values.

This utility consumes the JSONL files written by
watch_wisely.calls.DecisionCaller.write_trace. It only aggregates fields
already present in recorded traces.

Missing token measurements remain None in the summary rather than being
coerced to zero. Coverage counts are reported for every optional metric so a
partially instrumented backend cannot look artificially cheap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


def _percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def load_call_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Load DecisionCaller JSONL traces and validate their minimal schema."""
    records: list[dict[str, Any]] = []
    required = {
        "call_index",
        "round_index",
        "forced_answer",
        "status",
        "error",
        "response",
        "wall_latency_ms",
    }

    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = required.difference(row)
                if missing:
                    names = ", ".join(sorted(missing))
                    raise ValueError(f"{path}:{line_number}: missing fields: {names}")
                if row["wall_latency_ms"] < 0:
                    raise ValueError(
                        f"{path}:{line_number}: wall_latency_ms must be non-negative"
                    )
                records.append(row)

    if not records:
        raise ValueError("no call records found")
    return records


def _optional_response_metric(
    records: list[dict[str, Any]], field: str
) -> tuple[float | None, int]:
    values: list[float] = []
    for row in records:
        response = row.get("response")
        if response is None:
            continue
        value = response.get(field)
        if value is not None:
            values.append(float(value))
    return (sum(values) if values else None, len(values))


def summarize_call_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate efficiency statistics for one collection of calls."""
    wall_latencies = [float(row["wall_latency_ms"]) for row in records]
    accepted = sum(row["status"] == "accepted" for row in records)
    failed = len(records) - accepted

    input_tokens, input_coverage = _optional_response_metric(records, "input_tokens")
    output_tokens, output_coverage = _optional_response_metric(records, "output_tokens")
    visual_tokens, visual_coverage = _optional_response_metric(records, "visual_tokens")
    model_latency_ms, model_latency_coverage = _optional_response_metric(
        records, "latency_ms"
    )

    return {
        "calls": len(records),
        "accepted_calls": accepted,
        "failed_calls": failed,
        "wall_latency_ms": {
            "mean": mean(wall_latencies),
            "median": median(wall_latencies),
            "p95": _percentile(wall_latencies, 0.95),
            "total": sum(wall_latencies),
        },
        "input_tokens_total": input_tokens,
        "output_tokens_total": output_tokens,
        "visual_tokens_total": visual_tokens,
        "model_latency_ms_total": model_latency_ms,
        "measurement_coverage": {
            "input_tokens": input_coverage,
            "output_tokens": output_coverage,
            "visual_tokens": visual_coverage,
            "model_latency_ms": model_latency_coverage,
            "calls": len(records),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize measured DecisionCaller JSONL traces."
    )
    parser.add_argument("traces", nargs="+", type=Path, help="JSONL call trace files")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path; stdout is used when omitted.",
    )
    args = parser.parse_args()

    summary = summarize_call_records(load_call_records(args.traces))
    payload = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"

    if args.output is None:
        print(payload, end="")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
