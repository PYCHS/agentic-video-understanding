from copy import deepcopy
import json

import pytest

from evaluation.summarize_calls import load_call_records, summarize_call_records


def record():
    return {"call_index":1,"round_index":1,"forced_answer":False,"status":"accepted",
            "error":None,"wall_latency_ms":10,
            "response":{"raw_text":"test fixture","input_tokens":20,"output_tokens":5,
                        "visual_tokens":None,"latency_ms":8}}


@pytest.mark.parametrize("value",[-1, float("nan"), float("inf"), True, "10"])
def test_invalid_latency_rejected_in_memory_and_file(value,tmp_path):
    row = record()
    row["wall_latency_ms"] = value
    with pytest.raises(ValueError):
        summarize_call_records([row])
    path = tmp_path/"calls.jsonl"
    path.write_text(json.dumps(row)+"\n",encoding="utf-8")
    with pytest.raises(ValueError,match="calls.jsonl:1:"):
        load_call_records([path])


@pytest.mark.parametrize("value",[-1, .5, True, float("nan"), "4"])
def test_token_counts_must_be_nonnegative_integers(value):
    row = record()
    row["response"]["input_tokens"] = value
    with pytest.raises(ValueError):
        summarize_call_records([row])


def test_partial_measurements_report_coverage_without_zero_fill():
    first = record()
    second = deepcopy(first)
    second.update(status="backend_error",error="fixture failure",response=None)
    summary = summarize_call_records([first,second])
    assert summary["calls"] == 2
    assert summary["failed_calls"] == 1
    assert summary["input_tokens_total"] == 20
    assert summary["visual_tokens_total"] is None
    assert summary["model_latency_ms"] == {
        "mean": 8,
        "median": 8,
        "p95": 8,
        "total": 8,
    }
    assert summary["measurement_coverage"]["input_tokens"] == 1
    assert summary["measurement_coverage"]["visual_tokens"] == 0
    assert summary["measurement_coverage"]["model_latency_ms"] == 1


def test_empty_unknown_status_and_missing_response_fail():
    with pytest.raises(ValueError):
        summarize_call_records([])
    for changes in [{"status":"made_up"},{"response":None},{"call_index":0}]:
        with pytest.raises(ValueError):
            summarize_call_records([record() | changes])


def test_valid_jsonl_roundtrip_and_percentile(tmp_path):
    first,second = record(),record()
    second["wall_latency_ms"] = 30
    path = tmp_path/"calls.jsonl"
    path.write_text("\n".join(map(json.dumps,[first,second])),encoding="utf-8")
    summary = summarize_call_records(load_call_records([path]))
    assert summary["wall_latency_ms"] == {"mean":20,"median":20,"p95":29,"total":40}


def test_model_latency_summary_uses_only_measured_calls():
    first, second, third = record(), record(), record()
    second["response"]["latency_ms"] = 18
    third["response"]["latency_ms"] = None

    summary = summarize_call_records([first, second, third])

    assert summary["model_latency_ms"] == {
        "mean": 13,
        "median": 13,
        "p95": 17.5,
        "total": 26,
    }
    assert summary["model_latency_ms_total"] == 26
    assert summary["measurement_coverage"]["model_latency_ms"] == 2
