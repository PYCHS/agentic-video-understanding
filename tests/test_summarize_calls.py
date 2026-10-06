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
    assert summary["measurement_coverage"]["input_tokens"] == 1
    assert summary["measurement_coverage"]["visual_tokens"] == 0


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
