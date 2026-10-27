from copy import deepcopy
import pytest

from evaluation.result_predictions import predictions_from_records
from evaluation.pair_predictions import paired_correctness


def record():
    return dict(schema_version=1, synthetic=False, run_id="r1", method="Adaptive-24",
                question_id="q1", video_id="v1", result=dict(status="answered",
                answer={"option": "A"}, error=None, failed_stage=None))


def convert(records):
    return predictions_from_records(records, run_id="r1", method="Adaptive-24")


def test_answer_and_failure_reach_pairing_without_dropping_rows():
    success = record()
    failed = deepcopy(success)
    failed["question_id"] = "q2"
    failed["result"] = dict(status="failed", answer=None, error="decode failed", failed_stage="survey")
    rows = convert([failed, success])
    ids, a, b = paired_correctness([{"question_id":"q1","answer":"A"},
                                   {"question_id":"q2","answer":"B"}], rows, rows)
    assert ids == ["q1", "q2"] and a == b == [True, False]
    assert rows[0]["video_id"] == "v1"
    assert success == record()  # Conversion must not mutate source records.


@pytest.mark.parametrize("field,value", [("synthetic", True), ("synthetic", None),
    ("synthetic", 0), ("schema_version", True), ("schema_version", 2),
    ("run_id", "another"), ("method", "Uniform-24"), ("video_id", "")])
def test_rejects_mixed_or_unverified_records(field, value):
    item = record()
    item[field] = value
    with pytest.raises(ValueError):
        convert([item])


@pytest.mark.parametrize("result", [
    {"status":"failed", "answer":None},
    {"status":"answered", "answer":{"option":"E"}},
    {"status":"answered", "answer":{"option":"A"}, "error":"failure"},
    {"status":"failed", "answer":{"option":"A"}},
    {"status":"pending", "answer":None},
])
def test_rejects_invalid_outcomes(result):
    item = record()
    item["result"] = result
    with pytest.raises(ValueError):
        convert([item])


def test_rejects_duplicates_empty_and_missing_manifest_rows():
    with pytest.raises(ValueError):
        convert([record(), record()])
    with pytest.raises(ValueError):
        convert([])
    rows = convert([record()])
    with pytest.raises(ValueError, match="question IDs differ"):
        paired_correctness([{"question_id":"q2","answer":"A"}], rows, rows)
