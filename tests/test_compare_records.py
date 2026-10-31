import json
import pytest
from evaluation.compare_records import compare_records


def records(method, options):
    return [dict(schema_version=1, synthetic=False, run_id=method, method=method,
                 question_id=f"q{i}", video_id=f"v{i}", result=dict(
                 status="failed" if option is None else "answered",
                 answer=None if option is None else {"option":option},
                 error="decode failed" if option is None else None,
                 failed_stage="survey" if option is None else None))
            for i, option in enumerate(options)]


def compare(gold, base, adaptive):
    return compare_records(gold, base, adaptive, baseline_run_id="Uniform-24",
                           adaptive_run_id="Adaptive-24", bootstrap_samples=100)


def test_report_pairs_shuffled_records_and_keeps_failures():
    gold = [{"question_id":f"q{i}", "video_id":f"v{i}", "answer":"A"} for i in range(2)]
    report = compare(iter(gold), reversed(records("Uniform-24", ["A", None])),
                     records("Adaptive-24", ["A", "A"]))
    assert report["baseline"]["accuracy"] == .5
    assert report["baseline"]["failed"] == 1
    assert report["adaptive"]["accuracy"] == 1
    assert report["paired"]["accuracy_delta"] == .5
    assert report["video_cluster"]["accuracy_delta"] == .5
    assert report["question_ids"] == ["q0", "q1"]
    assert report["provenance_verified"] is False
    json.dumps(report, allow_nan=False)


def test_single_video_reports_unavailable_cluster_interval():
    report = compare([{"question_id":"q0", "video_id":"v0", "answer":"A"}],
                     records("Uniform-24", ["A"]), records("Adaptive-24", [None]))
    assert report["paired"]["accuracy_delta"] == -1
    assert report["video_cluster"] is None
    assert report["video_cluster_unavailable_reason"]


@pytest.mark.parametrize("corruption", ["synthetic", "video", "missing"])
def test_invalid_comparison_fails_before_statistics(corruption, monkeypatch):
    gold = [{"question_id":"q0", "video_id":"v0", "answer":"A"}]
    base, adaptive = records("Uniform-24", ["A"]), records("Adaptive-24", ["A"])
    if corruption == "synthetic":
        adaptive[0]["synthetic"] = True
    elif corruption == "video":
        adaptive[0]["video_id"] = "wrong"
    else:
        adaptive = []
    def forbidden(*args, **kwargs):
        pytest.fail("invalid records must not reach statistics")
    monkeypatch.setattr("evaluation.compare_records.summarize_paired_results", forbidden)
    with pytest.raises(ValueError):
        compare(gold, base, adaptive)
