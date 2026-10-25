import json
from dataclasses import replace

import pytest

from test_calls import ScriptedBackend
from test_runner import reply
from test_session import GridReader
from watch_wisely.backend import ModelResponse
from watch_wisely.runner import run_question
from watch_wisely.results import write_question_result


def save(path, result, **overrides):
    metadata = dict(run_id="test-run", question_id="q1", video_id="v1",
                    method="Adaptive-24", synthetic=True)
    metadata.update(overrides)
    return write_question_result(path, result, **metadata)


@pytest.mark.parametrize("valid", [True, False])
def test_runner_result_round_trip(tmp_path, valid):
    response = reply({"name": "answer"}) if valid else ModelResponse("broken", None, None, None, 1)
    result = run_question(GridReader(), ScriptedBackend([response]), lambda s, h: [])
    path = save(tmp_path / "q1.json", result)
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["synthetic"] is True
    assert record["question_id"] == "q1"
    assert record["result"]["status"] == ("answered" if valid else "failed")
    assert record["result"]["calls"][0]["response"]["raw_text"] == response.raw_text
    assert record["result"]["observations"][0]["actual_s"][0] == 7
    assert record["result"]["error"] == result.error
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save(path, result)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_publish_failure_leaves_no_partial_result(tmp_path, monkeypatch):
    result = run_question(GridReader(), ScriptedBackend([]), lambda s, h: [])
    def fail(*args):
        raise OSError("publication unavailable")
    monkeypatch.setattr("watch_wisely.results.os.link", fail)
    with pytest.raises(OSError):
        save(tmp_path / "q1.json", result)
    assert list(tmp_path.iterdir()) == []


def test_validation_precedes_filesystem_changes(tmp_path):
    result = run_question(GridReader(), ScriptedBackend([]), lambda s, h: [])
    path = tmp_path / "new" / "q1.json"
    for metadata in ({"question_id": ""}, {"synthetic": 1}):
        with pytest.raises(ValueError):
            save(path, result, **metadata)
    with pytest.raises(ValueError):
        save(path, replace(result, wall_latency_ms=float("nan")))
    assert not path.parent.exists()
