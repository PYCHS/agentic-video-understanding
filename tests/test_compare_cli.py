import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
from test_compare_records import records


def invoke(tmp_path, bad=None):
    inputs = {
        "manifest": [{"question_id":"q0", "video_id":"v0", "answer":"A"}],
        "baseline": records("Uniform-24", ["A"]),
        "adaptive": records("Adaptive-24", [None]),
    }
    if bad == "synthetic":
        inputs["adaptive"][0]["synthetic"] = True
    args = [sys.executable, "-m", "evaluation.compare"]
    for name, content in inputs.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(content), encoding="utf-8")
        args += [f"--{name}", str(path)]
    if bad == "duplicate":
        (tmp_path / "adaptive.json").write_text('[{"synthetic":true,"synthetic":false}]')
    if bad == "object":
        (tmp_path / "manifest.json").write_text('{}')
    if bad == "nan":
        (tmp_path / "manifest.json").write_text('[NaN]')
    args += ["--baseline-run-id", "Uniform-24", "--adaptive-run-id", "Adaptive-24",
             "--bootstrap-samples", "100"]
    return subprocess.run(args, cwd=Path(__file__).resolve().parents[1],
                          capture_output=True, text=True)


def test_cli_emits_report_and_hashes_exact_inputs(tmp_path):
    result = invoke(tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["paired"]["accuracy_delta"] == -1
    assert report["adaptive"]["failed"] == 1
    assert report["provenance_verified"] is False
    for name in ("manifest", "baseline", "adaptive"):
        assert report["input_sha256"][name] == hashlib.sha256(
            (tmp_path / f"{name}.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("bad", ["synthetic", "duplicate", "object", "nan"])
def test_invalid_cli_input_produces_no_report(tmp_path, bad):
    result = invoke(tmp_path, bad)
    assert result.returncode == 2
    assert not result.stdout
    assert "comparison failed:" in result.stderr
