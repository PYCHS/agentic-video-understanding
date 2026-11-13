from scripts.check_environment import inventory
import subprocess


def test_missing_dependencies_and_gpu_are_explicit(monkeypatch):
    from importlib.metadata import PackageNotFoundError
    def missing(name):
        raise PackageNotFoundError(name)
    def unavailable(*args, **kwargs):
        assert kwargs["timeout"] == 10
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr("scripts.check_environment.version", missing)
    monkeypatch.setattr("scripts.check_environment.subprocess.run", unavailable)
    result = inventory()
    assert all(v is None for v in result["packages"].values())
    assert result["nvidia_smi"]["status"] == "unavailable"
    assert result["inference_verified"] is False


def test_inventory_does_not_claim_inference_success(monkeypatch):
    monkeypatch.setattr("scripts.check_environment.version", lambda name: "test-version")
    monkeypatch.setattr("scripts.check_environment.subprocess.run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, "Test GPU, 1, 100 MiB\n", ""))
    result = inventory()
    assert result["nvidia_smi"]["csv"] == "Test GPU, 1, 100 MiB"
    assert result["inference_verified"] is False
