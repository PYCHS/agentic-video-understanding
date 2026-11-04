"""Command-line comparison of explicit JSON arrays; report goes to stdout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .compare_records import compare_records


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _load(path):
    raw = path.read_bytes()
    try:
        rows = json.loads(raw.decode("utf-8"), object_pairs_hook=_object,
                          parse_constant=_constant)
        if not isinstance(rows, list):
            raise ValueError("expected a JSON array")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return rows, hashlib.sha256(raw).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "baseline", "adaptive"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--adaptive-run-id", required=True)
    parser.add_argument("--baseline-method", default="Uniform-24")
    parser.add_argument("--adaptive-method", default="Adaptive-24")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args(argv)
    try:
        loaded = {name: _load(getattr(args, name)) for name in ("manifest", "baseline", "adaptive")}
        report = compare_records(*(loaded[name][0] for name in ("manifest", "baseline", "adaptive")),
                                 baseline_run_id=args.baseline_run_id,
                                 adaptive_run_id=args.adaptive_run_id,
                                 baseline_method=args.baseline_method,
                                 adaptive_method=args.adaptive_method,
                                 bootstrap_samples=args.bootstrap_samples, seed=args.seed)
        report["input_sha256"] = {name: data[1] for name, data in loaded.items()}
        encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    except (OSError, ValueError, TypeError) as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 2
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
