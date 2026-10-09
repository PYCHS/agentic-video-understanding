"""Align predictions to a frozen manifest before computing paired statistics."""
from collections.abc import Iterable, Mapping


def _index(rows: Iterable[Mapping], label: str):
    indexed = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}: expected an object")
        key = row.get("question_id")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label}: question_id must be nonempty text")
        if key in indexed:
            raise ValueError(f"{label}: duplicate question_id {key}")
        indexed[key] = row
    if not indexed:
        raise ValueError(f"{label}: no rows")
    return indexed


def paired_correctness(manifest, baseline, adaptive):
    """Return IDs and boolean correctness vectors in manifest order.

    Manifest rows: question_id, answer (A/B/C/D).
    Prediction rows: question_id, status (answered/failed), option (A/B/C/D
    for answered; null for failed). Failed predictions are retained as incorrect.
    This is a scoring boundary: gold answers must never reach model prompts.
    """
    gold = _index(manifest, "manifest")
    for row in gold.values():
        if row.get("answer") not in ("A", "B", "C", "D"):
            raise ValueError("manifest: answer must be A/B/C/D")
    vectors = []
    for name, rows in (("baseline",baseline),("adaptive",adaptive)):
        predictions = _index(rows,name)
        if predictions.keys() != gold.keys():
            missing = sorted(gold.keys()-predictions.keys())
            extra = sorted(predictions.keys()-gold.keys())
            raise ValueError(f"{name}: question IDs differ; missing={missing}, extra={extra}")
        correct = []
        for key, target in gold.items():
            row = predictions[key]
            if "option" not in row:
                raise ValueError(f"{name}: missing option for {key}; use null for failure")
            if row.get("status") == "failed":
                if row["option"] is not None:
                    raise ValueError(f"{name}: failed prediction must have null option")
                correct.append(False)
            elif row.get("status") == "answered":
                if row["option"] not in ("A", "B", "C", "D"):
                    raise ValueError(f"{name}: invalid answered option for {key}")
                correct.append(row["option"] == target["answer"])
            else:
                raise ValueError(f"{name}: unknown prediction status for {key}")
        vectors.append(correct)
    return list(gold), vectors[0], vectors[1]
