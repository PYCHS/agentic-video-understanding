"""Convert saved question records into rows accepted by paired_correctness."""
from collections.abc import Mapping


def predictions_from_records(records, *, run_id: str, method: str):
    """Reject mixed runs/methods and synthetic records; retain failed questions.

    This checks record identity and prediction shape, not model provenance or
    correctness of the enclosed trace. Call paired_correctness to enforce the
    full manifest denominator after conversion.
    """
    for value in (run_id, method):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected nonempty run_id and method")
    rows, seen = [], set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("question record must be an object")
        if type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            raise ValueError("unsupported question record schema")
        if record.get("synthetic") is not False:
            raise ValueError("scoring requires an explicitly non-synthetic record")
        if record.get("run_id") != run_id or record.get("method") != method:
            raise ValueError("question record belongs to a different run or method")
        question_id, video_id = record.get("question_id"), record.get("video_id")
        if any(not isinstance(v, str) or not v.strip() for v in (question_id, video_id)):
            raise ValueError("question and video IDs must be nonempty text")
        if question_id in seen:
            raise ValueError(f"duplicate question_id: {question_id}")
        seen.add(question_id)
        result = record.get("result")
        if not isinstance(result, Mapping) or "answer" not in result:
            raise ValueError("result must contain an explicit answer field")
        status, answer = result.get("status"), result["answer"]
        if status == "answered":
            if not isinstance(answer, Mapping) or answer.get("option") not in ("A", "B", "C", "D"):
                raise ValueError("answered record requires option A/B/C/D")
            if result.get("error") is not None or result.get("failed_stage") is not None:
                raise ValueError("answered record cannot carry a failure")
            option = answer["option"]
        elif status == "failed":
            if answer is not None:
                raise ValueError("failed record must have a null answer")
            if not isinstance(result.get("error"), str) or not result["error"].strip():
                raise ValueError("failed record requires an error reason")
            if result.get("failed_stage") not in ("survey", "messages", "decision", "apply"):
                raise ValueError("failed record requires a known runner stage")
            option = None
        else:
            raise ValueError("unknown question result status")
        rows.append(dict(question_id=question_id, video_id=video_id,
                         status=status, option=option))
    if not rows:
        raise ValueError("no question records")
    return rows
