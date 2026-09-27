"""Parse one model decision without repairing it or changing controller state."""
from dataclasses import dataclass
import json
import math

from .controller import Answer

MAX_RESPONSE_CHARS = 16_384


class DecisionError(ValueError):
    """The response does not follow the action contract."""


@dataclass(frozen=True)
class Inspect:
    start: float
    end: float
    k: int


@dataclass(frozen=True)
class Decision:
    candidate: Answer
    inspect: Inspect | None  # None means answer now.


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise DecisionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise DecisionError(f"non-JSON numeric constant: {value}")


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise DecisionError(f"{label} must contain exactly: {', '.join(expected)}")


def _number(value, label):
    if type(value) not in (int, float):
        raise DecisionError(f"{label} must be a finite number")
    try:
        valid = math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise DecisionError(f"{label} must be a finite number")
    return value


def parse_decision(raw: str, *, duration: float, observed_timestamps) -> Decision:
    """Validate syntax, types, evidence membership and interval bounds.

    The controller still decides whether the budget, rounds and stopping rule
    allow this action. This function performs no calls, retries or mutations.
    """
    if type(raw) is not str or not raw.strip() or len(raw) > MAX_RESPONSE_CHARS:
        raise DecisionError("expected a nonempty JSON response of at most 16384 characters")
    try:
        data = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
    except (ValueError, RecursionError) as exc:
        raise DecisionError(f"invalid decision JSON: {exc}") from exc
    _keys(data, ("option", "evidence", "missing_evidence", "confidence", "action"), "decision")
    if type(data["option"]) is not str or data["option"] not in ("A", "B", "C", "D"):
        raise DecisionError("option must be A, B, C or D")
    if type(data["missing_evidence"]) is not str:
        raise DecisionError("missing_evidence must be text")
    confidence = _number(data["confidence"], "confidence")
    if not 0 <= confidence <= 1:
        raise DecisionError("confidence must be in [0, 1]")
    evidence = data["evidence"]
    if type(evidence) is not list:
        raise DecisionError("evidence must be a list of observed timestamps")
    observed = set(observed_timestamps)
    for time in evidence:
        _number(time, "evidence timestamp")
        if time not in observed:
            raise DecisionError("evidence must reference an observed timestamp")
    if len(set(evidence)) != len(evidence):
        raise DecisionError("evidence timestamps must not repeat")
    candidate = Answer(data["option"], tuple(evidence), data["missing_evidence"], confidence)
    action = data["action"]
    if type(action) is not dict:
        raise DecisionError("action must be an object")
    name = action.get("name")
    if name == "answer":
        _keys(action, ("name",), "answer action")
        return Decision(candidate, None)
    if name == "inspect":
        _keys(action, ("name", "start", "end", "k"), "inspect action")
        start = _number(action["start"], "start")
        end = _number(action["end"], "end")
        if not 0 <= start < end <= duration:
            raise DecisionError("inspect interval must satisfy 0 <= start < end <= duration")
        if type(action["k"]) is not int or action["k"] not in (4, 8):
            raise DecisionError("inspect k must be the integer 4 or 8")
        return Decision(candidate, Inspect(start, end, action["k"]))
    raise DecisionError("action name must be inspect or answer")
