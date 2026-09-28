"""Bounded model calls and decision checks; does not execute observations."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

from .actions import Decision, parse_decision
from .backend import Backend, ModelResponse
from .controller import Controller


class CallStopped(RuntimeError):
    """This question cannot make another model call."""


@dataclass(frozen=True)
class CallRecord:
    call_index: int
    round_index: int
    forced_answer: bool
    status: str
    error: str | None
    response: ModelResponse | None
    wall_latency_ms: float


class DecisionCaller:
    """One caller per question; at most one call per round and four calls total.

    Failures end the session. No automatic repairs or backend retries are made.
    The caller checks a copy of the controller, leaving real observations alone.
    """

    def __init__(self, backend: Backend):
        self.backend = backend
        self.records: list[CallRecord] = []
        self.stopped = False
        self._last_round = 0

    def decide(self, messages: list[dict], controller: Controller) -> Decision:
        if self.stopped or controller.finished or len(self.records) >= 4:
            raise CallStopped("question ended or four-call limit reached")
        if controller.rounds != self._last_round + 1:
            raise CallStopped("each observation round permits exactly one call")
        self._last_round = controller.rounds
        # Snapshot before calling the backend: validation must not spend frames.
        snapshot = deepcopy(controller)
        response = None
        status = "backend_error"
        start = perf_counter()
        try:
            response = self.backend.generate(messages)
            status = "decision_rejected"
            decision = parse_decision(response.raw_text, duration=snapshot.duration,
                                      observed_timestamps=snapshot.seen)
            if decision.inspect is None:
                snapshot.answer(decision.candidate)
            else:
                request = decision.inspect
                snapshot.inspect(request.start, request.end, request.k)
        except Exception as exc:
            self.stopped = True
            self.records.append(CallRecord(len(self.records)+1, controller.rounds,
                                           controller.forced, status,
                                           f"{type(exc).__name__}: {exc}", response,
                                           (perf_counter()-start)*1000))
            raise CallStopped("model call failed; inspect records for the reason") from exc
        self.records.append(CallRecord(len(self.records)+1, controller.rounds,
                                       controller.forced, "accepted", None, response,
                                       (perf_counter()-start)*1000))
        if decision.inspect is None:
            self.stopped = True
        return decision

    def write_trace(self, path: Path):
        """Write this question's calls once; refuse to overwrite an existing trace.

        Token counts and backend latency are retained as reported, including nulls.
        Wall latency includes generation and validation, not video decoding.
        """
        lines = [json.dumps(asdict(record), allow_nan=False) for record in self.records]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write("".join(line + "\n" for line in lines))
