"""Run one question through observation, decision validation and tool execution."""
from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from .backend import Backend
from .calls import CallRecord, DecisionCaller
from .session import FrameReader, ObservationSession


@dataclass(frozen=True)
class Observation:
    round_index: int
    frame_ids: tuple[int, ...]
    requested_s: tuple[float, ...]
    actual_s: tuple[float, ...]


@dataclass(frozen=True)
class RunResult:
    status: str
    answer: dict | None
    error: str | None
    failed_stage: str | None
    calls: tuple[CallRecord, ...]
    observations: tuple[Observation, ...]
    wall_latency_ms: float


def run_question(reader: FrameReader, backend: Backend,
                 build_messages: Callable[[ObservationSession, tuple[CallRecord, ...]], list[dict]],
                 *, always_use_24: bool = False) -> RunResult:
    """No hidden retries. Message construction is injected until prompts are frozen.

    The builder must not mutate the session. It receives all observed frames and
    previous calls so an adapter can construct a timestamped multimodal history.
    Backend timeouts and persistent trace storage remain the caller's responsibility.
    """
    started = perf_counter()
    caller = DecisionCaller(backend)
    observations = []
    stage = "survey"

    def record(session, frames):
        observations.append(Observation(session.controller.rounds,
                            tuple(f.frame_id for f in frames),
                            tuple(f.requested_s for f in frames),
                            tuple(f.timestamp_s for f in frames)))

    try:
        session = ObservationSession(reader, always_use_24=always_use_24)
        record(session, session.frames)
        # The call wrapper and controller also enforce these bounds.
        for _ in range(4):
            stage = "messages"
            messages = build_messages(session, tuple(caller.records))
            stage = "decision"
            decision = caller.decide(messages, session.controller)
            stage = "apply"
            output = session.apply(decision)
            if decision.inspect is None:
                return RunResult("answered", output, None, None, tuple(caller.records),
                                 tuple(observations), (perf_counter()-started)*1000)
            record(session, output)
        raise RuntimeError("four rounds ended without an answer")
    except Exception as exc:
        return RunResult("failed", None, f"{type(exc).__name__}: {exc}", stage,
                         tuple(caller.records), tuple(observations),
                         (perf_counter()-started)*1000)
