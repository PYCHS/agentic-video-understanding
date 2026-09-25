"""Pure controller state: no inference, decoding, or hidden retries."""
from dataclasses import dataclass
import math
from .sampling import uniform_timestamps


@dataclass(frozen=True)
class Answer:
    option: str
    evidence: tuple[float, ...]
    missing_evidence: str
    confidence: float


class Controller:
    FRAME_CAP = 24
    MAX_ROUNDS = 4

    def __init__(self, duration: float, always_use_24: bool = False):
        self.duration = duration
        self.seen = uniform_timestamps(0, duration, 8)
        self.rounds = 1
        self.finished = False
        self.always_use_24 = always_use_24

    @property
    def remaining(self):
        return self.FRAME_CAP - len(self.seen)

    @property
    def forced(self):
        return self.remaining == 0 or self.rounds == self.MAX_ROUNDS

    def inspect(self, start: float, end: float, k: int):
        if self.finished or self.forced:
            raise ValueError("session ended or final decision required")
        if type(k) is not int or k not in (4, 8) or k > self.remaining:
            raise ValueError("invalid or over-budget batch")
        if end > self.duration:
            raise ValueError("interval exceeds video duration")
        frames = uniform_timestamps(start, end, k)
        if any(b-a < 1e-6 for a,b in zip(frames, frames[1:])):
            raise ValueError("batch timestamps are too close to distinguish")
        if any(abs(t-s) < 1e-6 for t in frames for s in self.seen):
            raise ValueError("duplicate timestamp")
        rounds_left = self.MAX_ROUNDS - (self.rounds + 1)
        if self.always_use_24 and self.remaining-k > rounds_left*8:
            raise ValueError("batch would make 24 frames unreachable")
        self.seen += frames
        self.rounds += 1
        return frames

    def answer(self, decision: Answer):
        if self.finished:
            raise ValueError("session already ended")
        if decision.option not in ("A", "B", "C", "D"):
            raise ValueError("invalid answer option")
        if not isinstance(decision.missing_evidence, str):
            raise ValueError("missing_evidence must be text")
        if not math.isfinite(decision.confidence) or not 0 <= decision.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if len(set(decision.evidence)) != len(decision.evidence):
            raise ValueError("repeated evidence timestamp")
        if any(not math.isfinite(t) or t not in self.seen for t in decision.evidence):
            raise ValueError("evidence must reference observed timestamps")
        sufficient = (bool(decision.evidence) and not decision.missing_evidence.strip()
                      and decision.confidence >= 0.80)
        if not self.forced and (not sufficient or self.always_use_24):
            raise ValueError("early-stop criteria not met")
        self.finished = True
        return {"option": decision.option, "evidence": list(decision.evidence),
                "forced": self.forced, "evidence_sufficient": sufficient,
                "unique_frames": len(self.seen), "rounds": self.rounds}
