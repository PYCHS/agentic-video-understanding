"""Fixed-budget observations for the uniform baselines (no model calls)."""
from dataclasses import dataclass
from time import perf_counter

from .sampling import uniform_timestamps
from .session import FrameReader, ObservationSession


@dataclass(frozen=True)
class UniformObservation:
    method: str
    frames: tuple
    selection_latency_ms: float


def observe_uniform(reader: FrameReader, frame_count: int) -> UniformObservation:
    """Decode one independent midpoint grid, retaining actual evidence times.

    Timing includes sampling and decoding, but excludes reader initialization.
    Errors propagate: never retry, pad, or substitute a smaller baseline.
    """
    if type(frame_count) is not int or frame_count not in (8, 16, 24):
        raise ValueError("uniform baseline frame_count must be 8, 16 or 24")
    started = perf_counter()
    requested = uniform_timestamps(0, reader.duration_s, frame_count)
    frames = tuple(reader.read(requested))
    ObservationSession._check_batch(frames, requested, ())
    return UniformObservation(f"Uniform-{frame_count}", frames,
                              (perf_counter() - started) * 1000)
