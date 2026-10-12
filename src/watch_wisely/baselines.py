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

@dataclass(frozen=True)
class CoarseToFineObservation:
    frames: tuple
    coarse_frame_ids: tuple[int, ...]
    fine_requested_s: tuple[float, ...]
    encoder_latency_ms: float
    selection_latency_ms: float


def observe_coarse_to_fine(reader: FrameReader, encode_images) -> CoarseToFineObservation:
    """Fixed 8+16 selection with an injected frozen CLIP image encoder.

    encode_images receives only eight images in time order and returns eight
    embeddings. It must finish (including GPU synchronization) before returning.
    No question text, retry, or adaptive tool call is involved.
    """
    from .sampling import coarse_to_fine_timestamps

    started = perf_counter()
    coarse = observe_uniform(reader, 8).frames
    encoder_started = perf_counter()
    embeddings = encode_images(tuple(frame.image for frame in coarse))
    encoder_ms = (perf_counter() - encoder_started) * 1000
    actual = tuple(frame.timestamp_s for frame in coarse)
    selected = coarse_to_fine_timestamps(actual, embeddings, fine_count=16)
    fine_requested = tuple(t for t in selected if t not in actual)
    if len(fine_requested) != 16:
        raise ValueError("fine grid must contain 16 new timestamps")
    fine = tuple(reader.read(fine_requested,
                            seen_frame_ids={frame.frame_id for frame in coarse}))
    ObservationSession._check_batch(fine, fine_requested, coarse)
    frames = tuple(sorted((*coarse, *fine), key=lambda frame: frame.timestamp_s))
    return CoarseToFineObservation(frames, tuple(f.frame_id for f in coarse),
                                  fine_requested, encoder_ms,
                                  (perf_counter() - started) * 1000)
