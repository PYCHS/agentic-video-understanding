"""Deterministic midpoint timestamps; decoder deduplication remains required."""
import math


def uniform_timestamps(start: float, end: float, k: int) -> tuple[float, ...]:
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not all(math.isfinite(x) for x in (start, end)) or not 0 <= start < end:
        raise ValueError("expected finite interval with 0 <= start < end")
    result = tuple(start + (i + 0.5) * (end - start) / k for i in range(k))
    if len(set(result)) != k:
        raise ValueError("interval cannot represent distinct timestamps")
    return result


def coarse_to_fine_timestamps(coarse, embeddings, fine_count=16):
    """Fixed selection from supplied CLIP embeddings; encoder not included.

    Largest adjacent cosine distance wins; ties choose the earliest interval.
    """
    if len(coarse) < 2 or len(coarse) != len(embeddings):
        raise ValueError("need matching timestamps and embeddings")
    if any(not math.isfinite(t) or t < 0 for t in coarse):
        raise ValueError("invalid timestamps")
    if any(a >= b for a, b in zip(coarse, coarse[1:])):
        raise ValueError("timestamps must strictly increase")
    dims = {len(e) for e in embeddings}
    if len(dims) != 1 or not next(iter(dims)):
        raise ValueError("embedding dimensions must match and be nonzero")
    normalized = []
    for e in embeddings:
        if any(not math.isfinite(x) for x in e):
            raise ValueError("nonfinite embedding")
        norm = math.sqrt(sum(x*x for x in e))
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("invalid embedding norm")
        normalized.append([x/norm for x in e])
    distances = [1-sum(x*y for x,y in zip(a,b))
                 for a,b in zip(normalized, normalized[1:])]
    i = max(range(len(distances)), key=distances.__getitem__)
    fine = uniform_timestamps(coarse[i], coarse[i+1], fine_count)
    return tuple(sorted((*coarse, *fine)))
