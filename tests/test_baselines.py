import pytest

from test_session import GridReader
from watch_wisely.baselines import observe_uniform


@pytest.mark.parametrize("count", [8, 16, 24])
def test_uniform_grids_use_actual_frames(count):
    result = observe_uniform(GridReader(), count)
    assert result.method == f"Uniform-{count}"
    assert len(result.frames) == count
    assert len({f.frame_id for f in result.frames}) == count
    assert [f.requested_s for f in result.frames] == [
        (i + .5) * 120 / count for i in range(count)]
    assert all(f.timestamp_s == int(f.requested_s) for f in result.frames)
    assert result.selection_latency_ms >= 0


@pytest.mark.parametrize("count", [True, 8.0, 0, 4, 32])
def test_invalid_budget_does_not_read(count):
    class NoRead(GridReader):
        def read(self, timestamps):
            pytest.fail("invalid budget must fail before decoding")
    with pytest.raises(ValueError):
        observe_uniform(NoRead(), count)


def test_decode_failure_is_not_retried_or_padded():
    class ShortReader(GridReader):
        duration_s = 1
        calls = 0

        def read(self, timestamps):
            self.calls += 1
            return super().read(timestamps)
    reader = ShortReader()
    with pytest.raises(ValueError, match="collision"):
        observe_uniform(reader, 24)
    assert reader.calls == 1
    reader = GridReader()
    reader.incomplete = True
    with pytest.raises(ValueError, match="incomplete"):
        observe_uniform(reader, 16)


class FineReader:
    duration_s = 128

    def __init__(self):
        self.requests = []

    def read(self, timestamps, *, seen_frame_ids=()):
        from types import SimpleNamespace
        self.requests.append((tuple(timestamps), set(seen_frame_ids)))
        frames = tuple(SimpleNamespace(frame_id=round(t * 1000), requested_s=t,
                                       timestamp_s=t, image=f"image-{t}")
                       for t in timestamps)
        return frames


def test_fixed_coarse_to_fine_uses_images_and_excludes_coarse_ids():
    from watch_wisely.baselines import observe_coarse_to_fine
    reader = FineReader()
    calls = []
    def encode(images):
        calls.append(images)
        return [(1, 0)] * 3 + [(-1, 0)] * 5
    result = observe_coarse_to_fine(reader, encode)
    assert len(calls) == 1 and len(calls[0]) == 8
    assert calls[0][0] == "image-8.0"
    assert len(result.frames) == 24
    assert len({f.frame_id for f in result.frames}) == 24
    assert all(40 < t < 56 for t in result.fine_requested_s)
    assert reader.requests[1][1] == set(result.coarse_frame_ids)
    assert result.selection_latency_ms >= result.encoder_latency_ms >= 0
    assert [f.timestamp_s for f in result.frames] == sorted(f.timestamp_s for f in result.frames)


def test_fixed_tie_uses_earliest_actual_interval():
    from watch_wisely.baselines import observe_coarse_to_fine
    class ShiftedReader(FineReader):
        def read(self, timestamps, **kwargs):
            frames = super().read(timestamps, **kwargs)
            for frame in frames:
                frame.timestamp_s -= .1
            return frames
    result = observe_coarse_to_fine(ShiftedReader(), lambda images: [(1, 0)] * 8)
    assert result.fine_requested_s[0] == pytest.approx(7.9 + .5)
    assert result.fine_requested_s[-1] == pytest.approx(23.9 - .5)


@pytest.mark.parametrize("embeddings", [[(0, 0)] * 8, [(1, 0)] * 7])
def test_invalid_embeddings_stop_before_fine_decode(embeddings):
    from watch_wisely.baselines import observe_coarse_to_fine
    reader = FineReader()
    with pytest.raises(ValueError):
        observe_coarse_to_fine(reader, lambda images: embeddings)
    assert len(reader.requests) == 1


def test_fine_collision_is_terminal():
    from watch_wisely.baselines import observe_coarse_to_fine
    class CollidingReader(FineReader):
        def read(self, timestamps, **kwargs):
            frames = super().read(timestamps, **kwargs)
            if kwargs.get("seen_frame_ids"):
                frames[0].frame_id = min(kwargs["seen_frame_ids"])
            return frames
    reader = CollidingReader()
    with pytest.raises(ValueError, match="repeated frame IDs"):
        observe_coarse_to_fine(reader, lambda images: [(1, 0)] * 8)
    assert len(reader.requests) == 2
