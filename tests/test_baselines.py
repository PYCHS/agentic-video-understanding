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
