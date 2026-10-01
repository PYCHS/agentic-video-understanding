"""Commit observations only after a complete video batch has been decoded."""
from copy import deepcopy
from typing import Protocol

from .actions import Decision
from .controller import Controller


class FrameReader(Protocol):
    duration_s: float

    def read(self, timestamps, *, seen_frame_ids=()):
        """Return SelectedFrame objects in request order, or raise on failure."""
        ...


class ObservationSession:
    """Own one question's observations; evidence uses actual decoded times.

    The reader must obey VideoReader's contract, including duplicate rejection.
    No model calls, prompt construction or automatic retries happen here.
    """

    def __init__(self, reader: FrameReader, *, always_use_24=False):
        self._reader = reader
        controller = Controller(reader.duration_s, always_use_24)
        frames = tuple(reader.read(controller.seen))
        self._check_batch(frames, controller.seen, ())
        controller.seen = tuple(frame.timestamp_s for frame in frames)
        self._controller = controller
        self._frames = frames

    @staticmethod
    def _check_batch(frames, requested, previous):
        if len(frames) != len(requested):
            raise ValueError("reader returned an incomplete batch")
        ids = [frame.frame_id for frame in frames]
        times = [frame.timestamp_s for frame in frames]
        if len(set(ids)) != len(ids) or set(ids).intersection(f.frame_id for f in previous):
            raise ValueError("reader returned repeated frame IDs")
        if len(set(times)) != len(times) or set(times).intersection(f.timestamp_s for f in previous):
            raise ValueError("reader returned repeated evidence timestamps")
        if tuple(frame.requested_s for frame in frames) != tuple(requested):
            raise ValueError("reader changed request order or timestamps")

    @property
    def controller(self):
        """Snapshot for DecisionCaller; outside checks cannot spend the budget."""
        return deepcopy(self._controller)

    @property
    def frames(self):
        return self._frames

    def apply(self, decision: Decision):
        """Return the answer dict or newly decoded frames; errors leave state intact.

        After an error, the future runner must stop the question under the
        zero-retry policy. State rollback is not permission for another call.
        """
        trial = deepcopy(self._controller)
        if decision.inspect is None:
            result = trial.answer(decision.candidate)
            self._controller = trial
            return result
        request = decision.inspect
        requested = trial.inspect(request.start, request.end, request.k)
        frames = tuple(self._reader.read(requested,
                       seen_frame_ids={frame.frame_id for frame in self._frames}))
        self._check_batch(frames, requested, self._frames)
        all_frames = self._frames + frames
        trial.seen = tuple(frame.timestamp_s for frame in all_frames)
        self._frames = all_frames
        self._controller = trial
        return frames
