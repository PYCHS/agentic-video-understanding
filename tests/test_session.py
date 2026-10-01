from dataclasses import dataclass
import unittest

from watch_wisely.actions import Decision, Inspect
from watch_wisely.controller import Answer
from watch_wisely.session import ObservationSession


@dataclass(frozen=True)
class Frame:
    frame_id: int
    requested_s: float
    timestamp_s: float


class GridReader:
    """Synthetic 1 FPS timeline to test the reader boundary, not video decoding."""
    duration_s = 120
    fail = False
    incomplete = False

    def read(self, timestamps, *, seen_frame_ids=()):
        if self.fail:
            raise OSError("decode failed")
        frames = tuple(Frame(int(t),t,float(int(t))) for t in timestamps)
        ids = [f.frame_id for f in frames]
        if len(set(ids)) != len(ids) or set(ids).intersection(seen_frame_ids):
            raise ValueError("frame collision")
        return frames[:-1] if self.incomplete else frames


def inspect(start, end, k=4):
    return Decision(Answer("A",(),"more evidence",.2),Inspect(start,end,k))


class SessionTests(unittest.TestCase):
    def test_survey_uses_actual_times_and_snapshot_is_detached(self):
        session = ObservationSession(GridReader())
        self.assertEqual(session.frames[0].requested_s,7.5)
        self.assertEqual(session.controller.seen[0],7)
        snapshot = session.controller
        snapshot.seen = ()
        self.assertEqual(session.controller.remaining,16)
        result = session.apply(Decision(Answer("A",(7,),"",.9),None))
        self.assertEqual(result["evidence"],[7])
        self.assertTrue(session.controller.finished)

    def test_failed_batch_does_not_consume_budget(self):
        reader = GridReader()
        session = ObservationSession(reader)
        before = session.frames
        for mode in ("fail","incomplete"):
            setattr(reader,mode,True)
            with self.assertRaises((OSError,ValueError)):
                session.apply(inspect(10,20))
            setattr(reader,mode,False)
            self.assertEqual(session.frames,before)
            self.assertEqual(session.controller.rounds,1)
            self.assertEqual(session.controller.remaining,16)

    def test_distinct_requests_cannot_reuse_decoded_frames(self):
        session = ObservationSession(GridReader())
        # Four distinct times, but each maps to the already-seen frame at 7s.
        with self.assertRaises(ValueError):
            session.apply(inspect(7.1,7.9))
        self.assertEqual(session.controller.remaining,16)

    def test_success_then_frame_cap(self):
        session = ObservationSession(GridReader())
        session.apply(inspect(8,20,8))
        session.apply(inspect(78,90,8))
        self.assertEqual(len(session.frames),24)
        self.assertTrue(session.controller.forced)
        with self.assertRaises(ValueError):
            session.apply(inspect(30,40))
        result = session.apply(Decision(Answer("B",(),"unknown",.1),None))
        self.assertTrue(result["forced"])

    def test_initial_short_clip_fails_without_session(self):
        reader = GridReader()
        reader.duration_s = 1
        with self.assertRaises(ValueError):
            ObservationSession(reader)
