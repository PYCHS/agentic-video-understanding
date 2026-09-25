"""Local video reader with explicit timestamp mapping and no model dependency."""
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import math

import av
from PIL import Image


@dataclass(frozen=True)
class SelectedFrame:
    frame_id: int
    requested_s: float
    timestamp_s: float
    image: Image.Image


def resize_frame(image: Image.Image) -> Image.Image:
    """RGB, 448px long side, nearest integer short side; no crop or padding."""
    width, height = image.size
    scale = 448 / max(width, height)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.convert("RGB").resize(size, Image.Resampling.LANCZOS)


class VideoReader:
    """Index the first video stream, then decode requested frames in order.

    Times are seconds relative to the first presented frame. A request selects
    the frame displayed at that time, not a rounded nominal-FPS index. IDs are
    zero-based presentation-order indices, scoped to this file and video stream.
    This reference reader scans sequentially; it is not a fast seeking backend.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("expected a local video file")
        self._signature = self._file_signature()
        times = []
        last_duration = 0.0
        with av.open(str(self.path)) as container:
            if not container.streams.video:
                raise ValueError("file has no video stream")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                    raise ValueError("video has corrupt frames or missing timestamps")
                time = float(frame.pts * frame.time_base)
                if not math.isfinite(time) or (times and time <= times[-1]):
                    raise ValueError("presentation timestamps must strictly increase")
                times.append(time)
                last_duration = float(frame.duration * frame.time_base)
            if not times:
                raise ValueError("video has no decoded frames")
            # Do not use total container duration: an audio track may be longer.
            if last_duration > 0:
                end = times[-1] + last_duration
            elif stream.duration is not None and stream.start_time is not None:
                end = float((stream.start_time + stream.duration) * stream.time_base)
            else:
                raise ValueError("cannot determine final video frame duration")
        if not math.isfinite(end) or end <= times[-1]:
            raise ValueError("invalid video end time")
        self._origin_s = times[0]
        self.timestamps = tuple(t - times[0] for t in times)
        self.duration_s = end - times[0]
        if self._signature != self._file_signature():
            raise ValueError("video changed while indexing")

    def _file_signature(self):
        stat = self.path.stat()
        return stat.st_size, stat.st_mtime_ns

    def read(self, timestamps, *, seen_frame_ids=()) -> tuple[SelectedFrame, ...]:
        """Reject a whole batch if timestamps resolve to repeated/seen frames.

        No reader state is mutated. Callers add returned IDs to their evidence
        only after the complete batch succeeds. Input order is preserved.
        """
        requested = tuple(timestamps)
        if not requested:
            raise ValueError("request at least one timestamp")
        if any(isinstance(t, bool) or not isinstance(t, (int, float))
               or not math.isfinite(t) or not 0 <= t < self.duration_s
               for t in requested):
            raise ValueError("timestamps must be finite seconds in [0, duration)")
        ids = tuple(bisect_right(self.timestamps, t)-1 for t in requested)
        if len(set(ids)) != len(ids) or set(ids).intersection(seen_frame_ids):
            raise ValueError("requested timestamps resolve to duplicate or seen frames")
        if self._signature != self._file_signature():
            raise ValueError("video changed since indexing")
        wanted = set(ids)
        images = {}
        with av.open(str(self.path)) as container:
            for index, frame in enumerate(container.decode(container.streams.video[0])):
                if index in wanted:
                    if (frame.is_corrupt or frame.pts is None or frame.time_base is None
                        or float(frame.pts * frame.time_base)-self._origin_s
                            != self.timestamps[index]):
                        raise ValueError("decoded frame no longer matches index")
                    images[index] = resize_frame(frame.to_image())
                if index == max(ids):
                    break
        if len(images) != len(ids) or self._signature != self._file_signature():
            raise ValueError("video changed or requested frames could not be decoded")
        return tuple(SelectedFrame(i, t, self.timestamps[i], images[i])
                     for i,t in zip(ids, requested))
