# Reading frames from a video

Install the optional video dependencies with `python -m pip install -e ".[video]"`.
This runs on CPU and does not load a model or download a dataset.

```python
from watch_wisely.sampling import uniform_timestamps
from watch_wisely.video import VideoReader

video = VideoReader("data/example.mp4")
times = uniform_timestamps(0, video.duration_s, 8)
frames = video.read(times)
seen = {frame.frame_id for frame in frames}
# For a later batch: video.read(new_times, seen_frame_ids=seen)
```

The reader uses the first video stream. It never decodes audio or subtitles. Times start
at the first presented video frame, even if the file itself starts at a nonzero timestamp.
A request selects the frame displayed at that time: the last frame whose presentation time
is at or before the request. The end of the video is excluded. This works with uneven frame
spacing and does not rely on average FPS. Missing or non-increasing timestamps are rejected.

Each result includes the requested time, actual frame time, frame ID and RGB image. Frame IDs
are presentation-order indices within this file, not IDs that can be compared across videos.
Use the actual time for evidence labels and keep both times in logs. The final frame's duration
defines the end; video-stream duration is a fallback. Unknown end times cause an error rather
than a guess based on an audio track or average frame rate.

Two requests that land on the same frame are rejected, as are already-seen frame IDs. A failed
batch returns nothing and does not change reader state. A very short clip may not have enough
distinct frames for an 8-frame survey; do not duplicate frames to fill the batch. The evaluation
exclusion rule still needs to be frozen. The caller must connect this check to the controller
before committing a new observation; that integration is not implemented yet.

Images use a 448-pixel long side and a rounded short side, with RGB conversion and Lanczos
resizing. There is no crop or padding. The short side is not forced to a multiple of 32;
model-processor alignment and pixel limits are still pending. Rotation metadata and unusual
pixel-aspect-ratio sources have not been validated; use ordinary square-pixel, upright clips
for now and check benchmark source formats before evaluation.

This is a simple reference reader: it scans once to index timestamps, stores only that index,
then scans again for each requested batch. It keeps resized images only for the requested
frames. It may decode intervening frames, but those frames never reach the VLM. This is slower
than indexed seeking; include all decoding time in future latency reports. The file must stay
unchanged during a session (size/modification time are checked; this is not a content hash).

Tests create lossless colored clips locally and check exact frame selection, uneven timestamps,
nonzero starts, repeated frames, video boundaries, short clips and resize dimensions. These
are reader tests, not QA experiments. Run `python -m unittest discover -s tests -v` after
installing the video extra; without it, reader tests are explicitly skipped.

API references: [PyAV containers](https://pyav.org/docs/stable/api/container.html),
[frame timestamps](https://pyav.org/docs/stable/api/frame.html), and
[Pillow resize](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize).
