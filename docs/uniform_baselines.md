# Uniform baseline frames

I use a separate midpoint grid for each budget, over the full video duration.
Uniform-16 is not Uniform-8 with eight extra frames: changing the budget changes
the grid. All three methods use the same decoder and 448px RGB preprocessing.

```python
from watch_wisely.baselines import observe_uniform
from watch_wisely.video import VideoReader

reader = VideoReader("data/example.mp4")
observation = observe_uniform(reader, 24)  # also accepts 8 or 16
```

Each returned frame carries its requested time, actual decoded time, frame ID
and image. Evidence and image labels must use the actual time. Duplicate frames,
incomplete batches and decode errors fail the question; there is no padding or
automatic retry. The evaluation runner must retain these failures when scoring.

`selection_latency_ms` covers grid construction, decoding and batch validation.
It excludes the reader's initial indexing pass and any model work. Record those
separately when measuring end-to-end latency; do not label this value model latency.

This implements frame selection only. Baseline answer prompts, model calls,
durable traces and the CLIP baseline are still pending. Tests use synthetic
timelines and generated clips, not Video-MME results.
