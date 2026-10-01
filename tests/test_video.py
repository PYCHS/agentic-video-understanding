"""Real encode/decode tests using tiny lossless clips made in a temp directory."""
from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

try:
    import av
    from PIL import Image
except ImportError:
    av = None

if av is not None:
    from watch_wisely.video import VideoReader, resize_frame


@unittest.skipIf(av is None, "install watch-wisely[video] to run decoder tests")
class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"fixture.mkv"

    def make_clip(self, pts=(0, 1, 2, 3)):
        colors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
        with av.open(str(self.path), "w") as out:
            stream = out.add_stream("ffv1", rate=4)
            stream.width, stream.height = 64, 32
            stream.pix_fmt = "bgr0"
            stream.codec_context.time_base = Fraction(1,4)
            for stamp, color in zip(pts, colors):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64,32), color))
                frame.pts, frame.time_base = stamp, Fraction(1,4)
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)

    def test_exact_mapping_pixels_and_order(self):
        self.make_clip()
        reader = VideoReader(self.path)
        self.assertEqual(reader.timestamps, (0, .25, .5, .75))
        self.assertEqual(reader.duration_s, 1)
        frames = reader.read([.99, 0, .25, .5])
        self.assertEqual([f.frame_id for f in frames], [3,0,1,2])
        self.assertEqual(frames[0].image.size, (448,224))
        self.assertEqual(frames[2].image.getpixel((0,0)), (0,255,0))
        self.assertEqual(frames[0].timestamp_s, .75)
        self.assertEqual(frames[0].requested_s, .99)

    def test_variable_spacing_and_nonzero_start(self):
        self.make_clip((4,5,8,10))
        reader = VideoReader(self.path)
        self.assertEqual(reader.timestamps, (0,.25,1,1.5))
        self.assertEqual([f.frame_id for f in reader.read([.9,1,1.5])], [1,2,3])

    def test_collisions_and_bounds(self):
        self.make_clip()
        reader = VideoReader(self.path)
        for times in [[], [0,.1], [-.1], [1], [float("nan")], [float("inf")], [True]]:
            with self.assertRaises(ValueError):
                reader.read(times)
        with self.assertRaises(ValueError):
            reader.read([.26], seen_frame_ids={1})
        # Failed batches have not consumed frames.
        self.assertEqual(reader.read([.26])[0].frame_id, 1)

    def test_single_frame_clip(self):
        self.make_clip((0,))
        reader = VideoReader(self.path)
        self.assertEqual(reader.read([0])[0].frame_id, 0)
        with self.assertRaises(ValueError):
            reader.read([0,.1])

    def test_changed_file(self):
        self.make_clip()
        reader = VideoReader(self.path)
        with self.path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaises(ValueError):
            reader.read([0])

    def test_real_reader_session_uses_decoded_evidence(self):
        from watch_wisely.actions import Decision
        from watch_wisely.controller import Answer
        from watch_wisely.session import ObservationSession
        with av.open(str(self.path), "w") as out:
            stream = out.add_stream("ffv1", rate=4)
            stream.width, stream.height = 64, 32
            stream.pix_fmt = "bgr0"
            for _ in range(33):
                frame = av.VideoFrame.from_image(Image.new("RGB", (64,32), "red"))
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        session = ObservationSession(VideoReader(self.path))
        first = session.frames[0]
        self.assertNotEqual(first.requested_s, first.timestamp_s)
        result = session.apply(Decision(Answer("A", (first.timestamp_s,), "", .9), None))
        self.assertEqual(result["unique_frames"], 8)
        self.assertEqual(result["evidence"], [first.timestamp_s])

    def test_resize_portrait_square_and_rgb(self):
        for size, expected in [((30,60),(224,448)), ((12,12),(448,448)),
                               ((1000,333),(448,149))]:
            result = resize_frame(Image.new("L",size))
            self.assertEqual(result.size, expected)
            self.assertEqual(result.mode, "RGB")
