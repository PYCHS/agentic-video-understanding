from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import unittest

from watch_wisely.qwen import QwenBackend, QwenSettings


class TokenList(list):
    def tolist(self):
        return list(self)


class Inputs(dict):
    def to(self, device):
        self.device = device
        return self


class QwenTests(unittest.TestCase):
    def settings(self, **overrides):
        values = dict(revision="a"*40,processor_revision="b"*40,
                      min_pixels=1024,max_pixels=200704,max_new_tokens=256)
        return QwenSettings(**(values | overrides))

    def test_settings_reject_unfrozen_or_invalid_values(self):
        for values in [dict(revision="main"),dict(processor_revision=None),
                       dict(min_pixels=True),dict(max_new_tokens=0),
                       dict(min_pixels=300000),dict(device="auto")]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                self.settings(**values)

    def test_greedy_generation_slices_prompt_and_counts_tokens(self):
        # Contract doubles only: no downloaded processor, checkpoint or GPU operation.
        inputs = Inputs(input_ids=[TokenList([1,99,99,2])])
        processor = Mock()
        processor.apply_chat_template.return_value = inputs
        processor.batch_decode.return_value = ['{"option":"A"}']
        model = Mock()
        model.eval.return_value = model
        model.config = SimpleNamespace(image_token_id=99)
        model.generate.return_value = [[1,99,99,2,10,11,12]]
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()),
                                inference_mode=lambda: nullcontext())
        backend = QwenBackend(model,processor,self.settings(),torch)
        result = backend.generate([{"role":"user","content":[]}])
        self.assertEqual((result.input_tokens,result.output_tokens,result.visual_tokens),(4,3,2))
        self.assertEqual(inputs.device,"cuda:0")
        self.assertEqual(model.generate.call_args.kwargs["do_sample"],False)
        self.assertEqual(model.generate.call_args.kwargs["num_beams"],1)
        self.assertEqual(model.generate.call_args.kwargs["max_new_tokens"],256)
        self.assertEqual(processor.batch_decode.call_args.args[0],[[10,11,12]])
        self.assertEqual(torch.cuda.synchronize.call_count,2)
        self.assertGreaterEqual(result.latency_ms,0)

    def test_backend_error_propagates_without_retry(self):
        model = Mock()
        model.eval.return_value = model
        processor = Mock()
        processor.apply_chat_template.side_effect = RuntimeError("processor failed")
        torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock()))
        backend = QwenBackend(model,processor,self.settings(),torch)
        with self.assertRaises(RuntimeError):
            backend.generate([])
        processor.apply_chat_template.assert_called_once()
        model.generate.assert_not_called()
