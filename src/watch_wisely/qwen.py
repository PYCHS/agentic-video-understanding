"""Qwen3-VL adapter. Real checkpoint/processor integration is not validated yet."""
from dataclasses import dataclass
import re
from time import perf_counter

from .backend import ModelResponse


@dataclass(frozen=True)
class QwenSettings:
    revision: str
    processor_revision: str
    min_pixels: int
    max_pixels: int
    max_new_tokens: int
    device: str = "cuda:0"

    def __post_init__(self):
        for revision in (self.revision, self.processor_revision):
            if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}",revision):
                raise ValueError("pin model and processor to full commit hashes")
        for value in (self.min_pixels,self.max_pixels,self.max_new_tokens):
            if type(value) is not int or value <= 0:
                raise ValueError("pixel budgets and output limit must be positive integers")
        if self.min_pixels > self.max_pixels:
            raise ValueError("min_pixels must not exceed max_pixels")
        if not re.fullmatch(r"cuda:\d+",self.device):
            raise ValueError("this initial adapter supports one explicit CUDA device")


class QwenBackend:
    MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

    def __init__(self, model, processor, settings: QwenSettings, torch_module):
        self.model = model.eval()
        self.processor = processor
        self.settings = settings
        self._torch = torch_module

    @classmethod
    def load(cls, settings: QwenSettings, *, local_files_only=True):
        """Default to cached files; downloading must be explicitly enabled by the caller.

        No quantization, offloading, automatic device selection or hidden retries.
        """
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(
            cls.MODEL_ID, revision=settings.processor_revision,
            min_pixels=settings.min_pixels, max_pixels=settings.max_pixels,
            local_files_only=local_files_only)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            cls.MODEL_ID, revision=settings.revision, dtype=torch.bfloat16,
            local_files_only=local_files_only).to(settings.device)
        return cls(model,processor,settings,torch)

    def generate(self, messages: list[dict]) -> ModelResponse:
        torch = self._torch
        torch.cuda.synchronize(self.settings.device)
        started = perf_counter()
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.settings.device)
        if len(inputs["input_ids"]) != 1:
            raise ValueError("only one question per call is supported")
        ids = inputs["input_ids"][0].tolist()
        image_token_id = getattr(self.model.config,"image_token_id",None)
        visual_tokens = ids.count(image_token_id) if image_token_id is not None else None
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, num_beams=1,
                                         max_new_tokens=self.settings.max_new_tokens,
                                         return_dict_in_generate=False)
        generated = output[0][len(ids):]
        text = self.processor.batch_decode([generated], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        torch.cuda.synchronize(self.settings.device)
        return ModelResponse(text,len(ids),len(generated),visual_tokens,
                             (perf_counter()-started)*1000)
