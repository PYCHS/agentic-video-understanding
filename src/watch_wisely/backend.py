"""Boundary for future real model adapters; no simulated inference backend."""
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelResponse:
    raw_text: str
    input_tokens: int | None
    output_tokens: int | None
    visual_tokens: int | None
    latency_ms: float


class Backend(Protocol):
    def generate(self, messages: list[dict]) -> ModelResponse:
        """Run one measured call with frozen processor and decoding settings."""
        ...
