"""Missing measurements remain null, never fabricated zeroes."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RoundTrace:
    question_id: str
    method: str
    round_index: int
    timestamps_s: tuple[float, ...]
    unique_frames: int
    vlm_calls: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    visual_tokens: int | None = None
    latency_ms: float | None = None
    synthetic: bool = False

    def append(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(self), allow_nan=False) + "\n")
