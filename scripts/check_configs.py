"""Check proposal constants, not preregistration readiness."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
c = json.loads((root / "configs/main.json").read_text())
assert c["agent"] == {"initial_frames": 8, "frame_cap": 24, "max_rounds": 4,
                       "inspect_batches": [4, 8], "stop_confidence": 0.8}
assert c["model"]["long_side_px"] == 448
assert not c["inputs"]["audio"] and not c["inputs"]["subtitles"]
assert c["clip"]["fine_frames"] + c["clip"]["coarse_frames"] == 24
p = json.loads((root / "configs/pilot.json").read_text())
assert p["target_questions"] == 20 and not p["pool_across_backbones"]
print("Proposal constants OK. Revisions, processor settings and manifests remain pending.")
