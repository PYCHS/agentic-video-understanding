"""Publish a completed question record without replacing an existing result."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from .runner import RunResult


def write_question_result(path, result: RunResult, *, run_id: str,
                          question_id: str, video_id: str, method: str,
                          synthetic: bool) -> Path:
    """Save runner output, including failures. This is not a provenance validator.

    Publication requires a filesystem supporting hard links. The final filename
    appears only after the complete temporary file has been flushed and synced.
    """
    identity = dict(run_id=run_id, question_id=question_id, video_id=video_id,
                    method=method)
    if any(not isinstance(v, str) or not v.strip() for v in identity.values()):
        raise ValueError("run, question, video and method identifiers must be nonempty text")
    if type(synthetic) is not bool:
        raise ValueError("synthetic must be explicitly true or false")
    if not isinstance(result, RunResult):
        raise TypeError("result must be a RunResult")
    payload = dict(schema_version=1, **identity, synthetic=synthetic,
                   result=asdict(result))
    # Reject unencodable values before creating any files or directories.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=path.parent, prefix=".question-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() refuses an existing destination, even in a race.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path
