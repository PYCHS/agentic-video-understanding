"""Validated, gold-free batch execution and comparable fixed-frame inference."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

from .actions import parse_decision
from .baselines import observe_uniform
from .calls import CallRecord
from .prompts import SYSTEM_PROMPT, make_message_builder
from .results import write_question_result
from .runner import Observation, RunResult, run_question

METHODS = ('Adaptive-24', 'Adaptive-always-24', 'Uniform-8', 'Uniform-16', 'Uniform-24')


@dataclass(frozen=True)
class Question:
    question_id: str
    video_id: str
    video_path: Path
    question: str
    options: dict


def load_questions(path):
    """Read a JSON array. Gold labels may exist but never enter Question or prompts."""
    path = Path(path).resolve(strict=True)
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('manifest must be a nonempty JSON array')
    questions, seen, videos = [], set(), {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('each manifest row must be an object')
        for key in ('question_id', 'video_id', 'video_path', 'question'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'nonempty text required: {key}')
        if row['question_id'] in seen:
            raise ValueError('duplicate question_id')
        seen.add(row['question_id'])
        options = row.get('options')
        if not isinstance(options, dict):
            raise ValueError('options must map A/B/C/D to text')
        make_message_builder(row['question'], options)
        video = (path.parent / row['video_path']).resolve(strict=True)
        if not video.is_file():
            raise ValueError('video_path must name a local file')
        if row['video_id'] in videos and videos[row['video_id']] != video:
            raise ValueError('one video_id maps to multiple files')
        videos[row['video_id']] = video
        questions.append(Question(row['question_id'], row['video_id'], video,
                                  row['question'], dict(options)))
    return tuple(questions)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run_uniform(reader, backend, question, options, count):
    """One answer call with the same strict evidence parser, no adaptive stop rule."""
    started = perf_counter()
    calls, observations = [], []
    stage = 'survey'
    try:
        observed = observe_uniform(reader, count)
        frames = observed.frames
        observations.append(Observation(1, tuple(f.frame_id for f in frames),
                            tuple(f.requested_s for f in frames),
                            tuple(f.timestamp_s for f in frames)))
        stage = 'messages'
        content = [{'type': 'text', 'text': json.dumps(dict(
            question=question, options=options, duration_s=reader.duration_s,
            forced_answer=True, remaining_frames=0, remaining_rounds=0))}]
        for frame in sorted(frames, key=lambda f: f.timestamp_s):
            content.extend([{'type': 'text', 'text': f'[t = {frame.timestamp_s!r} s]'},
                            {'type': 'image', 'image': frame.image}])
        # Preserve shared output/evidence rules; explicitly override adaptive-only rules.
        messages = [{'role': 'system', 'content': SYSTEM_PROMPT +
                    '\nFixed-frame baseline: these are all available frames. Answer now; do not inspect.'},
                    {'role': 'user', 'content': content}]
        stage = 'decision'
        response = None
        call_started = perf_counter()
        status = 'backend_error'
        try:
            response = backend.generate(messages)
            status = 'decision_rejected'
            decision = parse_decision(response.raw_text, duration=reader.duration_s,
                                      observed_timestamps=[f.timestamp_s for f in frames])
            if decision.inspect is not None:
                raise ValueError('fixed-frame baseline cannot inspect')
        except Exception as exc:
            calls.append(CallRecord(1, 1, True, status, f'{type(exc).__name__}: {exc}',
                                    response, (perf_counter()-call_started)*1000))
            raise
        calls.append(CallRecord(1, 1, True, 'accepted', None, response,
                                (perf_counter()-call_started)*1000))
        candidate = decision.candidate
        answer = dict(option=candidate.option, evidence=list(candidate.evidence), forced=True,
                      evidence_sufficient=bool(candidate.evidence) and
                      not candidate.missing_evidence.strip() and candidate.confidence >= .8,
                      unique_frames=len(frames), rounds=1)
        return RunResult('answered', answer, None, None, tuple(calls), tuple(observations),
                         (perf_counter()-started)*1000)
    except Exception as exc:
        return RunResult('failed', None, f'{type(exc).__name__}: {exc}', stage,
                         tuple(calls), tuple(observations), (perf_counter()-started)*1000)


def execute(questions, backend, output, *, run_id, methods, provenance,
            reader_factory=None, synthetic=False):
    """Exclusive output directory, stable filenames, full failure denominator.

    One model is reused. Readers are isolated by method; failed questions are saved
    and execution continues. A failed output publication aborts the batch.
    """
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError('run_id must be nonempty text')
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('select unique supported methods')
    if type(synthetic) is not bool:
        raise ValueError('synthetic must be boolean')
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError('questions must be nonempty with unique IDs')
    if reader_factory is None:
        from .video import VideoReader
        reader_factory = VideoReader
    # Hash inputs before creating output or making any inference calls.
    hashes = {str(q.video_path): sha256_file(q.video_path) for q in questions}
    metadata = dict(schema_version=1, run_id=run_id, methods=list(methods), synthetic=synthetic,
                    provenance=provenance, video_sha256=hashes,
                    questions=[dict(question_id=q.question_id, video_id=q.video_id) for q in questions])
    encoded = json.dumps(metadata, indent=2, allow_nan=False) + '\n'
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'run.json').write_text(encoded, encoding='utf-8')
    counts = {'answered': 0, 'failed': 0}
    for index, question in enumerate(questions):
        for method in methods:
            started = perf_counter()
            try:
                reader = reader_factory(question.video_path)
                if method.startswith('Uniform-'):
                    result = run_uniform(reader, backend, question.question, question.options,
                                         int(method.split('-')[1]))
                else:
                    result = run_question(reader, backend,
                        make_message_builder(question.question, question.options),
                        always_use_24=method == 'Adaptive-always-24')
            except Exception as exc:
                result = RunResult('failed', None, f'{type(exc).__name__}: {exc}', 'survey',
                                   (), (), (perf_counter()-started)*1000)
            result = replace(result, wall_latency_ms=(perf_counter()-started)*1000)
            if sha256_file(question.video_path) != hashes[str(question.video_path)]:
                raise ValueError('video changed during experiment; discard this run')
            write_question_result(output / method / f'{index:06d}.json', result,
                run_id=run_id, question_id=question.question_id, video_id=question.video_id,
                method=method, synthetic=synthetic)
            counts[result.status] += 1
    (output / 'summary.json').write_text(json.dumps(counts, indent=2)+'\n', encoding='utf-8')
    return counts
