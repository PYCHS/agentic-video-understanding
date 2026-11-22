"""CPU integration checks for batching, provenance, gold isolation and failures."""
from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from watch_wisely.backend import ModelResponse
from watch_wisely.experiment import execute, load_questions, run_uniform
from evaluation.run_report import build_report


@dataclass(frozen=True)
class Frame:
    frame_id: int
    requested_s: float
    timestamp_s: float
    image: object = 'test-image'


class Reader:
    duration_s = 120
    def __init__(self, path=None):
        pass
    def read(self, timestamps, *, seen_frame_ids=()):
        frames = tuple(Frame(int(t*1000), t, t) for t in timestamps)
        if set(f.frame_id for f in frames).intersection(seen_frame_ids):
            raise ValueError('collision')
        return frames


class Backend:
    def __init__(self, inspect=False, invalid=False):
        self.messages = []
        self.inspect = inspect
        self.invalid = invalid
    def generate(self, messages):
        self.messages.append(messages)
        content = messages[1]['content']
        state = json.loads(content[0]['text'])
        evidence = float(next(x['text'] for x in content[1:] if x['type']=='text').split(' = ')[1].split(' s]')[0])
        action = {'name': 'answer'}
        if self.inspect and not state['forced_answer'] and state.get('round') == 1:
            action = dict(name='inspect', start=0, end=30, k=8)
        raw = 'invalid' if self.invalid else json.dumps(dict(option='A', evidence=[evidence],
                   missing_evidence='', confidence=.9, action=action))
        return ModelResponse(raw, 50, 20, 10, 1)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        ((self.root/'video.mp4').resolve()).write_bytes(b'fixture, not an actual video')
        self.row = dict(question_id='q1', video_id='v1', video_path='video.mp4',
                        question='What happens?', options=dict(zip('ABCD', ['a','b','c','d'])),
                        answer='D', secret_gold='NEVER_SEND_THIS')
        self.manifest = self.root/'questions.json'
        self.write([self.row])
    def write(self, rows):
        self.manifest.write_text(json.dumps(rows))
    def test_gold_isolation_and_paths(self):
        questions = load_questions(self.manifest)
        self.assertEqual(questions[0].video_path, (self.root/'video.mp4').resolve())
        self.assertFalse(hasattr(questions[0], 'answer'))
        backend = Backend()
        execute(questions, backend, self.root/'run', run_id='test',
                methods=['Adaptive-24','Uniform-24'], provenance={}, reader_factory=Reader,
                synthetic=True)
        self.assertNotIn('NEVER_SEND_THIS', json.dumps(backend.messages))
        with self.assertRaisesRegex(ValueError, 'synthetic'):
            build_report(self.root/'run', self.manifest, samples=10)
    def test_manifest_rejections(self):
        for rows in ([self.row, self.row], [], [{**self.row, 'options': {'A':'only'}}],
                     [{**self.row, 'video_path':'missing.mp4'}]):
            self.write(rows)
            with self.assertRaises((ValueError, FileNotFoundError)):
                load_questions(self.manifest)
    def test_baseline_uses_one_call_and_actual_budget(self):
        for n in (8,16,24):
            backend = Backend()
            result = run_uniform(Reader(),backend,'Question',self.row['options'],n)
            self.assertEqual(result.status, 'answered')
            self.assertEqual(result.answer['unique_frames'],n)
            self.assertEqual(len(backend.messages),1)
    def test_rejection_is_retained_with_no_retry(self):
        backend = Backend(invalid=True)
        counts = execute(load_questions(self.manifest), backend, self.root/'bad', run_id='bad',
                methods=['Uniform-24','Adaptive-24'], provenance={}, reader_factory=Reader)
        self.assertEqual(counts, dict(answered=0, failed=2))
        self.assertEqual(len(backend.messages),2)
        report = build_report(self.root/'bad', self.manifest, samples=10)
        self.assertEqual(report['baseline']['failed'],1)
        self.assertEqual(report['adaptive']['accuracy'],0)
    def test_inspection_then_answer_and_report(self):
        backend = Backend(inspect=True)
        counts = execute(load_questions(self.manifest),backend,self.root/'ok', run_id='ok',
                methods=['Uniform-24','Adaptive-24'], provenance={'checkpoint':'test'},
                reader_factory=Reader)
        self.assertEqual(counts['answered'],2)
        report = build_report(self.root/'ok',self.manifest,samples=10)
        self.assertEqual(report['costs']['Adaptive-24']['mean_observed_frames'],16)
        self.assertEqual(report['costs']['Adaptive-24']['mean_calls'],2)
        with self.assertRaises(FileExistsError):
            execute(load_questions(self.manifest),backend,self.root/'ok',run_id='ok',
                    methods=['Uniform-24'],provenance={},reader_factory=Reader)
    def test_reader_failure_continues_to_next_question(self):
        self.write([self.row, {**self.row, 'question_id':'q2'}])
        class BrokenReader(Reader):
            def read(self, *args, **kwargs):
                raise OSError('decode error')
        counts=execute(load_questions(self.manifest),Backend(),self.root/'fail',run_id='fail',
                       methods=['Adaptive-24'],provenance={},reader_factory=BrokenReader)
        self.assertEqual(counts['failed'],2)
        self.assertTrue((self.root/'fail'/'Adaptive-24'/'000001.json').exists())

    def test_uniform_rejects_inspection(self):
        class Inspecting(Backend):
            def generate(self, messages):
                return ModelResponse(json.dumps(dict(option='A', evidence=[], missing_evidence='',
                    confidence=.9, action=dict(name='inspect',start=0,end=30,k=8))),1,1,1,1)
        result=run_uniform(Reader(),Inspecting(),'Q',self.row['options'],24)
        self.assertEqual(result.status,'failed')
        self.assertEqual(result.calls[0].status,'decision_rejected')

    def test_cli_preflight_without_inference_dependencies(self):
        from contextlib import redirect_stdout
        from io import StringIO
        from watch_wisely.cli import main
        settings=self.root/'settings.json'
        settings.write_text(json.dumps(dict(revision='a'*40,processor_revision='b'*40,
            min_pixels=3136,max_pixels=200704,max_new_tokens=128)))
        output=self.root/'preflight'
        with redirect_stdout(StringIO()) as stdout:
            code=main(['--manifest',str(self.manifest),'--settings',str(settings),
                       '--output',str(output),'--run-id','preflight','--validate-only'])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(stdout.getvalue())['questions'],1)
        self.assertFalse(output.exists())

    def test_always_24_executes_three_rounds(self):
        class AlwaysBackend(Backend):
            def generate(self,messages):
                state=json.loads(messages[1]['content'][0]['text'])
                response=super().generate(messages)
                raw=json.loads(response.raw_text)
                if not state['forced_answer']:
                    raw['action']=dict(name='inspect',start=0 if state['round']==1 else 30,
                                       end=30 if state['round']==1 else 60,k=8)
                return ModelResponse(json.dumps(raw),50,20,10,1)
        execute(load_questions(self.manifest),AlwaysBackend(),self.root/'always',run_id='always',
                methods=['Adaptive-always-24'],provenance={},reader_factory=Reader,synthetic=True)
        result=json.loads((self.root/'always'/'Adaptive-always-24'/'000000.json').read_text())['result']
        self.assertEqual(result['status'],'answered')
        self.assertEqual(result['answer']['unique_frames'],24)
        self.assertEqual(len(result['calls']),3)
