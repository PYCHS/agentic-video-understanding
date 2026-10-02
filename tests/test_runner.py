import json
import unittest

from watch_wisely.backend import ModelResponse
from watch_wisely.runner import run_question
from test_calls import ScriptedBackend
from test_session import GridReader


def reply(action, confidence=.9):
    return ModelResponse(json.dumps({"option":"A", "evidence":[7],
                         "missing_evidence":"", "confidence":confidence,
                         "action":action}), 100, 20, None, 1)


def inspect(a,b,k=8):
    return reply({"name":"inspect","start":a,"end":b,"k":k})


class RunnerTests(unittest.TestCase):
    def test_inspect_then_answer_preserves_history_and_actual_times(self):
        backend = ScriptedBackend([inspect(8,20),reply({"name":"answer"})])
        seen = []
        def messages(session, history):
            seen.append((len(session.frames),len(history),session.controller.seen[0]))
            return []
        result = run_question(GridReader(),backend,messages)
        self.assertEqual(result.status,"answered")
        self.assertEqual(seen,[(8,0,7),(16,1,7)])
        self.assertEqual(result.answer["evidence"],[7])
        self.assertEqual(len(result.calls),2)
        self.assertEqual(result.observations[0].requested_s[0],7.5)
        self.assertEqual(result.observations[0].actual_s[0],7)

    def test_decode_collision_ends_question_without_retry(self):
        backend = ScriptedBackend([inspect(7.1,7.9,4),reply({"name":"answer"})])
        result = run_question(GridReader(),backend,lambda s,h: [])
        self.assertEqual(result.status,"failed")
        self.assertEqual(result.failed_stage,"apply")
        self.assertIsNone(result.answer)
        self.assertEqual(backend.calls,1)
        self.assertEqual(len(result.observations),1)
        self.assertEqual(result.calls[0].status,"accepted")  # Valid action, failed decoding.

    def test_always_24_and_terminal_invalid_response(self):
        for final, status in [(reply({"name":"answer"},.1),"answered"),
                              (ModelResponse("broken",None,None,None,1),"failed")]:
            backend = ScriptedBackend([inspect(8,20),inspect(78,90),final])
            result = run_question(GridReader(),backend,lambda s,h: [],always_use_24=True)
            self.assertEqual(result.status,status)
            self.assertEqual(backend.calls,3)
            self.assertEqual(sum(len(o.frame_ids) for o in result.observations),24)
            self.assertTrue(result.calls[-1].forced_answer)

    def test_survey_and_message_failures_make_no_model_calls(self):
        reader = GridReader()
        reader.duration_s = 1
        backend = ScriptedBackend([])
        result = run_question(reader,backend,lambda s,h: [])
        self.assertEqual(result.failed_stage,"survey")
        self.assertEqual(result.observations,())
        def broken_messages(session, history):
            raise ValueError("cannot build prompt")
        result = run_question(GridReader(),backend,broken_messages)
        self.assertEqual(result.failed_stage,"messages")
        self.assertEqual(len(result.observations),1)
        self.assertEqual(backend.calls,0)

    def test_fourth_round_never_gets_a_repair_call(self):
        backend = ScriptedBackend([inspect(8,20,4),inspect(30,40,4),inspect(78,88,4),
                                   ModelResponse("broken",None,None,None,1)])
        result = run_question(GridReader(),backend,lambda s,h: [])
        self.assertEqual(result.status,"failed")
        self.assertEqual(backend.calls,4)
        self.assertEqual(len(result.observations),4)
        self.assertTrue(result.calls[-1].forced_answer)
