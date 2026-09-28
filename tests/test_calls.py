import json
from pathlib import Path
import tempfile
import unittest

from watch_wisely.backend import ModelResponse
from watch_wisely.calls import CallStopped, DecisionCaller
from watch_wisely.controller import Controller


def response(action, confidence=.9):
    return ModelResponse(json.dumps({"option":"A", "evidence":[7.5],
                         "missing_evidence":"", "confidence":confidence,
                         "action":action}), 100, 20, None, 1.0)


class ScriptedBackend:
    """Test fixture only; these responses are not experimental predictions."""
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        result = next(self.outputs)
        if isinstance(result, Exception):
            raise result
        return result


class CallTests(unittest.TestCase):
    def test_answer_and_trace_without_state_mutation(self):
        backend = ScriptedBackend([response({"name":"answer"})])
        caller = DecisionCaller(backend)
        controller = Controller(120)
        decision = caller.decide([], controller)
        self.assertFalse(controller.finished)
        self.assertEqual(controller.rounds, 1)
        self.assertEqual(decision.candidate.option, "A")
        with self.assertRaises(CallStopped):
            caller.decide([], controller)
        self.assertEqual(backend.calls, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"question.jsonl"
            caller.write_trace(path)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "accepted")
            self.assertEqual(record["response"]["input_tokens"], 100)
            self.assertIsNone(record["response"]["visual_tokens"])
            with self.assertRaises(FileExistsError):
                caller.write_trace(path)

    def test_backend_and_parser_failures_count_once(self):
        for output, status in [(RuntimeError("backend unavailable"), "backend_error"),
                               (ModelResponse("not json", 10, 2, None, 1), "decision_rejected")]:
            with self.subTest(status=status):
                backend = ScriptedBackend([output])
                caller = DecisionCaller(backend)
                controller = Controller(120)
                for _ in range(2):
                    with self.assertRaises(CallStopped):
                        caller.decide([], controller)
                self.assertEqual(backend.calls, 1)
                self.assertEqual(len(caller.records), 1)
                self.assertEqual(caller.records[0].status, status)
                self.assertIsNotNone(caller.records[0].error)
                self.assertEqual(controller.rounds, 1)

    def test_semantic_rejection_does_not_spend_frames(self):
        for action,confidence in [({"name":"answer"}, .7),
                                  ({"name":"inspect","start":0,"end":120,"k":8}, .9)]:
            caller = DecisionCaller(ScriptedBackend([response(action,confidence)]))
            controller = Controller(120)
            before = (controller.seen, controller.rounds, controller.finished)
            with self.assertRaises(CallStopped):
                caller.decide([], controller)
            self.assertEqual(before,(controller.seen,controller.rounds,controller.finished))
            self.assertEqual(caller.records[0].status,"decision_rejected")

    def test_four_rounds_and_no_terminal_repair(self):
        for final in [response({"name":"answer"}, .1), ModelResponse("bad",None,None,None,1)]:
            actions = [{"name":"inspect","start":a,"end":b,"k":4}
                       for a,b in [(10,20),(30,40),(70,80)]]
            backend = ScriptedBackend([*(response(a) for a in actions),final])
            caller = DecisionCaller(backend)
            controller = Controller(120)
            for _ in range(3):
                d = caller.decide([],controller)
                with self.assertRaises(CallStopped):
                    caller.decide([],controller)
                controller.inspect(d.inspect.start,d.inspect.end,d.inspect.k)
            if final.raw_text == "bad":
                with self.assertRaises(CallStopped):
                    caller.decide([],controller)
            else:
                caller.decide([],controller)
            with self.assertRaises(CallStopped):
                caller.decide([],controller)
            self.assertEqual(backend.calls,4)
            self.assertEqual(len(caller.records),4)
            self.assertTrue(caller.records[-1].forced_answer)

    def test_frame_cap_rejects_another_inspection(self):
        actions = [{"name":"inspect","start":a,"end":b,"k":8}
                   for a,b in [(10,20),(70,80),(30,40)]]
        backend = ScriptedBackend([response(a) for a in actions])
        caller = DecisionCaller(backend)
        controller = Controller(120)
        for _ in range(2):
            d = caller.decide([],controller)
            controller.inspect(d.inspect.start,d.inspect.end,d.inspect.k)
        with self.assertRaises(CallStopped):
            caller.decide([],controller)
        self.assertEqual(len(controller.seen),24)
        self.assertEqual(backend.calls,3)
        self.assertTrue(caller.records[-1].forced_answer)
