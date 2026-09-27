import json
import unittest

from watch_wisely.actions import DecisionError, MAX_RESPONSE_CHARS, parse_decision
from watch_wisely.controller import Controller


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.controller = Controller(120)
        self.data = {"option": "B", "evidence": [7.5], "missing_evidence": "",
                     "confidence": .8, "action": {"name": "answer"}}

    def parse(self, data=None, raw=None):
        return parse_decision(raw if raw is not None else json.dumps(data or self.data),
                              duration=self.controller.duration,
                              observed_timestamps=self.controller.seen)

    def test_answer_reaches_controller(self):
        decision = self.parse()
        self.assertIsNone(decision.inspect)
        self.assertEqual(self.controller.answer(decision.candidate)["option"], "B")

    def test_inspect_preserves_candidate_without_mutating_state(self):
        self.data["missing_evidence"] = "Need to see what happens next."
        self.data["action"] = {"name": "inspect", "start": 10, "end": 20, "k": 4}
        decision = self.parse()
        self.assertEqual(self.controller.rounds, 1)
        self.assertEqual(decision.candidate.option, "B")
        request = decision.inspect
        self.controller.inspect(request.start, request.end, request.k)
        self.assertEqual(self.controller.rounds, 2)

    def test_rejects_wrappers_duplicates_and_non_json(self):
        good = json.dumps(self.data)
        for raw in ["", "[]", "null", "```json\n"+good+"\n```", good+good,
                    '{"option":"A",'+good[1:], good.replace('0.8', 'NaN'),
                    good.replace('0.8', 'Infinity'), good.replace('0.8', '1e999'),
                    good.replace('{"name": "answer"}', '{"name":"inspect","name":"answer"}'),
                    '['*2000+']'*2000, ' '*MAX_RESPONSE_CHARS+'{}']:
            with self.subTest(raw=raw[:80]), self.assertRaises(DecisionError):
                self.parse(raw=raw)

    def test_rejects_fields_and_types(self):
        for key,value in [("option", "E"), ("option", ["A"]), ("confidence", True),
                          ("confidence", "0.8"), ("confidence", -1),
                          ("evidence", [False]), ("evidence", [7.5,7.5]),
                          ("evidence", [8]), ("evidence", "7.5"),
                          ("missing_evidence", []), ("extra", 1),
                          ("action", {"name":"answer","option":"C"})]:
            with self.subTest(key=key,value=value), self.assertRaises(DecisionError):
                self.parse({**self.data, key:value})
        for key in self.data:
            with self.subTest(missing=key), self.assertRaises(DecisionError):
                self.parse({k:v for k,v in self.data.items() if k != key})

    def test_rejects_bad_inspections(self):
        good = {"name":"inspect", "start":10, "end":20, "k":4}
        for field,value in [("start",True), ("start",-1), ("end",10), ("end",121),
                            ("k",4.0), ("k",True), ("k",16), ("name","execute")]:
            with self.subTest(field=field,value=value), self.assertRaises(DecisionError):
                self.parse({**self.data, "action":{**good,field:value}})

    def test_controller_remains_authority_for_stopping_and_limits(self):
        decision = self.parse({**self.data, "confidence":.7})
        with self.assertRaises(ValueError):
            self.controller.answer(decision.candidate)
        self.controller.inspect(10,20,8)
        self.controller.inspect(70,80,8)
        decision = self.parse({**self.data,"action":{"name":"inspect","start":30,"end":40,"k":4}})
        before = (self.controller.seen,self.controller.rounds,self.controller.finished)
        with self.assertRaises(ValueError):
            request = decision.inspect
            self.controller.inspect(request.start,request.end,request.k)
        self.assertEqual(before,(self.controller.seen,self.controller.rounds,self.controller.finished))
        terminal = self.parse({**self.data,"evidence":[],"confidence":.1,"missing_evidence":"Unknown"})
        self.assertTrue(self.controller.answer(terminal.candidate)["forced"])
