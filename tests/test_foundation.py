import json
import tempfile
import unittest
from pathlib import Path
from watch_wisely.controller import Answer, Controller
from watch_wisely.sampling import uniform_timestamps, coarse_to_fine_timestamps
from watch_wisely.tracing import RoundTrace


class FoundationTests(unittest.TestCase):
    def test_uniform(self):
        self.assertEqual(uniform_timestamps(0, 8, 4), (1, 3, 5, 7))
        for args in [(0,0,8), (0,float("inf"),8), (0,8,True), (-1,8,4)]:
            with self.assertRaises(ValueError):
                uniform_timestamps(*args)

    def test_cap_and_forced_answer(self):
        c = Controller(120)
        c.inspect(10,20,8)
        c.inspect(70,80,8)
        self.assertEqual(len(c.seen),24)
        with self.assertRaises(ValueError):
            c.inspect(90,100,4)
        r = c.answer(Answer("B", (), "missing", 0.1))
        self.assertTrue(r["forced"])
        self.assertFalse(r["evidence_sufficient"])
        with self.assertRaises(ValueError):
            c.answer(Answer("B", (), "", 1))

    def test_round_limit(self):
        c = Controller(120)
        for a,b in [(10,20),(30,40),(70,80)]:
            c.inspect(a,b,4)
        self.assertEqual(len(c.seen),20)
        self.assertTrue(c.forced)
        with self.assertRaises(ValueError):
            c.inspect(90,100,4)

    def test_rejection_is_atomic(self):
        c = Controller(120)
        before = (c.seen,c.rounds)
        for args in [(0,120,8), (0,121,4), (10,20,16), (-1,1,4)]:
            with self.assertRaises(ValueError):
                c.inspect(*args)
            self.assertEqual((c.seen,c.rounds),before)
        c.inspect(10,20,8)
        c.inspect(30,40,4)
        with self.assertRaises(ValueError):
            c.inspect(80,90,8)

    def test_evidence_stop(self):
        c = Controller(120)
        for d in [Answer("A",(),"",.9), Answer("A",(999,),"",.9),
                  Answer("A",(c.seen[0],),"missing",.9),
                  Answer("A",(c.seen[0],),"",.79),
                  Answer("A",(c.seen[0],),"",float("nan"))]:
            with self.assertRaises(ValueError):
                c.answer(d)
        self.assertFalse(c.answer(Answer("A",(c.seen[0],),"",.8))["forced"])

    def test_always_24_feasibility(self):
        c = Controller(120,always_use_24=True)
        c.inspect(10,20,4)
        c.inspect(30,40,4)
        with self.assertRaises(ValueError):
            c.inspect(70,80,4)
        with self.assertRaises(ValueError):
            c.answer(Answer("A",(c.seen[0],),"",1))
        c.inspect(70,80,8)
        self.assertEqual(len(c.seen),24)

    def test_fixed_baseline(self):
        result=coarse_to_fine_timestamps((1,3,5),[(1,0),(1,0),(-1,0)],4)
        self.assertEqual(result,(1,3,3.25,3.75,4.25,4.75,5))
        with self.assertRaises(ValueError):
            coarse_to_fine_timestamps((1,3),[(0,0),(1,0)])

    def test_trace_is_jsonl_with_null_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"trace.jsonl"
            t=RoundTrace("synthetic","adaptive_24",1,(1.0,),1,0,synthetic=True)
            t.append(path)
            t.append(path)
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows),2)
            self.assertIsNone(rows[0]["visual_tokens"])


if __name__ == "__main__":
    unittest.main()
