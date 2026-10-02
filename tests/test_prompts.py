import json
from types import SimpleNamespace
import unittest

from watch_wisely.controller import Controller
from watch_wisely.prompts import make_message_builder
from watch_wisely.calls import CallRecord
from watch_wisely.backend import ModelResponse


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.options = {k:"choice "+k for k in "DCBA"}
        self.builder = make_message_builder("What happens?",self.options)
        self.controller = Controller(120)
        self.frames = tuple(SimpleNamespace(timestamp_s=t, image=object())
                            for t in self.controller.seen)
        self.session = SimpleNamespace(controller=self.controller,frames=self.frames)

    def test_images_follow_exact_labels_and_option_order(self):
        self.options["A"] = "changed outside builder"
        messages = self.builder(self.session,())
        content = messages[1]["content"]
        state = json.loads(content[0]["text"])
        self.assertEqual(list(state["options"]),list("ABCD"))
        self.assertEqual(state["options"]["A"],"choice A")
        self.assertEqual(state["remaining_frames"],16)
        self.assertEqual(len(content),17)
        for index,frame in enumerate(self.frames):
            self.assertEqual(content[1+2*index]["text"],f"[t = {frame.timestamp_s!r} s]")
            self.assertIs(content[2+2*index]["image"],frame.image)
        self.assertFalse(state["forced_answer"])

    def test_sorted_images_preserve_precise_timestamps(self):
        self.session.frames = tuple(reversed(self.frames))
        self.frames[0].timestamp_s = .12345678912345678
        label = self.builder(self.session,())[1]["content"][1]["text"]
        self.assertEqual(float(label[5:-3]),self.frames[0].timestamp_s)

    def test_forced_state_and_history_are_supplied(self):
        self.controller.inspect(10,20,8)
        self.controller.inspect(70,80,8)
        history = tuple(CallRecord(i,i,False,"accepted",None,
                        ModelResponse("previous",None,None,None,1),1) for i in (1,2))
        state = json.loads(self.builder(self.session,history)[1]["content"][0]["text"])
        self.assertTrue(state["forced_answer"])
        self.assertEqual(state["remaining_frames"],0)
        self.assertEqual(state["previous_decisions"],["previous","previous"])

    def test_invalid_inputs_history_and_finished_session(self):
        for q,options in [("",self.options),("q",{"A":"a"}),("q",{**self.options,"gold":"A"})]:
            with self.assertRaises(ValueError):
                make_message_builder(q,options)
        with self.assertRaises(ValueError):
            self.builder(self.session,(object(),))
        self.controller.finished = True
        with self.assertRaises(ValueError):
            self.builder(self.session,())
