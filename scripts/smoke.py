"""Synthetic controller exercise only: no model, dataset, or accuracy measurement."""
import json
from watch_wisely.controller import Answer, Controller

c = Controller(120)
c.inspect(10, 20, 8)
c.inspect(70, 80, 8)
result = c.answer(Answer("A", (c.seen[0],), "synthetic unresolved evidence", 0.2))
print(json.dumps({"synthetic": True, "model_calls": 0, "controller": result}, indent=2))
