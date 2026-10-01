# Checking a model decision

The model must return one JSON object, without a Markdown code block or surrounding text.
Here is a made-up inspection request, assuming 7.5 seconds is an observed timestamp:

```json
{
  "option": "B",
  "evidence": [7.5],
  "missing_evidence": "Need to see what happens after the person enters.",
  "confidence": 0.6,
  "action": {"name": "inspect", "start": 10, "end": 20, "k": 4}
}
```

To answer, use `"action": {"name": "answer"}`. The option and evidence are shared top-level
fields, so there cannot be two conflicting answers in the same response. An inspection also
keeps the current candidate answer for later logging. This JSON layout is an implementation
choice; freeze it with the prompt before evaluation.

`parse_decision(raw, duration=..., observed_timestamps=...)` returns a `Decision` with a
candidate `Answer` and an optional `Inspect` request. It raises `DecisionError` for bad input.
It does not change controller state, execute a tool, repair JSON or call a model.

All five top-level fields are required. Unknown or repeated keys are rejected, including
inside the action. Numeric strings, booleans used as numbers, NaN and infinity are rejected.
Confidence must be in [0, 1]. Evidence must contain unique observed timestamps; an empty
list is allowed so a forced answer does not have to invent evidence. Inspection intervals
must stay within the video, with start < end and integer k equal to 4 or 8.

Responses over 16,384 characters are rejected before parsing. Extra prose, truncated JSON
and excessively nested input fail rather than triggering an automatic repair. This limit is
an engineering default, not a token budget from the proposal. The model generation limit
still needs to be fixed when the backend is connected.

Passing the parser does not authorize an action. The controller still checks remaining
frames, rounds, repeated sample timestamps and the 0.80 early-stop rule. The reader must
also check actual frame IDs before committing an observation. ObservationSession now handles
this step and supplies decoded times to its controller. Use its snapshot for call validation;
the standalone controller still uses requested timestamps.

The [call wrapper](model_calls.md) records raw responses and rejection reasons and counts
backend attempts. It stops on the first error without a retry, including at a forced final
round. Its tests use scripted responses. The real model and full video runner are still pending.
