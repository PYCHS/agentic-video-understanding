# Building the image-and-text prompt

`make_message_builder(question, options)` returns the callback used by run_question.
Options must contain exactly A, B, C and D. There is no gold-answer input.

The draft prompt explains inspect/answer JSON, the 24-frame/four-round limits and early
stopping. The current state includes duration, remaining frames/rounds, forced-answer and
always-use-24 flags. Prior accepted responses are included as tentative decision history.
The user payload pairs each selected image with its actual `[t = ... s]` label, sorted by
time. Float labels retain enough digits to round-trip through strict evidence validation.

All previously observed images are included again each round. This does not increase the
unique-frame count, but the future adapter must count repeated visual tokens and processing
time. Image entries contain in-memory reader images, never an entire video or audio track.
Question/options and visible text are treated as task data, not instructions to change rules.

The message dictionaries are an internal format, not a verified Qwen processor integration.
The next adapter must check image handling and chat-template behavior. Prompt version is
`adaptive-v1-draft`; freeze the text/hash and shared baseline answer instructions before
scoring. No model inference or accuracy result is claimed by the prompt tests.

Example wiring, once a real backend exists:

```python
builder = make_message_builder(question, {"A": a, "B": b, "C": c, "D": d})
result = run_question(reader, backend, builder)
```
