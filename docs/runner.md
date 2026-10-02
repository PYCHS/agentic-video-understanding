# Running one question

`run_question(reader, backend, build_messages, always_use_24=False)` connects the survey,
call wrapper and observation session. It starts with eight frames, asks for a decision,
applies an inspection or answer, and stops after at most four calls. Any failure ends that
question. There are no repair calls, including after a bad final response or a decode error.

The message builder receives the session and previous call records. It can use all selected
images, actual timestamps, remaining budget and response history. A real multimodal prompt
builder has not been implemented or frozen yet. The builder is trusted application code and
must not mutate the session. The backend still needs a real Qwen adapter.

RunResult contains the answer or failure stage/reason, every attempted model call, each
successful batch's frame IDs and requested/actual times, and wall time. A valid inspection
that fails during decoding remains an accepted *decision* in the call record, but the run
is failed at the apply stage with no new observation committed. It never becomes an answer.
Survey failures make no model calls. Prompt construction failures also stop before generation.

Wall time covers the survey, prompts, model calls and subsequent reads, but excludes reader
construction/indexing performed before run_question. Include indexing separately in future
end-to-end reports. Records currently remain in memory; durable traces, run/question IDs,
environment hashes, timeouts and cancellation still need work.

Tests use a synthetic frame timeline and scripted replies. They cover inspect-to-answer,
decode collisions, always-use-24, malformed terminal responses and early failures. These
check orchestration, not model accuracy or a completed benchmark. The real-video reader and
session are tested separately.
