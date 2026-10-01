# Connecting decoded frames to the controller

ObservationSession owns the frames and controller for one question. It reads the 8-frame
survey before the session is ready. A clip with too few distinct frames fails rather than
filling the survey with repeats.

For an inspection, it checks a controller copy, reads the complete batch, and checks frame
IDs. Only then does it update the real frames, round and budget. Decode errors, repeated
frames and incomplete batches leave state unchanged. The runner must still stop after a
failed action under the zero-retry policy; rollback is not permission for another model call.

Evidence uses actual decoded timestamps. Requested timestamps remain in SelectedFrame for
logging. The standalone Controller uses requested times; the session replaces these with
actual times after every successful batch. Frame IDs provide the final duplicate check.

Pass `session.controller` to DecisionCaller, then pass the decision to `session.apply`.
An inspection returns the newly read frames; an answer returns the answer dict. The controller
property returns a detached snapshot. Do not share sessions across questions or concurrent
workers, and do not modify returned images.

This is not the complete model loop. Prompts, Qwen integration and persistent observation
traces remain pending. Tests cover rollback, collisions, actual-time evidence and the cap,
plus a real lossless-video survey/answer check. No QA accuracy is measured.
