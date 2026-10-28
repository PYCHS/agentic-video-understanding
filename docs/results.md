# Saving a question result

After `run_question` returns, call `write_question_result` from
`watch_wisely.results`. Supply the destination path, returned result, and explicit
`run_id`, `question_id`, `video_id`, `method` and `synthetic` keyword arguments.
Use `synthetic=True` for scripted backends and generated test inputs.

The JSON record keeps the answer or failure, raw model responses, token counts,
latencies, frame IDs, requested times and actual times. Missing measurements stay
null. Images are not embedded. Save failed questions too, so scoring cannot
silently exclude them.

The writer prepares a complete temporary file in the destination directory,
flushes and syncs it, then publishes it with a hard link. An existing destination
causes an error; it is never overwritten. Filesystems without hard-link support
fail explicitly. Normal write/publication failures remove the temporary file.
An abrupt process termination can leave a `.question-*.tmp` file; it is not a
completed result. Power-loss durability of the directory entry is not guaranteed.

This saves completed runner output only. It does not checkpoint a model call in
progress, validate claimed identities, or prove that a run used frozen weights,
prompts and manifests. Those provenance checks and per-round recovery are still
pending. The caller must choose a unique path for each run/method/question and
handle disk errors; saving must never trigger another model attempt.

## Preparing predictions for scoring

Load the JSON files and pass the records to
`evaluation.result_predictions.predictions_from_records`, with the expected
`run_id` and `method`. It rejects synthetic records, mixed runs/methods, duplicate
question IDs and inconsistent outcomes. Failures become null-option prediction
rows, keeping their video IDs for later grouping. Pass those rows to
`paired_correctness(..., require_video_ids=True)` with the frozen manifest to check
completeness, match each video ID and score. Include `video_id` in every manifest row.

The synthetic flag is a caller claim, not proof of a real experiment. This adapter
does not validate model revisions, hashes, trace consistency or file-content hashes. Keep the original records; prediction rows are only a scoring view.
