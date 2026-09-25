# Video-MME screening and manifest

Obtain data through the official benchmark distribution and comply with its terms. Do not
commit videos or copied benchmark content. Reference stable question/video IDs and source
revision in manifests; keep local paths in ignored data/.

Planned reviewer columns: `question_id, video_id, duration_group, reviewer_id,
silent_visual_answerable, brief_event_dependent, evidence_start_s, evidence_end_s,
exclusion_reason`. Review independently before reconciliation. Report separate Cohen's kappa
for answerability and brief-event labels; undefined kappa remains undefined with counts.

Freeze reconciled IDs, strata, inclusion status, rationale and evidence intervals before
method comparison. Record file SHA-256 and dataset revision. Keep gold answers out of
model prompts and selection policies. Identify development/pilot questions before tuning;
never tune against scored outcomes. The proposal leaves pilot overlap unspecified: decide
and document it before freezing.

If fewer than 100 eligible questions exist, report achieved counts/reasons without relaxing
criteria based on model performance. No sample IDs or reviewer scores are fabricated here.
