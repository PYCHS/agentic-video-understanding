"""Compose saved-result validation, pairing and descriptive accuracy comparisons."""
from dataclasses import asdict

from .pair_predictions import paired_correctness
from .paired_stats import summarize_paired_results, paired_video_bootstrap_ci
from .result_predictions import predictions_from_records


def compare_records(manifest, baseline_records, adaptive_records, *,
                    baseline_run_id, adaptive_run_id, baseline_method="Uniform-24",
                    adaptive_method="Adaptive-24", bootstrap_samples=10_000, seed=128):
    """Return a JSON-compatible report; no files or model calls are made.

    Run identity fields are claims, not verified checkpoint/config provenance.
    Effects and intervals are proportions, not percentage points.
    """
    manifest = list(manifest)
    baseline = predictions_from_records(baseline_records, run_id=baseline_run_id,
                                        method=baseline_method)
    adaptive = predictions_from_records(adaptive_records, run_id=adaptive_run_id,
                                        method=adaptive_method)
    ids, base_correct, adapt_correct = paired_correctness(
        manifest, baseline, adaptive, require_video_ids=True)
    video_ids = [row["video_id"] for row in manifest]
    paired = summarize_paired_results(base_correct, adapt_correct,
                                     bootstrap_samples=bootstrap_samples, seed=seed)
    cluster = None
    reason = "fewer than two distinct videos"
    if len(set(video_ids)) >= 2:
        cluster = asdict(paired_video_bootstrap_ci(base_correct, adapt_correct, video_ids,
                                                 samples=bootstrap_samples, seed=seed))
        reason = None

    def method_summary(rows, correct, run_id, method):
        return dict(run_id=run_id, method=method, correct=sum(correct),
                    failed=sum(row["status"] == "failed" for row in rows),
                    accuracy=sum(correct)/len(ids))

    return dict(schema_version=1, provenance_verified=False,
                question_ids=ids, question_count=len(ids), video_count=len(set(video_ids)),
                effect_direction="adaptive_minus_baseline", accuracy_units="proportion",
                bootstrap=dict(samples=bootstrap_samples, seed=seed, confidence=.95),
                baseline=method_summary(baseline, base_correct, baseline_run_id, baseline_method),
                adaptive=method_summary(adaptive, adapt_correct, adaptive_run_id, adaptive_method),
                paired=asdict(paired), video_cluster=cluster, video_cluster_unavailable_reason=reason)
