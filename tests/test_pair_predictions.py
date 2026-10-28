import pytest

from evaluation.pair_predictions import paired_correctness
from evaluation.paired_stats import summarize_paired_results

GOLD = [{"question_id":"q2","answer":"B"},{"question_id":"q1","answer":"A"}]
BASE = [{"question_id":"q1","status":"answered","option":"A"},
        {"question_id":"q2","status":"failed","option":None}]
ADAPT = [{"question_id":"q1","status":"answered","option":"C"},
         {"question_id":"q2","status":"answered","option":"B"}]


def test_aligns_by_id_instead_of_row_position_and_preserves_failures():
    ids,base,adaptive = paired_correctness(GOLD,BASE,ADAPT)
    assert ids == ["q2","q1"]
    assert base == [False,True]
    assert adaptive == [True,False]
    result = summarize_paired_results(base,adaptive,bootstrap_samples=100,seed=128)
    assert result.accuracy_delta == 0
    assert result.discordant_a_only == result.discordant_b_only == 1


@pytest.mark.parametrize("rows",[[],BASE[:1],BASE+[BASE[0]],
    BASE+[{"question_id":"extra","status":"answered","option":"A"}]])
def test_rejects_missing_extra_and_duplicate_predictions(rows):
    with pytest.raises(ValueError):
        paired_correctness(GOLD,rows,ADAPT)
    with pytest.raises(ValueError):
        paired_correctness(GOLD,BASE,rows)


@pytest.mark.parametrize("patch",[{"status":"unknown"},{"option":"E"},
                                 {"status":"failed","option":"A"}])
def test_rejects_ambiguous_prediction_rows(patch):
    with pytest.raises(ValueError):
        paired_correctness(GOLD,[BASE[0] | patch,BASE[1]],ADAPT)


@pytest.mark.parametrize("manifest",[[],GOLD+[GOLD[0]],
    [{"question_id":"q1","answer":"E"}], [{"question_id":"","answer":"A"}]])
def test_rejects_invalid_manifest(manifest):
    with pytest.raises(ValueError):
        paired_correctness(manifest,BASE,ADAPT)


def test_video_matching_preserves_shared_videos_and_manifest_order():
    gold = [row | {"video_id":"same-video"} for row in GOLD]
    base = [row | {"video_id":"same-video"} for row in BASE]
    adaptive = [row | {"video_id":"same-video"} for row in ADAPT]
    assert paired_correctness(gold, base, adaptive, require_video_ids=True) == (
        ["q2", "q1"], [False, True], [True, False])


@pytest.mark.parametrize("method", ["baseline", "adaptive"])
@pytest.mark.parametrize("video_id", [None, "wrong-video", ""])
def test_video_mismatch_is_rejected_even_for_failed_prediction(method, video_id):
    gold = [row | {"video_id":"v1"} for row in GOLD]
    base = [row | {"video_id":"v1"} for row in BASE]
    adaptive = [row | {"video_id":"v1"} for row in ADAPT]
    rows = base if method == "baseline" else adaptive
    rows[1]["video_id"] = video_id
    with pytest.raises(ValueError, match="video_id differs"):
        paired_correctness(gold, base, adaptive)


def test_strict_video_matching_cannot_silently_skip_missing_manifest_ids():
    with pytest.raises(ValueError, match="every row"):
        paired_correctness(GOLD, BASE, ADAPT, require_video_ids=True)
    with pytest.raises(ValueError, match="every row"):
        paired_correctness([GOLD[0] | {"video_id":"v1"}, GOLD[1]], BASE, ADAPT)
