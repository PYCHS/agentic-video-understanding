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
