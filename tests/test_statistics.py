import pytest
from q1_supermimic.statistics import wilson_interval, paired_difference


def test_no_falls_is_not_proof_of_zero_risk():
    low,high=wilson_interval(0,12)
    assert low==pytest.approx(0,abs=1e-12) and .24<high<.25
    assert wilson_interval(0,128)[1]<.03
    assert wilson_interval(12,12)[0]>.75


def test_paired_interval_respects_identical_initial_state_pairs():
    result=paired_difference([1,20,-3,50],[1.2,20.2,-2.8,50.2])
    assert result['mean_difference']==pytest.approx(.2)
    assert result['bootstrap_95_interval']==pytest.approx([.2,.2])
    with pytest.raises(ValueError): paired_difference([1],[2])
