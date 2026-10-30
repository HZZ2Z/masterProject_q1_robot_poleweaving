import numpy as np
from train import retain_candidate


def test_real_selection_rejects_falls_and_return_regression():
    baseline = dict(fall_rate=0., tracking_return=.97)
    assert retain_candidate(dict(fall_rate=0., tracking_return=.98), baseline, 1)
    assert not retain_candidate(dict(fall_rate=.1, tracking_return=.99), baseline, 1)
    assert not retain_candidate(dict(fall_rate=0., tracking_return=.96), baseline, 1)
    assert not retain_candidate(dict(fall_rate=0., tracking_return=np.nan), baseline, 1)
    assert not retain_candidate(baseline, baseline, 0)
