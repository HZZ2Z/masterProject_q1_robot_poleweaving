"""Uncertainty for independent episode samples, not correlated frame counts."""
import math
import numpy as np


def wilson_interval(events, trials, z=1.959963984540054):
    if trials < 1 or not 0 <= events <= trials:
        raise ValueError('Invalid event counts')
    rate = events / trials
    denominator = 1 + z*z/trials
    center = (rate + z*z/(2*trials)) / denominator
    half = z * math.sqrt(rate*(1-rate)/trials + z*z/(4*trials*trials)) / denominator
    return [max(0.,center-half), min(1.,center+half)]


def paired_difference(baseline, candidate, seed=7163):
    a, b = np.asarray(baseline), np.asarray(candidate)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Need matching finite episode vectors, at least two pairs')
    difference = b-a
    random = np.random.default_rng(seed)
    means = difference[random.integers(len(a), size=(10000,len(a)))].mean(1)
    return dict(pairs=len(a), mean_difference=float(difference.mean()),
                bootstrap_95_interval=np.quantile(means,[.025,.975]).tolist())
