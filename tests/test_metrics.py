import torch
from q1_supermimic.metrics import EpisodeMetrics


def test_episode_metrics_match_completed_episode_units_and_do_not_publish_empty_rates():
    metrics = EpisodeMetrics()
    metrics.update({'steps': torch.ones(2), 'reward': torch.tensor([2., 3.]),
                    'success_rate': torch.zeros(2), 'fall_rate': torch.zeros(2),
                    'survival': torch.tensor([.1, .1])}, torch.zeros(2, dtype=torch.bool))
    assert metrics.drain(.02) == {}
    metrics.update({'steps': torch.ones(2), 'reward': torch.tensor([4., 5.]),
                    'success_rate': torch.tensor([1., 0.]), 'fall_rate': torch.zeros(2),
                    'survival': torch.tensor([.1, .1])}, torch.tensor([True, False]))
    result = metrics.drain(.02)
    assert result['reward'] == 6 and result['success_rate'] == 1
    assert result['episode/duration_s'] == .04
    assert abs(result['episode/survival'] - .2) < 1e-6
    assert metrics.drain(.02) == {}


def test_window_weights_episodes_and_evicts_old_results():
    metrics = EpisodeMetrics()
    def finish(n, falls):
        metrics.update({'steps': torch.ones(n), 'reward': torch.zeros(n),
                        'success_rate': torch.zeros(n), 'fall_rate': torch.full((n,), falls)},
                       torch.ones(n, dtype=torch.bool))
        return metrics.drain(.02)
    # Constant environment count, differing numbers of completed episodes.
    metrics.update({'steps': torch.ones(1000), 'reward': torch.zeros(1000),
                    'success_rate': torch.zeros(1000), 'fall_rate': torch.zeros(1000)},
                   torch.ones(1000, dtype=torch.bool))
    metrics.drain(.02)
    mask = torch.zeros(1000, dtype=torch.bool); mask[0] = True
    metrics.update({'steps': torch.ones(1000), 'reward': torch.zeros(1000),
                    'success_rate': torch.zeros(1000), 'fall_rate': mask.float()}, mask)
    result = metrics.drain(.02)
    assert result['instant/fall_rate'] == 1
    assert result['fall_rate'] == .001
    assert result['statistics/window_episodes'] == 1000
    assert result['statistics/window_ready'] == 1
    assert metrics.drain(.02) == {}
