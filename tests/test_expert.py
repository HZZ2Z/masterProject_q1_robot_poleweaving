"""Teacher isolation, causal histories, genuine terminations and dataset labels."""

import mujoco
import numpy as np
import pytest
import torch

from expert.runtime import ROOT,SoccerPlayer
from expert.collect import collect_episode,episode_seed,split_for
from expert.dataset import observation_history,outcome,validate_arrays


@pytest.fixture
def player():
    torch.set_num_threads(2)
    return SoccerPlayer(seed=42,reset_mode='default')


def test_teacher_imports_are_package_relative_and_assets_are_local():
    for file in ['runtime.py','motion.py','kinematics.py']:
        text=(ROOT/file).read_text()
        assert 'from utils.' not in text and 'from envs.' not in text
    # Inference never requires its original neighboring workspace.
    assert (ROOT/'configs/Q1.yaml').exists()
    assert (ROOT/'resources/q1_description/q1_soccer.xml').exists()


def test_history_reconstruction_is_causal_and_newest_first():
    observations=np.arange(6*3,dtype=np.float32).reshape(6,3)
    initial=np.array([observations[0],[-1,-2,-3],[-4,-5,-6],[-7,-8,-9]],dtype=np.float32)
    np.testing.assert_array_equal(observation_history(observations,0,initial),initial)
    np.testing.assert_array_equal(observation_history(observations,1,initial),np.concatenate((observations[1:2],initial[:3])))
    np.testing.assert_array_equal(observation_history(observations,5,initial),observations[2:6][::-1])


def test_timeout_failure_and_collector_cutoff_never_become_imitation_success():
    contact=np.ones((100,10),dtype=bool)
    for cause in ['timeout','fall','out','collector_limit','interrupted','invalid']:
        result=outcome(cause,contact,1.,2.,.2)
        assert not result['task_success'] and not result['policy_imitation_eligible']
        assert result['timeout']==(cause=='timeout')
        assert result['terminated']==(cause in {'fall','out'})
        assert result['truncated']==(cause not in {'fall','out'})
    success=outcome('goal',contact,1.,2.,.2)
    assert success['verified_success'] and success['policy_imitation_eligible']
    assert not outcome('goal',np.zeros_like(contact),1.,2.,None)['verified_success']
    assert not outcome('goal',contact,.01,2.,.2)['verified_success']
    assert not outcome('goal',contact,1.,.5,.2)['verified_success']


def test_collection_cutoff_is_not_environment_timeout(player):
    arrays,meta=collect_episode(player,max_control_steps=4)
    assert meta['termination']=='collector_limit' and not meta['timeout']
    assert not meta['task_success'] and not meta['policy_imitation_eligible']
    assert validate_arrays(arrays,meta)==4
    for t in range(4):
        history=observation_history(arrays['observations'],t,arrays['initial_history'])
        with torch.inference_mode():
            action=player.policy(torch.tensor(arrays['observations'][t:t+1]),torch.tensor(history[None]))
        np.testing.assert_array_equal(action.numpy()[0],arrays['raw_actions'][t])
    assert arrays['world_model_valid'].all() and not arrays['policy_imitation_valid'].any()


def test_environment_timeout_is_truncation_and_physics_has_no_reset(player):
    player.cfg['rewards']['episode_length_s']=.04
    arrays,meta=collect_episode(player)
    assert meta['termination']=='timeout' and meta['timeout']
    assert meta['steps']==3 and meta['truncated'] and not meta['terminated']
    assert arrays['time'][-1]==pytest.approx(.06)
    assert not arrays['terminated'].any() and arrays['truncated'][-1]


def test_original_fall_and_out_labels_are_preserved(player):
    player.data.qpos[player.root_addr+2]=.05
    mujoco.mj_forward(player.model,player.data)
    arrays,meta=collect_episode(player,max_control_steps=10)
    assert meta['termination']=='fall' and not meta['timeout']
    assert meta['terminated'] and not meta['policy_imitation_eligible']
    player.reset_robot();player.data.qpos[player.ball_addr:player.ball_addr+3]=[0,5,.11]
    mujoco.mj_forward(player.model,player.data)
    arrays,meta=collect_episode(player,max_control_steps=10)
    assert meta['termination']=='out' and meta['terminated']
    assert not meta['task_success']


def test_seed_cohorts_and_splits_are_deterministic():
    cursor=100000;seeds=[]
    for group in [0,1,2,2]*2:
        seed=episode_seed(cursor,group);cursor=seed+1;seeds.append(seed)
        assert int(np.random.default_rng(seed).choice(3,p=[.25,.25,.5]))==group
        assert split_for(seed,'abc')==split_for(seed,'abc')
    assert len(set(seeds))==len(seeds)
