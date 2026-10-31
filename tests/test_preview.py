"""Preview reset routing can be checked without creating a CUDA simulation."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from q1_4v4.preview import CoursePreview


def preview_with_result(course, terminated, truncated):
    obs=torch.arange(8*128,dtype=torch.float32).reshape(1,8,128)
    state=torch.arange(1028,dtype=torch.float32).reshape(1,1028)
    reward=torch.ones((1,8,1))
    info={}
    if course in ('teamplay','match'):
        info=dict(world_terminated=terminated if terminated.ndim==1 else terminated.all(-1),
                  world_truncated=truncated if truncated.ndim==1 else truncated.all(-1),
                  robot_resets=terminated if terminated.ndim==2 else torch.zeros((1,8),dtype=torch.bool))
    env=SimpleNamespace(course=course,device='cpu',
        step=Mock(return_value=(obs,state,reward,terminated,truncated,info)),
        reset=Mock(side_effect=AssertionError('Unexpected whole-world reset')),
        reset_agents=Mock(return_value=(obs+100,state+100)),check=Mock())
    preview=CoursePreview.__new__(CoursePreview)
    preview.env=env
    preview.sync=Mock()
    return preview


@pytest.mark.parametrize('course,fallen,timed_out',[
    ('balance',[1],[]),
    ('step_in_place',[6],[]),
    ('step_in_place',[1,6],[3]),
    ('start_stop',[],[0,7]),
])
def test_robot_endings_do_not_end_preview_world(course,fallen,timed_out):
    terminated=torch.zeros((1,8),dtype=torch.bool)
    truncated=torch.zeros_like(terminated)
    terminated[0,fallen]=True
    truncated[0,timed_out]=True
    preview=preview_with_result(course,terminated,truncated)
    obs,state,reward,done,timeout,info=preview.step(np.zeros((8,22)))
    # play.py must not take its global-reset branch for a robot ending.
    assert done is False and timeout is False
    assert info['reset_agents']==sorted(fallen+timed_out)
    assert info['terminated_agents']==fallen
    assert info['truncated_agents']==timed_out
    preview.env.reset_agents.assert_called_once()
    pairs=preview.env.reset_agents.call_args.args[0]
    assert pairs.tolist()==[[0,a] for a in sorted(fallen+timed_out)]
    # The next actor input must contain reset state, not stale terminal state.
    next_obs,next_state=preview.env.reset_agents.return_value
    np.testing.assert_array_equal(obs,next_obs[0].numpy())
    np.testing.assert_array_equal(state,next_state[0].numpy())
    np.testing.assert_array_equal(reward,np.ones((8,1)))
    preview.env.reset.assert_not_called()
    preview.env.check.assert_called_once()
    preview.sync.assert_called_once()


def test_unfinished_robots_keep_running_without_resets():
    mask=torch.zeros((1,8),dtype=torch.bool)
    preview=preview_with_result('step_in_place',mask,mask)
    obs,state,_,done,timeout,info=preview.step(np.zeros((8,22)))
    assert not done and not timeout and 'reset_agents' not in info
    preview.env.reset_agents.assert_not_called()
    preview.env.reset.assert_not_called()
    np.testing.assert_array_equal(obs,preview.env.step.return_value[0][0].numpy())


def test_match_fall_resets_only_the_fallen_robot_in_gui():
    fallen=torch.zeros((1,8),dtype=torch.bool);fallen[0,6]=True
    preview=preview_with_result('match',fallen,torch.zeros_like(fallen))
    *_,done,timeout,info=preview.step(np.zeros((8,22)))
    assert not done and not timeout and info['reset_agents']==[6]
    assert preview.env.reset_agents.call_args.args[0].tolist()==[[0,6]]
    preview.env.reset.assert_not_called()


@pytest.mark.parametrize('course,done,timeout',[
    ('approach',True,False),('match',False,True),
])
def test_soccer_rally_endings_still_reach_play_world_reset(course,done,timeout):
    preview=preview_with_result(course,torch.tensor([done]),torch.tensor([timeout]))
    *_,actual_done,actual_timeout,info=preview.step(np.zeros((8,22)))
    assert actual_done is done and actual_timeout is timeout
    assert 'reset_agents' not in info
    preview.env.reset_agents.assert_not_called()
