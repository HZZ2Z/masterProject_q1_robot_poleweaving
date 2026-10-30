import os
import pytest
import torch
from q1_4v4.parallel_curriculum import ParallelCurriculum,PREREQUISITES
from q1_4v4.course_config import Curriculum,COURSES
from q1_4v4.gpu_learning import normalize_by_course


def test_prerequisite_graph_and_sampling_budget():
    c=ParallelCurriculum();assert c.allocation(16)=={'balance':16}
    legacy=Curriculum('balance_push').state_dict();c.load_state_dict(legacy)
    assert c.mastered=={'balance'} and c.ready==['balance_push','weight_shift']
    assert c.frontier==['step_in_place']
    allocation=c.allocation(4096)
    assert sum(allocation.values())==4096 and set(allocation)=={'balance','balance_push','weight_shift','step_in_place'}
    assert allocation['balance_push']==allocation['weight_shift']
    assert abs(allocation['balance']/4096-.1)<.001
    clone=ParallelCurriculum();clone.load_state_dict(c.state_dict());assert clone.state_dict()==c.state_dict()
    visited=set()
    for name in COURSES:
        assert set(PREREQUISITES[name])<=visited;visited.add(name)
    with pytest.raises(ValueError,match='AFTER course 0'):ParallelCurriculum().load_state_dict(Curriculum().state_dict())


def test_each_course_needs_its_own_team_evidence_and_three_windows():
    c=ParallelCurriculum();c.load_state_dict(Curriculum('balance_push').state_dict())
    # Only weight shift passes; samples from balance_push must not help its gates.
    for update in range(64):
        for team in (0,1):
            for _ in range(32):c.record(dict(stage='weight_shift',team=team,stable=True,success=True))
            c.record(dict(stage='balance_push',team=team,stable=True,success=False))
        c.advance()
    assert c.mastered=={'balance','weight_shift'}
    assert 'step_in_place' in c.ready and 'balance_push' in c.ready
    assert c.courses['step_in_place'].stage_updates==0
    assert c.courses['weight_shift'].stage=='weight_shift'
    assert c.courses['weight_shift'].passed_windows==3


def test_advantage_moments_are_separate_for_each_course_and_ignore_inactive():
    x=torch.tensor([[[[1.],[3.]],[[100.],[300.]]]])
    valid=torch.ones_like(x,dtype=torch.bool);ids=torch.tensor([1,2])
    result=normalize_by_course(x,valid,ids)
    torch.testing.assert_close(result,torch.tensor([[[[-1.],[1.]],[[-1.],[1.]]]]),atol=2e-5,rtol=2e-5)
    valid[0,1,1]=False
    assert normalize_by_course(x,valid,ids)[0,1,0]==0


@pytest.fixture(scope='module')
def env():
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.mixed_env import MixedCourseEnv
    return MixedCourseEnv(8,'cuda:0',42,{'balance':2,'weight_shift':2,'step_in_place':2,'duel':2},episode_steps=4)


def test_mixed_low_level_and_football_terminal_masks_and_evidence(env):
    env.reset()
    action=torch.zeros((8,8,22),device='cuda:0')
    for _ in range(4):
        obs,state,reward,term,trunc,_=env.step(action)
        terminal=state.clone()
        next_obs,records=env.finish_step()
    assert obs.shape==(8,8,128) and state.shape==(8,8,1028)
    assert term.shape==trunc.shape==(8,8)
    assert len(records)==50  # 6 low-level worlds x 8 robots + 2 football rallies.
    assert set(records[:,-1].cpu().tolist())=={0.,2.,3.,23.}
    assert all(torch.isfinite(x).all() for x in (next_obs,reward,terminal))
    assert torch.all(next_obs[:,:,-2]==0)  # Every timeout reset independently.
    assert torch.all(env.active[-2:].sum(-1)==2)
    env.check()


def test_mixed_scene_state_and_rng_are_independent(env):
    first=env.groups['weight_shift'][1];other=env.groups['duel'][1]
    before=other.qpos.clone()
    first.reset_agents(torch.tensor([[0,0]],device='cuda:0'))
    assert torch.equal(before,other.qpos)
    rng=env.course_rng();root=env.generator.get_state()
    expected=env.reset()[0].clone()
    env.generator.set_state(root);env.restore_course_rng(rng)
    actual=env.reset()[0]
    torch.testing.assert_close(actual,expected)
    # Scene groups have distinct streams and explicit producer/consumer dependencies.
    assert len({g.torch_stream.cuda_stream for _,g in env.groups.values()})==len(env.groups)


def test_reallocation_keeps_budget_and_all_28_courses_share_a_batch():
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.mixed_env import MixedCourseEnv
    e=MixedCourseEnv(28,'cuda:0',103,{s:1 for s in COURSES},episode_steps=4)
    e.generator.manual_seed(777);e.restore_course_rng({});e.reset()
    for _ in range(4):
        obs,state,reward,term,trunc,_=e.step(torch.zeros((28,8,22),device='cuda:0'))
        _,rows=e.finish_step()
    assert len(rows)==13*8+15
    assert set(rows[:,-1].cpu().tolist())==set(float(i) for i in range(28))
    assert torch.isfinite(reward).all()
    e.check()
    e.configure({'balance':4,'balance_push':10,'weight_shift':10,'step_in_place':4})
    assert sum(e.allocations.values())==28 and e.course_ids.shape==(28,)
    e.step(torch.zeros((28,8,22),device='cuda:0'));e.finish_step();e.check()


def test_transition_waits_for_real_promotion_not_just_a_high_success_rate():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('balance_transition',Path(__file__).parents[1]/'scripts/continue_after_balance.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checkpoint={'policy_sharing':'shared','curriculum':Curriculum('balance').state_dict()}
    checkpoint['curriculum']['passed_windows']=3
    assert not module.balance_passed(checkpoint)
    checkpoint['curriculum']=Curriculum('balance_push').state_dict()
    assert module.balance_passed(checkpoint)
    checkpoint['curriculum']={'kind':'parallel','mastered':['balance']}
    assert module.balance_passed(checkpoint)
