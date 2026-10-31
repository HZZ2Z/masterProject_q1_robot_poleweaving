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


def test_parallel_finish_resets_only_ended_robots_and_leaves_other_course_untouched(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from q1_4v4.mixed_env import MixedCourseEnv
    # Exercise the production dispatcher on CPU; only CUDA stream calls are stubbed.
    monkeypatch.setattr(torch.cuda,'set_stream',lambda stream:None)
    stream=SimpleNamespace(wait_stream=lambda other:None)
    groups={}
    pending=[]
    observations=[]
    for name,offset in [('step_in_place',0),('slow_walk',2)]:
        obs=torch.arange(offset*8,(offset+2)*8,dtype=torch.float32).reshape(2,8,1)+1
        term=torch.zeros((2,8),dtype=torch.bool)
        trunc=torch.zeros_like(term)
        if name=='step_in_place':
            term[1,6]=True
            trunc[0,2]=True
        info={key:torch.zeros((2,8),dtype=torch.bool) for key in ('stable','success','touch')}
        info.update(fallen=term,duration=torch.full((2,8),4.))
        group=SimpleNamespace(course=name,device='cpu',torch_stream=stream,
            active=torch.ones((2,8),dtype=torch.bool),
            reset=Mock(side_effect=AssertionError('Must not reset whole worlds')))
        def reset_agents(pairs,obs=obs):
            obs[pairs[:,0],pairs[:,1]]=0
            return obs,None
        group.reset_agents=Mock(side_effect=reset_agents)
        sl=slice(offset,offset+2)
        groups[name]=(sl,group)
        pending.append((sl,group,term,trunc,info))
        observations.append(obs)
    mixed=MixedCourseEnv.__new__(MixedCourseEnv)
    mixed.device='cpu';mixed.groups=groups;mixed.torch_stream=stream
    mixed._obs=torch.cat(observations);mixed._pending=pending
    before=mixed._obs.clone()
    next_obs,records=mixed.finish_step()
    expected=before.clone();expected[0,2]=0;expected[1,6]=0
    torch.testing.assert_close(next_obs,expected)
    assert records.shape==(2,10) and records[:,-1].tolist()==[3.,3.]
    assert records[:,-2].tolist()==[0.,1.]  # Endings retain their own team identity.
    stepping=groups['step_in_place'][1]
    assert stepping.reset_agents.call_args.args[0].tolist()==[[0,2],[1,6]]
    stepping.reset.assert_not_called()
    walking=groups['slow_walk'][1]
    walking.reset_agents.assert_not_called();walking.reset.assert_not_called()
    assert not mixed._pending


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
    assert len(records)==52  # 48 low-level samples + 2 rallies x 2 team outcomes.
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
    assert len(rows)==13*8+10+5*2
    assert set(rows[:,-1].cpu().tolist())==set(float(i) for i in range(28))
    assert torch.isfinite(reward).all()
    e.check()
    e.configure({'balance':4,'balance_push':10,'weight_shift':10,'step_in_place':4})
    assert sum(e.allocations.values())==28 and e.course_ids.shape==(28,)
    e.step(torch.zeros((28,8,22),device='cuda:0'));e.finish_step();e.check()


def test_per_course_reward_is_raw_and_rollout_accumulators_reset(env):
    from q1_4v4.telemetry import active_team_reward,SKILL_NAMES
    from q1_4v4.gait import METRIC_NAMES
    env.reset();env.begin_rollout()
    _,_,reward,_,_,_=env.step(torch.zeros((8,8,22),device='cuda:0'))
    expected=torch.zeros_like(env.course_reward_sum)
    for name,(sl,group) in env.groups.items():
        expected[COURSES.index(name)]=active_team_reward(reward[sl],group.active)
    torch.testing.assert_close(env.course_reward_sum,expected)
    assert env.course_skill_sum.shape==(28,len(SKILL_NAMES))
    assert env.course_stat_sum.shape==(28,len(METRIC_NAMES))
    reward.add_(123.)
    torch.testing.assert_close(env.course_reward_sum,expected)
    env.finish_step();env.begin_rollout()
    for value in (env.course_reward_sum,env.course_skill_sum,env.course_stat_sum):
        assert torch.count_nonzero(value)==0


def test_reallocation_releases_physical_memory_before_building_new_scenes(monkeypatch):
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    import gc,weakref
    import q1_4v4.mixed_env as module
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    cold_free=torch.cuda.mem_get_info()[0]
    e=module.MixedCourseEnv(256,'cuda:0',71,
        {'balance':13,'balance_push':13,'weight_shift':204,'step_in_place':26})
    action=torch.zeros((256,8,22),device='cuda:0')
    e.step(action);e.finish_step();torch.cuda.synchronize()
    old_free=torch.cuda.mem_get_info()[0]
    refs=[weakref.ref(group) for _,group in e.groups.values()]
    original=module.CourseWarpEnv
    measured=[]
    def construct(*args,**kwargs):
        if not measured:
            assert not any(r() is not None for r in refs)
            free=torch.cuda.mem_get_info()[0];measured.append(free)
            # Includes Warp-owned buffers, which Torch's allocator counters omit.
            assert free-old_free >= .75*(cold_free-old_free)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'CourseWarpEnv',construct)
    e.configure({'balance':9,'balance_push':9,'weight_shift':9,'step_in_place':203,'slow_walk':13,'turn':13})
    e.step(action);e.finish_step();e.check()
    assert len(measured)==1 and sum(e.allocations.values())==256
