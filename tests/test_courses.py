import os
import pytest
import torch
from q1_4v4.course_config import Curriculum, COURSES, LOW_LEVEL, FULL_MATCHES, command, make_curriculum


def test_explicit_direct_match_bypasses_parallel_scheduler_loading_without_claiming_mastery():
    from q1_4v4.parallel_curriculum import ParallelCurriculum
    previous=ParallelCurriculum();previous.mastered.update(COURSES[:3])
    checkpoint={'iteration':1200,'curriculum':previous.state_dict()}
    selected=make_curriculum('serial','balance',checkpoint,'27')
    assert selected.stage=='match' and selected.stage_updates==0 and len(selected.history)==0
    assert previous.mastered==set(COURSES[:3])
    assert make_curriculum('serial','balance',checkpoint,'match').stage=='match'
    with pytest.raises(ValueError,match='requires --start-course'):make_curriculum('serial','balance',checkpoint)
    with pytest.raises(ValueError,match='only for serial'):make_curriculum('parallel','balance',checkpoint,'27')
    with pytest.raises(ValueError,match='Unknown start course'):make_curriculum('serial','balance',checkpoint,'28')
    with pytest.raises(ValueError,match='requires --checkpoint'):make_curriculum('serial','balance',None,'27')


def test_disjoint_consecutive_gates_and_legacy_migration():
    c=Curriculum()
    c.load_state_dict({'index':3,'stage_updates':200,'history':[]})
    assert c.stage=='balance' and c.stage_updates==0
    for iteration in range(64):
        for j in range(32): c.record(dict(stage='balance',stable=True,success=True,duration=6,team=j%2))
        advanced=c.advance()
        if iteration<63: assert not advanced
    assert advanced and c.stage=='balance_push'
    d=Curriculum();d.load_state_dict(c.state_dict());assert d.state_dict()==c.state_dict()
    for _ in range(100):
        for _ in range(128): d.record(dict(stage=d.stage,stable=True,success=False,duration=6))
        assert not d.advance()


def test_task_commands_preserve_identity():
    assert len({tuple(command(s)) for s in COURSES})==len(COURSES)
    assert command('balance')==[1,0,0,0,0,0]
    assert all(len(command(s))==6 for s in COURSES)


def test_team_windows_survive_large_ordered_bursts_and_restart():
    c=Curriculum('step_in_place')
    for team in (0,1):
        for _ in range(10000):
            c.record(dict(stage=c.stage,team=team,stable=True,success=True,duration=12))
    m=c.metrics()
    assert m['red_samples']==m['blue_samples']==1024
    assert m['red_stability']==m['blue_stability']==1
    assert m['red_gate_samples']==m['blue_gate_samples']==1024
    restored=Curriculum();restored.load_state_dict(c.state_dict())
    assert restored.state_dict()==c.state_dict()
    restored.stage_updates=31
    assert not restored.advance() and restored.passed_windows==1
    assert len(restored.history)==0
    # A new window needs fresh observations from BOTH teams.
    for _ in range(10000):
        restored.record(dict(stage=restored.stage,team=1,stable=True,success=True))
    restored.stage_updates=47
    assert not restored.advance() and restored.passed_windows==0
    assert restored.last_reason=='samples'


def test_missing_team_is_unknown_and_bad_team_cannot_hide_in_average():
    c=Curriculum()
    for _ in range(256):c.record(dict(stage=c.stage,team=0,stable=False,success=True))
    assert c.metrics()['blue_stability'] is None and c.metrics()['blue_samples']==0
    for _ in range(10000):c.record(dict(stage=c.stage,team=1,stable=True,success=True))
    c.stage_updates=31
    assert not c.advance() and c.last_reason=='stability'
    stage=c.stage
    c.recertify('reward_change_recertification')
    assert c.stage==stage and c.stage_updates==0 and len(c.monitor)==0


def test_final_match_never_reports_a_stability_only_pass():
    c=Curriculum('match')
    for _ in range(64):
        for team in (0,1):
            for _ in range(32):c.record(dict(stage='match',team=team,stable=True,success=False))
        assert not c.advance()
    assert c.stage=='match' and c.last_reason=='final_training' and c.passed_windows==0


@pytest.fixture(scope='module')
def env():
    if os.environ.get('Q1_TEST_WARP')!='1': pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.course_env import CourseWarpEnv
    return CourseWarpEnv(count=4)


@pytest.mark.parametrize('course',COURSES)
def test_each_course_spawn_and_physics(env,course):
    env.configure(course);env.episode_steps=300;env.reset()
    for _ in range(3):
        obs,state,reward,done,trunc,info=env.step(torch.zeros((4,8,22),device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(state).all() and torch.isfinite(reward).all()
    env.check()
    assert obs.shape==(4,8,128) and state.shape==(4,8,1028)
    assert done.shape==((4,8) if course in LOW_LEVEL|FULL_MATCHES else (4,))
    assert not info['success'].any()


def test_independent_robot_reset(env):
    env.configure('move');env.reset()
    env.step(torch.zeros((4,8,22),device=env.device))
    q,v,steps=env.qpos.clone(),env.qvel.clone(),env.agent_steps.clone()
    env.reset_agents(torch.tensor([[0,0]],device=env.device))
    # The seven other articulated bodies and shared ball are untouched.
    allowed=torch.ones(env.model.nq,device=env.device,dtype=torch.bool)
    allowed[env.qidx[0]]=False;allowed[env.rootq[0]:env.rootq[0]+7]=False
    assert torch.equal(q[0,allowed],env.qpos[0,allowed])
    assert torch.equal(q[1:],env.qpos[1:]) and torch.equal(v[1:],env.qvel[1:])
    assert env.agent_steps[0,0]==0 and torch.equal(steps[0,1:],env.agent_steps[0,1:])


@pytest.mark.parametrize('side',['left','right'])
def test_both_foot_contacts_and_body_exclusion(env,side):
    env.configure('match');env.reset()
    f=0 if side=='left' else 1
    foot=env.feet[2,f]
    env.qpos[0,env.ballq:env.ballq+3]=env.xpos[0,foot]+env.tensor([.12,0,.05])
    # A ball at torso height may touch the body, but must never count as a foot contact.
    torso=env.model.body('red2_torso_link').id
    env.qpos[1,env.ballq:env.ballq+3]=env.xpos[1,torso]+env.tensor([.05,0,.05])
    env.forward()
    env.step(torch.zeros((4,8,22),device=env.device));env.check()
    assert env.foot_hits[0,2,f]==1
    assert not env.foot_hits[1].any()
    assert env.forward_kick[1].sum()==0 and env.side_kick[1].sum()==0


@pytest.mark.parametrize('course',COURSES[13:])
def test_single_ball_participants_and_task_relative_random_spawns(env,course):
    env.configure(course);env.reset()
    expected=2 if course in ('pass_receive','duel') else 3 if course=='attack_2v1' else 4 if course=='game_2v2' else 8 if course in ('teamplay','match') else 1
    assert (env.active.sum(-1)==expected).all()
    positions=env.qpos[:,env.rootq[:,None]+torch.arange(3,device=env.device)]
    assert (positions[~env.active][:,1]>6).all()
    assert (positions[env.active][:,:2].abs()<env.tensor([7.7,4.7])).all()
    first=env.ball.clone();first_robot=positions[env.rows,env.practice].clone()
    if course in ('duel','attack_2v1','game_2v2','teamplay','match'):
        assert (first[:,:2].abs()<=.2).all()
    elif course!='goalkeeper':
        assert (first[:,:2].abs()<=env.tensor([3.5,2.3])).all()
        distance=torch.linalg.vector_norm(first_robot[:,:2]-first[:,:2],dim=-1)
        assert (distance>.28).all() and (distance<1.4).all()
    if course=='pass_receive':
        receiver=(env.practice//4)*4+(env.practice%4+1)%4
        assert (receiver//4==env.practice//4).all()
        assert env.active[env.rows,receiver].all()
    env.reset()
    assert not torch.equal(first,env.ball)
    positions=env.qpos[:,env.rootq[:,None]+torch.arange(3,device=env.device)]
    assert not torch.equal(first_robot,positions[env.rows,env.practice])
    _,_,reward,_,_,_=env.step(torch.zeros((4,8,22),device=env.device))
    assert not reward.squeeze(-1)[~env.active].any()
    env.check()


def test_inactive_falls_do_not_end_a_solo_drill(env):
    env.configure('kick_left');env.reset()
    inactive=(env.practice+1)%8
    env.qpos[env.rows,env.rootq[inactive]+2]=.1
    env.forward()
    _,_,_,done,_,_=env.step(torch.zeros((4,8,22),device=env.device))
    assert not done.any()


def test_contested_goals_have_separate_winner_and_loser_evidence(env):
    from q1_4v4.mixed_env import ended_rows
    env.configure('duel');env.reset();env.episode_steps=1000;env.steps[:]=300
    env.ever_feet[:,2,0]=True;env.ever_feet[:,6,0]=True
    env.qpos[:2,env.ballq:env.ballq+3]=env.tensor([[7.05,0,.11],[-7.05,0,.11]])
    env.qvel[:2,env.ballv]=env.tensor([10.,-10.]);env.forward()
    _,_,_,term,trunc,info=env.step(torch.zeros((4,8,22),device=env.device))
    assert info['team_success'][:2].tolist()==[[True,False],[False,True]]
    rows,_=ended_rows(env,term,trunc,info)
    assert rows.shape==(4,10) and rows[:,8].tolist()==[0,1,0,1]
    assert rows[:,1].tolist()==[1,0,0,1]


@pytest.mark.parametrize('course',['kick_left','kick_right','side_left','side_right'])
def test_designated_foot_spawn_is_correct_in_both_teams(env,course):
    env.configure(course)
    seen=set()
    for _ in range(4):
        env.reset();p=env.practice;seen.update((p//4).tolist())
        position=env.qpos[env.rows[:,None],env.rootq[p,None]+torch.arange(2,device=env.device)]
        q=env.qpos[env.rows[:,None],env.rootq[p,None]+torch.arange(3,7,device=env.device)]
        yaw=2*torch.atan2(q[:,3],q[:,0])
        delta=env.ball[:,:2]-position
        local_x=delta[:,0]*torch.cos(yaw)+delta[:,1]*torch.sin(yaw)
        local_y=-delta[:,0]*torch.sin(yaw)+delta[:,1]*torch.cos(yaw)
        assert ((local_y>0) if course.endswith('left') else (local_y<0)).all()
        if course.startswith('side_'):
            assert (local_y.abs()>.25).all() and (local_x.abs()<.16).all()
        else:assert (local_x>.25).all()
    assert seen=={0,1}
