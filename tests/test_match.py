"""Full-match reward accounting, terminal semantics, and continuous robot respawns."""
import os
import pytest
import torch
from q1_4v4.match_task import MatchTask,MatchControl
from q1_4v4.mixed_env import finish_course_step
from q1_4v4.telemetry import MATCH_TERM_LABELS,match_reward_specs


def test_control_requires_both_teams_and_persists_its_window():
    control=MatchControl()
    good={k:16. for k in ('near_red','near_blue','touch_red','touch_blue',
                          'advance_red','advance_blue','quality_red','quality_blue')}
    for _ in range(23):control.update([.99,.99],[1.,1.],good)
    resumed=MatchControl(control.state_dict())
    for _ in range(25):resumed.update([.99,.99],[1.,1.],good)
    assert resumed.level==1 and resumed.amplitude==.475 and resumed.windows==0
    # Stable standing alone cannot certify even the first touch stage.
    stationary=MatchControl()
    for _ in range(64):stationary.update([1.,1.],[0.,0.])
    assert stationary.level==0 and stationary.amplitude==.45
    resumed.update([.99,.99],[0.,13.]);assert resumed.amplitude==.45
    with pytest.raises(ValueError):MatchControl({'amplitude':.2})


def test_bad_team_and_standard_rehearsal_cannot_supply_near_ball_certification():
    good={k:16. for k in ('near_red','near_blue','touch_red','touch_blue',
                          'advance_red','advance_blue','quality_red','quality_blue')}
    good['touch_blue']=0.
    c=MatchControl()
    for _ in range(64):c.update([1.,1.],[0.,0.],good)
    assert c.level==0 and c.reason=='touch' and c.windows==0
    only_standard=dict(standard_red=1000.,standard_blue=1000.,standard_touch_red=1000.,standard_touch_blue=1000.)
    c=MatchControl()
    for _ in range(64):c.update([1.,1.],[0.,0.],only_standard)
    assert c.level==0 and c.reason=='samples'


def test_crouched_wide_pose_loses_survival_quality_and_has_bounded_costs():
    up=torch.ones(2);z=torch.tensor([.417,.35]);width=torch.tensor([.1754,.59])
    quality,healthy,height_cost,width_cost=MatchTask.posture(up,z,width,torch.zeros(2,dtype=torch.bool))
    assert quality[0]==1 and healthy.tolist()==[True,False]
    assert quality[1]<1e-5 and height_cost[1]>.5 and width_cost[1]>.9
    assert (.2*quality[1]-.6*height_cost[1]-.4*width_cost[1])<0


def test_potential_has_no_discounted_closed_loop_windfall():
    # Terminal zero cancels all intermediate shaping, including a negative excursion.
    path=[0.,.5,-.3,.8,0.];total=0.
    for i,(a,b) in enumerate(zip(path,path[1:])):
        total+=.99**i*MatchTask.potential_reward(torch.tensor(a),torch.tensor(b),torch.tensor(i==3)).item()
    assert total==pytest.approx(0.,abs=1e-6)
    before=torch.tensor([[.4,-.4]])
    shaped=MatchTask.potential_reward(before,before,torch.tensor([[True,False]]))
    torch.testing.assert_close(shaped,torch.tensor([[-.4,.004]]))


def test_roles_have_one_healthy_field_chaser_and_mirror_both_teams():
    sign=torch.tensor([1.]*4+[-1.]*4)
    red=torch.tensor([[-6.,0.,.417],[-3.,0.,.417],[-1.,-1.,.417],[-2.,2.,.417]])
    positions=torch.cat((red,red*torch.tensor([-1.,-1.,1.])))[None]
    ball=torch.zeros((1,3));healthy=torch.ones((1,8),dtype=torch.bool)
    anchors,chase=MatchTask.role_targets(ball,positions,healthy,sign)
    assert chase.tolist()==[[False,False,True,False,False,False,True,False]]
    torch.testing.assert_close(anchors[:,:4],-anchors[:,4:])
    healthy[:,[1,2,3]]=False
    _,chase=MatchTask.role_targets(ball,positions,healthy,sign)
    assert not chase[0,:4].any() and chase[0,4:].sum()==1


def test_reward_dashboard_accounts_for_every_component_once_per_team():
    _,plots=match_reward_specs('match')
    keys=[k for plot in plots for k in plot['keys']]
    assert len(plots)==6 and len(keys)==len(set(keys))==2*len(MatchTask.terms)
    assert len(MATCH_TERM_LABELS)==len(MatchTask.terms)==20


@pytest.fixture(scope='module')
def env():
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.course_env import CourseWarpEnv
    return CourseWarpEnv(count=2,stage='match',episode_steps=1500)


def measure(env,goal=None,out=None):
    """Evaluate controlled physical evidence without taking another physics step."""
    env.forward()
    env.foot_slip=torch.zeros((2,8,2),device=env.device)
    env.gait_bad_contact.zero_();env.gait_contact.fill_(1)
    ball=env.ball.clone();pos=env.xpos[:,env.bodies].clone()
    base=dict(goal=env.tensor(goal or [0.,0.]),out=torch.tensor(out or [False,False],device=env.device))
    return env.match_task.step(torch.zeros((2,8,22),device=env.device),env.previous_action.clone(),ball,pos,base)


def test_fall_preserves_ball_peers_world_clock_and_failure_evidence(env):
    env.reset();env.qpos[0,env.rootq[2]+2]=.1;env.forward()
    _,_,reward,term,trunc,info=env.step(torch.zeros((2,8,22),device=env.device))
    assert term.shape==(2,8) and term[0,2] and term.sum()==1
    assert not info['world_terminated'].any() and not info['world_truncated'].any()
    q,v,clock=env.qpos.clone(),env.qvel.clone(),env.steps.clone()
    obs,rows=finish_course_step(env,term,trunc,info)
    assert obs is not None and len(rows)==0
    keep_q=torch.ones(env.model.nq,device=env.device,dtype=torch.bool)
    keep_q[env.rootq[2]:env.rootq[2]+7]=False;keep_q[env.qidx[2]]=False
    keep_v=torch.ones(env.model.nv,device=env.device,dtype=torch.bool)
    keep_v[env.rootv[2]:env.rootv[2]+6]=False;keep_v[env.vidx[2]]=False
    assert torch.equal(q[0,keep_q],env.qpos[0,keep_q])
    assert torch.equal(v[0,keep_v],env.qvel[0,keep_v])
    assert torch.equal(q[1],env.qpos[1]) and torch.equal(v[1],env.qvel[1])
    assert torch.equal(clock,env.steps) and env.match_task.failures[0,0]==1
    env.steps.fill_(250)
    env.match_task.quality_steps.fill_(250)
    *_,info=measure(env)
    assert info['team_stable'][0].tolist()==[False,True]
    assert info['team_fallen'][0].tolist()==[True,False]
    env.check()


@pytest.mark.parametrize('goal,foot_valid',[(1,True),(-1,True),(1,False)])
def test_goal_credit_is_foot_valid_while_conceding_always_costs(env,goal,foot_valid):
    env.reset();env.match_task.foot_valid[0]=foot_valid
    env.match_task.failures[0,0]=1  # A prior respawn must not suppress a real goal.
    _,_,_,term,trunc,info=measure(env,goal=[float(goal),0.])
    winning=0 if goal>0 else 1;losing=1-winning
    ledger=env.match_task.last_terms
    # Means include two worlds and four robots/team.
    assert ledger[MatchTask.terms.index('goal_for'),winning]==(10. if foot_valid else 0.)
    assert ledger[MatchTask.terms.index('goal_against'),losing]==-10.
    assert info['team_success'][0,winning]==foot_valid
    obs,rows=finish_course_step(env,term,trunc,info)
    assert len(rows)==2 and obs is not None and env.steps[0]==0
    assert env.match_task.last_residual==0


def test_timeout_ends_world_once_and_retains_terminal_potential_bootstrap(env):
    env.reset();env.steps[0]=1500
    env.match_task.has_foot[0]=True;env.match_task.advance_origin[0]=env.ball[0,0]-.5
    *_,term,trunc,info=measure(env)
    assert trunc[0].all() and not term[0].any()
    # No terminal-zero shaping on a time-limit truncation.
    assert env.match_task.last_terms[MatchTask.terms.index('ball_potential'),0].item()==pytest.approx(-.0025,abs=1e-6)
    _,rows=finish_course_step(env,term,trunc,info)
    assert len(rows)==2 and env.steps[0]==0


def test_first_foot_bonus_cannot_be_farmed_by_respawning_or_body_contact(env):
    env.reset();env.foot_hits[0,2,0]=1
    *_,info=measure(env)
    bonus=MatchTask.terms.index('foot_contact')
    assert env.match_task.last_terms[bonus,0].item()==pytest.approx(.25)
    assert env.match_task.foot_valid[0] and env.match_task.has_foot[0]
    # Moving the potential origin to the first touch does not give a free own-half bonus.
    assert env.match_task.last_terms[MatchTask.terms.index('ball_potential')].abs().max()<1e-6
    env.reset_agents(torch.tensor([[0,2]],device=env.device))
    measure(env);assert env.match_task.last_terms[bonus,0]==0
    env.foot_hits.zero_();env.nonfoot_ball_hits[0,2]=1
    measure(env)
    assert not env.match_task.foot_valid[0] and env.match_task.has_foot[0]
    assert env.match_task.last_terms[bonus].sum()==0


def test_respawn_avoids_occupied_original_position_and_ball(env):
    env.reset();point=env.match_task.spawn[0,2].clone()
    env.qpos[0,env.rootq[3]:env.rootq[3]+2]=point
    env.qpos[0,env.ballq:env.ballq+2]=point
    env.forward();env.reset_agents(torch.tensor([[0,2]],device=env.device))
    pos=env.qpos[0,env.rootq[2]:env.rootq[2]+2]
    assert torch.linalg.vector_norm(pos-point)>=.65
    assert torch.linalg.vector_norm(pos-env.ball[0,:2])>=.45
    assert env.observe()[0,2,-1].item()==pytest.approx(.45)


def test_negative_potential_cannot_be_repeatedly_collected_on_robot_respawns(env):
    env.reset();task=env.match_task;task.has_foot[0]=True
    task.advance_origin[0]=env.ball[0,0]+6.
    pairs=torch.tensor([[0,2]],device=env.device)
    original=task.advance_origin.clone()
    env.reset_agents(pairs)
    assert task.advance_origin[0,2]==env.ball[0,0]
    assert torch.equal(task.advance_origin[0,:2],original[0,:2])
    env.qpos[0,env.rootq[2]+2]=.1
    _,_,reward,_,_,_=measure(env)
    # No +6 inherited potential payment on the new individual trajectory.
    assert reward[0,2,0]<-4.9
    assert task.has_foot[0]  # Team history is preserved.


def test_kickoff_curriculum_moves_robots_and_keeps_midfield_ball_and_eight_players(env):
    import warp as wp
    env.match_layout='curriculum';env.reset()
    assert env.active.all() and not env.match_task.standard.any()
    assert (env.ball[:,:2].abs()<=.2).all()
    before=env.ball.clone()
    pos=env.qpos[:,env.rootq[:,None]+torch.arange(2,device=env.device)]
    distance=(pos[:,:,0]-env.ball[:,0:1]).abs().reshape(2,2,4)[:,:,1:].amin(-1)
    owner=env.match_task.kickoff_team
    own=distance[torch.arange(2,device=env.device),owner]
    assert (own>=.23).all() and (own<=.31).all()
    n=int(wp.to_torch(env.data.nacon)[0].item())
    geoms=wp.to_torch(env.data.contact.geom)[:n]
    mask=(geoms==env.reference.ballgeom).any(-1)
    assert not (wp.to_torch(env.data.contact.dist)[:n][mask]<=0).any()
    env.match_task.control.level=2;env.reset()
    pos=env.qpos[:,env.rootq[:,None]+torch.arange(2,device=env.device)]
    distance=(pos[:,:,0]-env.ball[:,0:1]).abs().reshape(2,2,4)[:,:,1:].amin(-1)
    own=distance[torch.arange(2,device=env.device),env.match_task.kickoff_team]
    assert (own>=.8).all() and (own<=1.1).all()
    assert (env.ball[:,:2].abs()<=.2).all()
    env.match_task.control=MatchControl();env.match_layout='mixed';env.reset()


def test_standard_rehearsal_is_preserved_and_has_separate_episode_evidence(env):
    env.reset();assert env.match_task.standard.tolist()==[True,False]
    env.steps.fill_(1500);env.match_task.quality_steps.fill_(1500)
    *_,info=measure(env)
    metrics=dict(zip(MatchTask.metrics,env.match_task.last_metrics.tolist()))
    assert metrics['standard_red']==metrics['standard_blue']==1
    assert metrics['near_red']+metrics['near_blue']==1
    assert metrics['touch_red']==metrics['touch_blue']==0


def test_unhealthy_closest_player_cannot_drop_its_chasing_responsibility(env):
    env.match_layout='standard';env.reset()
    env.qpos[0,env.rootq[2]]=env.ball[0,0]-.7
    env.qpos[0,env.rootq[2]+1]=env.ball[0,1]
    env.qpos[0,env.rootq[2]+2]=.28
    measure(env)
    penalty=env.match_task.last_terms[MatchTask.terms.index('approach_distance'),0]
    assert penalty<0
    env.match_layout='mixed';env.reset()
