"""Contact constraints, safe strike exceptions and checkpoint-mode continuity."""
from types import SimpleNamespace
import pytest
import torch
from q1_4v4.match_motion import MatchMotion,configure_motion
from q1_4v4.match_task import MatchControl


def state():
    n=1
    env=SimpleNamespace(count=n,active=torch.ones(n,8,dtype=torch.bool),dt=.02,
        gait_contact=torch.ones(n,8,2,dtype=torch.bool),foot_slip=torch.zeros(n,8,2),
        gait_bad_contact=torch.zeros(n,8,dtype=torch.bool),gait_command=torch.zeros(n,8,3),
        qpos=torch.zeros(n,176),qvel=torch.zeros(n,176),qidx=torch.arange(176).reshape(8,22),
        vidx=torch.arange(176).reshape(8,22),neutral=torch.zeros(22),
        match_task=SimpleNamespace(ball_skill=SimpleNamespace(strikes=torch.zeros(n,8,2,dtype=torch.bool))))
    motion=MatchMotion(n,'cpu');motion.enabled=True;motion.iteration=64
    return env,motion,torch.ones(n,8),torch.full((n,8),.417),torch.full((n,8),.1754)


def test_support_slide_cost_matches_physical_contact_and_never_pays_air_feet():
    e,m,up,z,w=state();r=m.evaluate(e,up,z,w)
    assert all(not v.any() for v in r.values()) and m.quality.min()==1
    e.foot_slip.fill_(.25);r=m.evaluate(e,up,z,w)
    assert r['support_slide'].max()==pytest.approx(-.06)
    assert m.quality.max()<.14
    e.gait_contact.zero_();r=m.evaluate(e,up,z,w)
    assert not r['support_slide'].any()


def test_strike_exempts_only_moving_ball_strike_foot_and_only_briefly():
    e,m,up,z,w=state();e.foot_slip[:,:,0]=.25
    assert m.evaluate(e,up,z,w)['support_slide'].min()<0
    e.match_task.ball_skill.strikes[:,:,0]=True
    for _ in range(5):assert not m.evaluate(e,up,z,w)['support_slide'].any()
    assert m.evaluate(e,up,z,w)['support_slide'].min()<0  # continuous ball dragging
    e.match_task.ball_skill.strikes.zero_();m.evaluate(e,up,z,w)
    e.match_task.ball_skill.strikes[:,:,0]=True;e.foot_slip[:,:,1]=.25
    assert m.evaluate(e,up,z,w)['support_slide'].min()<0  # support foot still slides


def test_air_time_cannot_replace_grounded_walking_and_individual_reset_is_local():
    e,m,up,z,w=state();e.gait_contact.zero_();e.gait_command[:,:,0]=.18
    for _ in range(12):r=m.evaluate(e,up,z,w)
    assert r['excess_flight'].max()==pytest.approx(-.06)
    m.reset_agents(torch.tensor([0]),torch.tensor([2]))
    assert not m.air_time[0,2] and m.air_time[0,1]>.2
    e.gait_contact[:,:,0]=True;r=m.evaluate(e,up,z,w)
    assert not r['excess_flight'].any() and not m.air_time.any()


def test_pose_margins_agree_with_gate_without_forcing_a_phase_or_neutral_legs():
    e,m,up,z,w=state();z[:,2]=.34;w[:,3]=.38;up[:,4]=.8
    r=m.evaluate(e,up,z,w)
    assert (r['posture_margin'][0,2:5]<0).all() and (m.quality[0,2:5]<1).all()
    # Only the nine unsupervised upper joints are restrained; arbitrary leg
    # angles alone do not produce a neutral-pose penalty.
    e.qpos[:,e.qidx[:,:12]]=1.
    assert not m.evaluate(e,up,z,w)['upper_motion'].any()
    e.qpos[:,e.qidx[:,14]]=1.;e.qvel[:,e.vidx[:,14]]=8.
    assert (m.evaluate(e,up,z,w)['upper_motion']<0).all()


def test_constraints_ramp_and_legacy_mode_resume_without_reward_changes():
    e,m,up,z,w=state();e.foot_slip.fill_(.25)
    m.iteration=0;r=m.evaluate(e,up,z,w)
    assert not r['support_slide'].any() and m.quality.min()==1
    m.iteration=32;assert m.evaluate(e,up,z,w)['support_slide'].max()==pytest.approx(-.03)
    m.enabled=False;assert all(not v.any() for v in m.evaluate(e,up,z,w).values())
    cfg,changed=configure_motion(SimpleNamespace(match_motion_constraints=None),{'iteration':200},200)
    assert not cfg['enabled'] and not changed
    cfg,changed=configure_motion(SimpleNamespace(match_motion_constraints=True),{'iteration':200},200)
    assert cfg['enabled'] and changed and cfg['start_iteration']==200
    restored,changed=configure_motion(SimpleNamespace(match_motion_constraints=None),{'motion_constraints':cfg},230)
    assert restored==cfg and not changed


@pytest.mark.parametrize('key,reason',[('owner_pose_pass','owner_posture'),
    ('owner_reset_pass','owner_respawns'),('owner_time_pass','short_episode')])
def test_gate_distinguishes_failures_without_lowering_joint_threshold(key,reason):
    c=MatchControl();evidence=dict.fromkeys(c.counters,16.)
    evidence.update(training_quality_red=0.,training_quality_blue=0.)
    evidence[key+'_red']=0.;evidence[key+'_blue']=0.
    for _ in range(32):c.update([.99,.99],[.5,.5],evidence)
    assert c.reason==reason and c.level==0 and c.windows==0


def test_warp_contact_constraints_preserve_inputs_and_reward_ledger():
    import os
    if os.environ.get('Q1_TEST_WARP')!='1':pytest.skip('Explicit CUDA opt-in')
    from q1_4v4.course_env import CourseWarpEnv
    from q1_4v4.mixed_env import finish_course_step
    from q1_4v4.match_task import MatchTask
    e=CourseWarpEnv(count=2,stage='match');e.match_task.gait_reward_enabled=False
    e.match_task.motion.enabled=True;e.match_task.motion.iteration=64
    penalty=0.
    for _ in range(12):
        obs,state,reward,term,trunc,info=e.step(torch.randn((2,8,22),device=e.device)*.3)
        assert obs.shape==(2,8,135) and state.shape==(2,8,1084)
        assert torch.isfinite(reward).all() and e.match_task.last_residual==0
        assert all(not e.match_task.last_terms[MatchTask.terms.index(k)].any() for k in e.match_task.gait_terms)
        penalty+=sum(e.match_task.last_terms[MatchTask.terms.index(k)].abs().sum().item() for k in MatchMotion.terms)
        assert 'owner_pose_pass' in info and 'owner_bad_fraction' in info
        finish_course_step(e,term,trunc,info)
    assert penalty>0
    e.check()
