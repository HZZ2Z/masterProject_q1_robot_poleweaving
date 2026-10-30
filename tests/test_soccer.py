import copy
import mujoco
import numpy as np
import torch

from q1_supermimic.physics import SCENE
from q1_supermimic.soccer import (load_config,SpawnSampler,SoccerGeometry,SoccerCamera,
                                SoccerHead,EpisodeTracker,Curriculum)
from q1_supermimic.observations import MotorObservation
from q1_supermimic.networks import MotorPolicy,mlp
from q1_supermimic.dynamics import Dynamics
from q1_supermimic.soccer_learning import soccer_update


def setup():
    return mujoco.MjModel.from_xml_path(str(SCENE)),load_config()


def test_fixed_spawns_near_gap_and_area_sampling():
    model,cfg=setup();sampler=SpawnSampler(model,cfg,12345)
    groups=np.repeat([0,1,2],[16,16,32])
    states,gaps=sampler.sample(groups)
    assert np.isfinite(states).all()
    assert (gaps[groups==2]>=.1-1e-5).all() and (gaps[groups==2]<=.25+1e-5).all()
    near=states[groups==2]
    forward=np.stack((1-2*near[:,6]**2,2*near[:,3]*near[:,6]),-1)
    np.testing.assert_allclose((forward*(-near[:,:2])).sum(-1)/np.linalg.norm(near[:,:2],axis=-1),1,atol=1e-6)
    assert np.linalg.norm(states[groups==1,:2],axis=-1).max()<=3.5
    assert np.all(states[:,model.nq:]==0)


def test_camera_masks_unseen_truth_and_preserves_delay():
    model,cfg=setup();sampler=SpawnSampler(model,cfg,17)
    state=torch.tensor(sampler.sample(np.array([2,2]))[0]);camera=SoccerCamera(model,cfg)
    # Place ball directly behind the robot, far outside physical camera FOV.
    state[:,3:7]=torch.tensor([1.,0,0,0]);state[:,:2]=0
    state[:,camera.geometry.ball_q:camera.geometry.ball_q+3]=torch.tensor([-2.,0,.1])
    first=camera.measurement(state,torch.Generator().manual_seed(8))
    state[:,camera.geometry.ball_q:camera.geometry.ball_q+3]=torch.tensor([-5.,1,.3])
    second=camera.measurement(state,torch.Generator().manual_seed(8))
    torch.testing.assert_close(first,torch.zeros_like(first));torch.testing.assert_close(second,first)
    queue=torch.arange(2*20*3,dtype=torch.float32).reshape(2,20,3)
    features,next_queue=camera.advance(state,queue,torch.tensor([6,6]),torch.tensor([2,2]),torch.Generator().manual_seed(7))
    torch.testing.assert_close(features[:,:3],queue[:,5])
    torch.testing.assert_close(next_queue[:,0],torch.zeros(2,3))


def test_timeout_not_fall_no_zero_failure_before_completed_episode():
    model,cfg=setup();geometry=SoccerGeometry(model)
    states=torch.tensor(SpawnSampler(model,cfg,32).sample(np.array([2,2,2,2]))[0])
    tracker=EpisodeTracker(4,cfg,'cpu');tracker.reset(torch.arange(4),states,geometry)
    assert 'full_episode_fall_rate' not in tracker.summary()
    tracker.update(states,geometry,torch.tensor([1,0,0,0]).bool(),torch.full((4,),600),
                   torch.tensor([0,1,0,0]).bool(),torch.zeros(4,dtype=torch.bool),
                   torch.ones(4,dtype=torch.bool),torch.tensor([0,1,2,2]))
    assert tracker.summary()['completed_episodes']==4
    assert tracker.summary()['full_episode_fall_rate']==.25


def test_near_phase_promotes_on_kick_without_requiring_goal():
    _,cfg=setup();curriculum=Curriculum(cfg)
    records=[dict(group=g,effective_kick=True,progress=.15,fall=False,approached=False,goal=False)
             for g in [0]*32+[1]*32+[2]*64]
    assert not curriculum.consider(records[:8])
    assert curriculum.consider(records) and curriculum.phase==1
    assert not curriculum.consider(records)


def test_visual_gradients_update_encoder_and_freeze_shared_motor_and_world():
    torch.manual_seed(16)
    model,cfg=setup();observe=MotorObservation(model);camera=SoccerCamera(model,cfg);head=SoccerHead(model,cfg)
    policy=MotorPolicy(720,540,70,22,motor_widths=(16,),visual_widths=(16,))
    world=Dynamics(model,widths=(16,));critic=mlp(144,1,(16,))
    state=torch.tensor(SpawnSampler(model,cfg,32).sample(np.array([2,2]))[0])
    history=observe(state,torch.zeros(2,22)).unsqueeze(1).repeat(1,10,1)
    visual=camera.features(state,torch.zeros(2,3)).unsqueeze(1).repeat(1,10,1)
    batch={k:v.unsqueeze(0) for k,v in dict(state=state,history=history,visual=visual,
           queue=torch.zeros(2,20,3),delay=torch.ones(2,dtype=torch.long)*6,
           age=torch.zeros(2,dtype=torch.long),context=torch.zeros(2,0)).items()}
    before=[{k:v.clone() for k,v in module.state_dict().items()} for module in (policy,world)]
    result=soccer_update(policy,world,critic,head,observe,camera,
            torch.optim.Adam(policy.visual_encoder.parameters(),lr=1e-4),
            torch.optim.Adam(critic.parameters(),lr=1e-4),batch,horizon=2)
    assert result['visual_grad_norm']>0 and all(np.isfinite(list(result.values())))
    assert all(torch.equal(value,before[0][key]) for key,value in policy.state_dict().items() if not key.startswith('visual_encoder.'))
    assert any(not torch.equal(value,before[0][key]) for key,value in policy.state_dict().items() if key.startswith('visual_encoder.'))
    assert all(torch.equal(value,before[1][key]) for key,value in world.state_dict().items())


def test_native_collision_contact_is_distinct_from_gap_proxy():
    model,cfg=setup();geometry=SoccerGeometry(model);sampler=SpawnSampler(model,cfg,4)
    state=sampler.sample(np.array([2]))[0][0];data=mujoco.MjData(model)
    data.qpos[:]=state[:model.nq]
    to_robot=data.qpos[:2]-data.qpos[geometry.ball_q:geometry.ball_q+2]
    data.qvel[geometry.ball_v:geometry.ball_v+2]=to_robot/np.linalg.norm(to_robot)*3
    observe=MotorObservation(model);hit=False
    for _ in range(200):
        q=torch.tensor(data.qpos,dtype=torch.float32);v=torch.tensor(data.qvel,dtype=torch.float32)
        data.ctrl[:]=observe.control(q,v,torch.zeros(22)).numpy()
        mujoco.mj_step(model,data)
        hit|=geometry.native_contact(data)
    assert hit


def test_unseen_ball_cannot_change_actor_features_or_history():
    model,cfg=setup();observe=MotorObservation(model);camera=SoccerCamera(model,cfg)
    state=torch.tensor(SpawnSampler(model,cfg,12).sample(np.array([2]))[0])
    state[:,:2]=0;state[:,3:7]=torch.tensor([1.,0,0,0])
    first=state.clone();second=state.clone()
    first[:,camera.geometry.ball_q:camera.geometry.ball_q+3]=torch.tensor([-2.,0,.1])
    second[:,camera.geometry.ball_q:camera.geometry.ball_q+3]=torch.tensor([-5.,1,.2])
    second[:,model.nq+camera.geometry.ball_v:]=10
    zero=torch.zeros(1,22)
    torch.testing.assert_close(observe(first,zero),observe(second,zero))
    queue=torch.zeros(1,20,3)
    a=camera.advance(first,queue,torch.tensor([0]),torch.tensor([2]),torch.Generator().manual_seed(1))[0]
    b=camera.advance(second,queue,torch.tensor([0]),torch.tensor([2]),torch.Generator().manual_seed(1))[0]
    torch.testing.assert_close(a,b)


def test_truth_ball_input_bypasses_fov_noise_delay_and_reset_blindness():
    model,cfg=setup();cfg['goal_input_mode']='truth'
    camera=SoccerCamera(model,cfg)
    state=torch.tensor(SpawnSampler(model,cfg,12).sample(np.array([2]))[0])
    state[:,:2]=0;state[:,3:7]=torch.tensor([1.,0,0,0])
    state[:,camera.geometry.ball_q:camera.geometry.ball_q+3]=torch.tensor([-2.,1.,.1])
    expected=torch.tensor([[-2.,1.,1.]])
    torch.testing.assert_close(camera.measurement(state),expected)
    torch.testing.assert_close(camera.initial_features(state)[:,:3],expected)
    features,queue=camera.advance(state,torch.ones(1,20,3)*99,torch.tensor([19]),torch.tensor([1]))
    torch.testing.assert_close(features[:,:3],expected)
    torch.testing.assert_close(queue[:,0],expected)
    # A ball that was outside vision must change the privileged actor input.
    state[:,camera.geometry.ball_q]=-3
    assert camera.measurement(state)[0,0]==-3


def test_joint_actor_updates_decoder_and_goal_encoder_but_not_world_or_reference():
    torch.manual_seed(16)
    model,cfg=setup();cfg['goal_input_mode']='truth'
    observe=MotorObservation(model);camera=SoccerCamera(model,cfg);head=SoccerHead(model,cfg)
    policy=MotorPolicy(720,540,70,22,motor_widths=(16,),visual_widths=(16,))
    policy.reference_encoder.requires_grad_(False)
    world=Dynamics(model,widths=(16,));critic=mlp(144,1,(16,))
    state=torch.tensor(SpawnSampler(model,cfg,32).sample(np.array([2,2]))[0])
    history=observe(state,torch.zeros(2,22)).unsqueeze(1).repeat(1,10,1)
    visual=camera.initial_features(state).unsqueeze(1).repeat(1,10,1)
    batch={k:v.unsqueeze(0) for k,v in dict(state=state,history=history,visual=visual,
           queue=torch.zeros(2,20,3),delay=torch.ones(2,dtype=torch.long)*6,
           age=torch.zeros(2,dtype=torch.long),context=torch.zeros(2,0)).items()}
    before=[{k:v.clone() for k,v in module.state_dict().items()} for module in (policy,world)]
    parameters=list(policy.visual_encoder.parameters())+list(policy.decoder.parameters())
    result=soccer_update(policy,world,critic,head,observe,camera,
            torch.optim.Adam(parameters,lr=1e-4),torch.optim.Adam(critic.parameters(),lr=1e-4),
            batch,horizon=2,train_decoder=True)
    assert result['decoder_grad_norm']>0 and result['decoder_output_weight_delta_norm']>0
    for prefix in ['decoder.','visual_encoder.']:
        assert any(not torch.equal(value,before[0][key]) for key,value in policy.state_dict().items() if key.startswith(prefix))
    assert all(torch.equal(value,before[0][key]) for key,value in policy.state_dict().items() if key.startswith('reference_encoder.'))
    assert all(torch.equal(value,before[1][key]) for key,value in world.state_dict().items())
    assert all(p.grad is None for p in world.parameters())


def test_ball_velocity_supervision_is_not_hidden_by_average_joint_loss():
    from q1_supermimic.soccer_dynamics import SoccerDynamics
    model,cfg=setup();cfg['ball_velocity_loss_weight']=2.
    world=SoccerDynamics(model,cfg,widths=(16,))
    target=torch.tensor(np.r_[model.qpos0,np.zeros(model.nv)],dtype=torch.float32).unsqueeze(0)
    ball=target.clone();joint=target.clone()
    ball[:,world.ball_v]+=1
    joint[:,model.nq+6]+=1
    assert world.error(ball,target)>10*world.error(joint,target)


def test_approach_retention_does_not_count_as_kicking_or_allow_more_falls():
    from train_soccer import select_candidate
    baseline=dict(completed_episodes=128,full_episode_fall_rate=0.,effective_kick_success_rate=0.,
                  contact_rate=0.,goal_rate=0.,ball_goal_net_progress_m=0.,near_foot_gap_reduction_m=0.)
    candidate=dict(baseline,near_foot_gap_reduction_m=.002)
    assert select_candidate(candidate,baseline,0,10)==(True,'approach_shaping')
    assert not select_candidate(dict(candidate,full_episode_fall_rate=.01),baseline,0,10)[0]
    assert not select_candidate(candidate,baseline,1,10)[0]
    assert not select_candidate(candidate,baseline,0,0)[0]
    assert not select_candidate(dict(candidate,near_foot_gap_reduction_m=.00001),baseline,0,10)[0]


def test_joint_initialization_preserves_the_motor_baseline_action():
    from train_soccer import make_policy
    torch.manual_seed(18)
    model,cfg=setup();observe=MotorObservation(model)
    source=MotorPolicy(720,540,4,22,motor_widths=(16,16),visual_widths=(64,64))
    policy=make_policy(model,observe,{'policy':source.state_dict()},16,train_decoder=True)
    state=torch.tensor(np.r_[model.qpos0,np.zeros(model.nv)],dtype=torch.float32).unsqueeze(0)
    state[:,observe.control.qpos_indices]=observe.control.neutral
    history=observe(state,torch.zeros(1,22)).repeat(1,10)
    goal=observe.reference(state).repeat(1,10)
    torch.testing.assert_close(source.track(history,goal),policy.soccer(history,torch.randn(1,350)))
    assert all(p.requires_grad for p in policy.decoder.parameters())
    assert all(not p.requires_grad for p in policy.reference_encoder.parameters())
