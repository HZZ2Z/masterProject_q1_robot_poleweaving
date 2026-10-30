"""Paired native action-response audit on independent physical trajectories.

The scalar Q is a one-step diagnostic using the frozen learned critic, not a
claim about true long-term value or the entire 32-step policy gradient.
"""
import argparse
import copy
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from .physics import SCENE
from .robot import Q1Control
from .observations import MotorObservation
from .tracking import TrackingHead
from .dynamics import Dynamics
from .networks import MotorPolicy, mlp


def agreement(predicted, actual):
    p, a = np.asarray(predicted), np.asarray(actual)
    denominator = np.linalg.norm(p)*np.linalg.norm(a)
    return dict(cosine=float(np.dot(p,a)/denominator) if denominator>1e-15 else None,
                relative_error=float(np.linalg.norm(p-a)/max(np.linalg.norm(a),1e-12)),
                actual_rms=float(np.sqrt(np.mean(a*a))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--seed',type=int,default=381731)
    parser.add_argument('--trajectories',type=int,default=64)
    parser.add_argument('--output',type=Path,default=Path('logs/motor_response_audit.json'))
    args=parser.parse_args()
    if args.trajectories < 1: parser.error("trajectories must be positive")
    torch.set_num_threads(2)
    model=mujoco.MjModel.from_xml_path(str(SCENE))
    control=Q1Control(model); observe=MotorObservation(model); head=TrackingHead(model)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=saved.get('config',{})
    mw,pw=config.get('model_width',256),config.get('policy_width',128)
    world=Dynamics(model,widths=(mw,mw,mw))
    policy=MotorPolicy(720,540,4,22,motor_widths=(pw,pw),visual_widths=(64,64))
    critic=mlp(610,1,(mw,mw//2))
    for name,module in [('world',world),('policy',policy),('critic',critic)]:
        module.load_state_dict(saved[name])
    action_noise=config.get('exploration_std',.05)
    random=np.random.default_rng(args.seed)
    reference=torch.tensor(np.r_[model.qpos0,np.zeros(model.nv)],dtype=torch.float32)
    reference[control.qpos_indices]=control.neutral
    goal=observe.reference(reference).repeat(10)
    snapshots=[];histories=[]
    terminated_trajectories = 0
    for _ in range(args.trajectories):
        data=mujoco.MjData(model)
        data.qpos[:]=reference[:model.nq].numpy()
        data.qpos[control.qpos_indices.numpy()]+=random.normal(0,.005,22)
        mujoco.mj_forward(model,data)
        state=torch.tensor(np.r_[data.qpos,data.qvel],dtype=torch.float32)
        history=observe(state,torch.zeros(22)).repeat(10,1)
        for t in range(151):
            quat=data.qpos[3:7]
            if abs(data.qpos[2]-reference[2].item())>.25 or 2*(quat[1]**2+quat[2]**2)>.8:
                terminated_trajectories += 1
                break
            if t in (0,25,75,150):
                snapshots.append(copy.deepcopy(data));histories.append(history.clone())
            with torch.no_grad():
                action=policy.track(history.flatten(),goal)+torch.tensor(random.normal(0,action_noise,22),dtype=torch.float32)
                for _ in range(control.substeps):
                    state=torch.tensor(np.r_[data.qpos,data.qvel],dtype=torch.float32)
                    data.ctrl[:]=control(state[:model.nq],state[model.nq:],action).numpy()
                    mujoco.mj_step(model,data)
                state=torch.tensor(np.r_[data.qpos,data.qvel],dtype=torch.float32)
                history=torch.cat((history[1:],observe(state,action).unsqueeze(0)))
    states=torch.tensor(np.array([np.r_[d.qpos,d.qvel] for d in snapshots]),dtype=torch.float32)
    histories=torch.stack(histories);count=len(states)
    direction=torch.tensor(random.normal(size=(count,22)),dtype=torch.float32)
    direction=direction/direction.norm(dim=-1,keepdim=True)
    epsilon=.05*np.sqrt(22)
    previous=histories[:,-1,-22:]
    with torch.no_grad(): center=policy.track(histories.flatten(1),goal.expand(count,-1))
    def scalar(next_state,action):
        reward,_,done=head(next_state,reference.expand_as(next_state),action,previous)
        value=critic(torch.cat((next_state,goal.expand(count,-1)),-1)).squeeze(-1)
        return reward,reward+.99*(1-done)*value
    actual=[];predicted=[]
    for sign in (-1,1):
        action=center+sign*epsilon*direction
        following=[]
        for index,snapshot in enumerate(snapshots):
            data=copy.deepcopy(snapshot)
            for _ in range(control.substeps):
                with torch.no_grad():
                    q=torch.tensor(data.qpos,dtype=torch.float32);v=torch.tensor(data.qvel,dtype=torch.float32)
                    data.ctrl[:]=control(q,v,action[index]).numpy()
                mujoco.mj_step(model,data)
            following.append(np.r_[data.qpos,data.qvel])
        with torch.no_grad():
            actual.append(scalar(torch.tensor(np.array(following),dtype=torch.float32),action))
            predicted.append(scalar(world(states,action,torch.zeros(count,0)),action))
    with torch.enable_grad():
        action=center.detach().requires_grad_()
        reward,qvalue=scalar(world(states,action,torch.zeros(count,0)),action)
        reward_gradient=torch.autograd.grad(reward.sum(),action,retain_graph=True)[0]
        q_gradient=torch.autograd.grad(qvalue.sum(),action)[0]
    result=dict(seed=args.seed,independent_trajectories=args.trajectories,
                terminated_trajectories=terminated_trajectories,pairs=count,epsilon_l2=epsilon,
                scope='pre-terminal states from noisy native standing rollouts; samples clustered by trajectory; frozen one-step critic diagnostic; no kicking evidence',
                collection_action_noise_std=action_noise,collection_seconds=3.,
                rollout_prefix_control_steps=[0,25,75,150])
    for i,name in enumerate(('reward','critic_bootstrapped_q')):
        actual_delta=((actual[1][i]-actual[0][i])/(2*epsilon)).numpy()
        predicted_delta=((predicted[1][i]-predicted[0][i])/(2*epsilon)).numpy()
        analytic=((reward_gradient if i==0 else q_gradient)*direction).sum(-1).detach().numpy()
        result[name]=dict(finite_response=agreement(predicted_delta,actual_delta),
                          analytic_gradient=agreement(analytic,actual_delta),
                          analytic_vs_model_finite=agreement(analytic,predicted_delta))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
