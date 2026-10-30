"""Bounded GPU/native soccer contracts, not policy success acceptance."""
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from .soccer import SoccerEnvironment,load_config


def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1)
    config=load_config();config['episode_steps']=40
    env=SoccerEnvironment(4,config,85297)
    state=env.state();g=env.head.geometry
    near=env.groups==2
    relative=state[:,:2]-state[:,g.ball_q:g.ball_q+2]
    direction=relative/relative.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    state[near,env.model.nq+g.ball_v:env.model.nq+g.ball_v+2]=3*direction[near]
    env.physics.reset_indices(torch.arange(4,device=env.device),state[:,:env.model.nq],state[:,env.model.nq:])
    # Native reference from identical first world, without ball perturbation.
    native=mujoco.MjData(env.model);native.qpos[:]=state[0,:env.model.nq].cpu().numpy()
    native.qvel[:]=state[0,env.model.nq:].cpu().numpy()
    contacts=0;difference=0.
    for step in range(40):
        transition=env.step(torch.zeros(4,22,device=env.device))
        if step==0:
            for _ in range(env.observe.control.substeps):
                q=torch.tensor(native.qpos,dtype=torch.float32,device=env.device)
                v=torch.tensor(native.qvel,dtype=torch.float32,device=env.device)
                native.ctrl[:]=env.observe.control(q,v,torch.zeros(22,device=env.device)).cpu().numpy()
                mujoco.mj_step(env.model,native)
            difference=float(np.max(np.abs(transition['next_state'][0,:env.model.nq].cpu().numpy()-native.qpos)))
            if difference>2e-3:raise AssertionError(f'Native/Warp mismatch {difference}')
        contacts+=int(transition['contact'].sum())
    if not contacts:raise AssertionError('No actual foot/ball GPU contacts')
    if env.tracker.completed<4:raise AssertionError('Missing complete episodes')
    result=dict(worlds=4,control_steps=40,native_warp_max_qpos_difference=difference,
                actual_contact_transitions=contacts,summary=env.tracker.summary(),
                scope='Injected ball velocity contract check; not task success')
    print(json.dumps(result),flush=True)
    Path('logs/soccer_integration/physics_check.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
