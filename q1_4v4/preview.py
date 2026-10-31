"""One GPU curriculum world with native MuJoCo state for the existing viewer."""
import torch
import mujoco
from .course_env import CourseWarpEnv
from .course_config import LOW_LEVEL, DURATIONS


class CoursePreview:
    dt=.02
    def __init__(self,stage,device='cuda:0',marching_target_observation='origin'):
        self.env=CourseWarpEnv(count=1,device=device,stage=stage,episode_steps=DURATIONS[stage],marching_target_observation=marching_target_observation)
        self.model=self.env.model
        self.data=mujoco.MjData(self.model)
        for key in ('observation_space','share_observation_space','action_space'):
            setattr(self,key,getattr(self.env,key))
        self.sync()

    def sync(self):
        self.data.qpos[:]=self.env.qpos[0].cpu().numpy()
        self.data.qvel[:]=self.env.qvel[0].cpu().numpy()
        mujoco.mj_forward(self.model,self.data)

    def observe(self): return self.env.observe()[0].cpu().numpy()
    def state(self): return self.env.state()[0].cpu().numpy()
    def reset(self):
        self.env.reset();self.sync()
        return self.observe(),self.state()
    def step(self,action):
        o,s,r,d,t,info=self.env.step(torch.as_tensor(action,device=self.env.device,dtype=torch.float32)[None])
        self.env.check();self.sync()
        # Preview reopens the entire rally on a fall; training uses independent low-level resets.
        return o[0].cpu().numpy(),s[0].cpu().numpy(),r[0].cpu().numpy(),bool(d.any()),bool(t.any()),{'stage':self.env.course}
