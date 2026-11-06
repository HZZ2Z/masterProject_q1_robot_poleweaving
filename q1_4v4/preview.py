"""One GPU curriculum world with native MuJoCo state for the existing viewer."""
import torch
import mujoco
from .course_env import CourseWarpEnv
from .course_config import LOW_LEVEL, DURATIONS, FULL_MATCHES
from .match_task import MatchControl


class CoursePreview:
    dt=.02
    def __init__(self,stage,device='cuda:0',marching_target_observation='origin',match_control=None,match_layout='curriculum',match_gait_reward=True):
        self.env=CourseWarpEnv(count=1,device=device,stage=stage,episode_steps=DURATIONS[stage],marching_target_observation=marching_target_observation,match_layout=match_layout)
        if stage in FULL_MATCHES:
            self.env.match_task.control=MatchControl(match_control)
            self.env.match_task.gait_reward_enabled=match_gait_reward
            self.env.course_amplitude=self.env.match_task.control.amplitude
            self.env.reset()
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
        preview_info={'stage':self.env.course}
        if self.env.course in FULL_MATCHES:
            done=bool(info['world_terminated'].any());truncated=bool(info['world_truncated'].any())
            pairs=(info['robot_resets']&~(info['world_terminated']|info['world_truncated'])[:,None]).nonzero()
            if len(pairs):
                preview_info['reset_agents']=pairs[:,1].cpu().tolist()
                o,s=self.env.reset_agents(pairs)
        elif self.env.course in LOW_LEVEL:
            # Match training: a completed robot must not interrupt its seven peers.
            pairs=(d|t).nonzero(as_tuple=False)
            if len(pairs):
                preview_info.update(reset_agents=pairs[:,1].cpu().tolist(),
                                    terminated_agents=d[0].nonzero().flatten().cpu().tolist(),
                                    truncated_agents=t[0].nonzero().flatten().cpu().tolist())
                o,s=self.env.reset_agents(pairs)
            done=truncated=False  # Robot resets are handled here, not by play.py's world reset.
        else:
            done,truncated=bool(d.any()),bool(t.any())
        self.env.check();self.sync()
        return o[0].cpu().numpy(),s[0].cpu().numpy(),r[0].cpu().numpy(),done,truncated,preview_info
