"""Training-only adversarial motion prior; never adds actor observations."""
from pathlib import Path
import csv
import hashlib
import numpy as np
import mujoco
import torch
from torch import nn
import torch.distributed as dist
import yaml
from .scene import ROOT


class MotionExpert:
    def __init__(self, path, device):
        cfg = yaml.safe_load((ROOT / 'resources/Q1.yaml').read_text())
        names = cfg['asset']['dof_names']
        files = sorted(Path(path).glob('*.csv'))
        if not files:
            raise ValueError('AMP requires Q1 motion CSV files')
        model = mujoco.MjModel.from_xml_path(str(ROOT / 'resources/q1_description/q1_soccer.xml'))
        data = mujoco.MjData(model)
        addresses = [model.jnt_qposadr[model.joint(n).id] for n in names]
        root = model.body('pelvis').id
        feet = [model.body(n).id for n in cfg['asset']['foot_names']]
        rootq = model.joint('floating_base_joint').qposadr[0]
        pairs = [];digest = hashlib.sha256()
        header = ['x','y','z','qx','qy','qz','qw','v_x','v_y','v_z','w_x','w_y','w_z']
        header += ['q_' + n for n in names] + ['dq_' + n for n in names]
        for file in files:
            digest.update(file.name.encode());digest.update(file.read_bytes())
            with file.open() as stream:
                rows = list(csv.reader(stream))
            values = np.asarray(rows[1:], dtype=np.float32)
            if rows[0] != header or values.ndim != 2 or values.shape[1] != 57 or not np.isfinite(values).all():
                raise ValueError('Invalid Q1 motion schema: ' + str(file))
            foot_pos = np.empty((len(values), 6), dtype=np.float32)
            data.qpos[rootq:rootq+7] = [0,0,0,1,0,0,0]
            for index, row in enumerate(values):
                data.qpos[addresses] = row[13:35]
                mujoco.mj_kinematics(model, data)
                foot_pos[index] = (data.xpos[feet] - data.xpos[root]).reshape(-1)
            q = torch.as_tensor(values[:, 3:7], device=device)
            q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            gravity = torch.zeros((len(q),3), device=device);gravity[:,2] = -1
            v, w = q[:,:3], q[:,3:4]
            projected = gravity * (2*w.square()-1) - 2*w*torch.cross(v,gravity,dim=-1) + 2*v*(v*gravity).sum(-1,keepdim=True)
            features = torch.cat((projected, torch.as_tensor(values[:,7:13],device=device),
                                  torch.as_tensor(values[:,13:35],device=device),
                                  torch.as_tensor(values[:,35:57],device=device)*.05,
                                  torch.as_tensor(foot_pos,device=device)), -1)
            safe = (features[:,:9].square().sum(-1) < 100)
            pairs.append(torch.cat((features[:-1],features[1:]),-1)[safe[:-1]&safe[1:]])
        self.pairs = torch.cat(pairs)
        if len(self.pairs)<2:raise ValueError('No valid consecutive AMP expert frames')
        self.sha256 = digest.hexdigest()
        supervised = set(cfg['motion']['amp_joint_names'])
        mask = torch.ones(59,device=device)
        for index,name in enumerate(names):
            if name not in supervised:mask[9+index]=mask[31+index]=0
        frames = self.pairs.reshape(-1,59)
        self.mean = frames.mean(0)
        self.std = frames.std(0,unbiased=False).clamp(min=.1)
        self.mask = mask

    def sample(self, count):
        return self.pairs[torch.randint(len(self.pairs),(count,),device=self.pairs.device)]


class MotionPrior(nn.Module):
    def __init__(self, path, device, coefficient, start_iteration, saved=None):
        super().__init__()
        self.expert = MotionExpert(path,device)
        self.discriminator = nn.Sequential(nn.Linear(118,128),nn.Tanh(),nn.Linear(128,64),nn.Tanh(),nn.Linear(64,1)).to(device)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(),lr=1e-4)
        self.coefficient = coefficient
        self.start_iteration = start_iteration
        self.iteration = start_iteration
        self.register_buffer('mean',self.expert.mean)
        self.register_buffer('std',self.expert.std)
        self.register_buffer('mask',self.expert.mask)
        self.replay = torch.zeros((65536,118),device=device)
        self.size = self.cursor = 0
        self.previous = {}
        self.statistics = {}
        if saved:
            if saved['motion_sha256']!=self.expert.sha256:raise ValueError('AMP motion data differs from checkpoint')
            self.load_state_dict(saved['model'])
            self.optimizer.load_state_dict(saved['optimizer'])
            self.start_iteration = saved['start_iteration']
        if dist.is_initialized():
            for p in self.parameters():dist.broadcast(p.data,0)

    @staticmethod
    def features(env):
        pos = env.xpos[:,env.bodies]
        rot = env.xmat[:,env.bodies].reshape(env.count,8,3,3)
        velocity = env.qvel[:,env.rootv[:,None]+torch.arange(6,device=env.device)]
        local = lambda v:torch.einsum('nai,naij->naj',v,rot)
        feet = torch.einsum('nafi,naij->nafj',env.xpos[:,env.feet]-pos[:,:,None],rot).flatten(-2)
        # MuJoCo free-joint rotational qvel is already in the body's axes.
        return torch.cat((-rot[:,:,2],local(velocity[:,:,:3]),velocity[:,:,3:],
                          env.qpos[:,env.qidx],env.qvel[:,env.vidx]*.05,feet),-1)

    def normalized(self, pairs):
        return (((pairs.reshape(-1,2,59)-self.mean)/self.std).clamp(-5,5)*self.mask).flatten(-2)

    @torch.no_grad()
    def begin_step(self, groups):
        self.previous.clear()
        for _,group in groups:
            group.match_task.amp_reward_provider=self.reward
            self.previous[id(group.match_task)]=(self.features(group),group.xpos[:,group.feet].clone())

    @torch.no_grad()
    def reward(self, env):
        previous, feet = self.previous[id(env.match_task)]
        following = self.features(env)
        pairs = torch.cat((previous,following),-1)
        selected = pairs[::16].reshape(-1,118)
        length=min(len(selected),len(self.replay));selected=selected[-length:]
        slots=(torch.arange(length,device=pairs.device)+self.cursor)%len(self.replay)
        self.replay[slots]=selected;self.cursor=(self.cursor+length)%len(self.replay);self.size=min(len(self.replay),self.size+length)
        score = self.discriminator(self.normalized(pairs)).reshape(env.count,8)
        style = (1-.25*(score-1).square()).clamp(0,1)
        speed = torch.linalg.vector_norm(following[:,:,3:6],dim=-1)
        foot_speed = torch.linalg.vector_norm((env.xpos[:,env.feet]-feet)/env.dt,dim=-1).amax(-1)
        healthy = (following[:,:,2]<-.8)&(env.xpos[:,env.bodies,2]>.30)&~env.gait_bad_contact.bool()
        productive = (speed>.02)|(foot_speed>.05)
        task = env.match_task.ball_skill
        gate = healthy*productive*(1-torch.maximum(task.near_stall,task.far_stall))
        ramp = min(1.,max(0.,(self.iteration-self.start_iteration)/200))
        return env.dt*self.coefficient*ramp*style*gate*env.active

    def update(self):
        available=torch.tensor(self.size,device=self.mean.device)
        if dist.is_initialized():dist.all_reduce(available,op=dist.ReduceOp.MIN)
        if available.item()<2:return {}
        totals=torch.zeros(4,device=self.mean.device)
        for _ in range(4):
            policy=self.normalized(self.replay[torch.randint(self.size,(1024,),device=self.mean.device)])
            expert=self.normalized(self.expert.sample(1024))
            expert.requires_grad_(True)
            real=self.discriminator(expert);fake=self.discriminator(policy)
            gradient=torch.autograd.grad(real.sum(),expert,create_graph=True)[0]
            penalty=gradient.square().sum(-1).mean()
            loss=.5*((real-1).square().mean()+(fake+1).square().mean())+5*penalty
            self.optimizer.zero_grad();loss.backward()
            if dist.is_initialized():
                for p in self.parameters():
                    dist.all_reduce(p.grad);p.grad.div_(dist.get_world_size())
            nn.utils.clip_grad_norm_(self.parameters(),1.,error_if_nonfinite=True)
            self.optimizer.step()
            totals+=torch.stack((loss.detach(),penalty.detach(),real.detach().mean(),fake.detach().mean()))
        if not torch.isfinite(totals).all():raise FloatingPointError('Nonfinite AMP update')
        return dict(zip(('amp_loss','amp_gradient_penalty','amp_expert_score','amp_policy_score'),(totals/4).tolist()))

    def checkpoint(self):
        return dict(model=self.state_dict(),optimizer=self.optimizer.state_dict(),
                    coefficient=self.coefficient,start_iteration=self.start_iteration,motion_sha256=self.expert.sha256)

    def parameter_difference(self):
        if not dist.is_initialized():return 0.
        local=torch.cat([p.detach().flatten() for p in self.parameters()])
        reference=local.clone();dist.broadcast(reference,0)
        difference=(local-reference).abs().max();dist.all_reduce(difference,op=dist.ReduceOp.MAX)
        if difference.item()!=0:raise RuntimeError('Distributed AMP parameter divergence')
        return difference.item()
