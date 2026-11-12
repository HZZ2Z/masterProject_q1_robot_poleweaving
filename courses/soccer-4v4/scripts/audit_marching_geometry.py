"""Check Q1 foot-lift reachability using the retained neutral and policy target bounds.

This is a kinematic audit; it does not establish dynamic gait stability.
"""
import argparse,json,sys,hashlib,subprocess
from pathlib import Path
import numpy as np,mujoco,torch,yaml
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from q1_4v4.env import SoccerEnv
from q1_4v4.scene import ROOT
from q1_4v4.course_config import AMPLITUDE
from q1_4v4.gait import marching_shaping


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);a=p.parse_args()
 env=SoccerEnv();model,data=env.model,env.data
 cfg_path=ROOT/'shared/q1-mujoco/resources/Q1.yaml'
 cfg=yaml.safe_load(cfg_path.read_text());names=cfg['asset']['dof_names']
 low,high=np.array(cfg['asset']['foot_collision_bounds'])
 sole=np.array([[x,y,low[2]] for x in (low[0],high[0]) for y in (low[1],high[1])])
 results=[]
 for robot,prefix in ((0,'red0_'),(4,'blue0_')):
  for side in ('left','right'):
   for height in (.02,.025,.03,.04,.05):
    for bounded in (True,False):
     env.reset();base=data.qpos.copy();root=env.bodies[robot];R=data.xmat[root].reshape(3,3).copy();pelvis=data.xpos[root].copy()
     feet=[model.body(prefix+s+'_ankle_roll_link').id for s in ('left','right')]
     knees=[model.body(prefix+s+'_knee_link').id for s in ('left','right')]
     neutral_knee=(data.xpos[knees]-pelvis)@R
     index=0 if side=='left' else 1;foot=feet[index]
     foot_xy=data.xpos[foot,:2].copy();targetR=data.xmat[foot].reshape(3,3).copy()
     initial_clearance=(data.xpos[foot]+sole@targetR.T)[:,2].min()
     target_z=data.xpos[foot,2]+height-initial_clearance
     local_indices=np.array([i for i,n in enumerate(names) if n.startswith(side+'_') and any(k in n for k in ('hip_','knee_','ankle_'))])
     qidx=env.qidx[robot,local_indices];neutral=env.neutral[local_indices]
     delta=AMPLITUDE['step_in_place']*env.scale[local_indices]
     limits=env.limits[local_indices].copy()
     if bounded:limits[:,0]=np.maximum(limits[:,0],neutral-delta);limits[:,1]=np.minimum(limits[:,1],neutral+delta)
     def residual(q):
      data.qpos[:]=base;data.qpos[qidx]=q;mujoco.mj_forward(model,data)
      rot=data.xmat[foot].reshape(3,3)
      clearance=(data.xpos[foot]+sole@rot.T)[:,2].min()
      # Foot may move forward during marching; penalize forward excursion only weakly.
      xy=(data.xpos[foot,:2]-foot_xy)@R[:2,:2]
      orientation=Rotation.from_matrix(targetR.T@rot).as_rotvec()
      return np.r_[20*xy[1],20*(data.xpos[foot,2]-target_z),2*orientation,.01*xy[0]]
     fit=least_squares(residual,neutral,bounds=(limits[:,0]+1e-7,limits[:,1]-1e-7),max_nfev=400,ftol=1e-11,xtol=1e-11,gtol=1e-11)
     residual(fit.x)
     footlocal=(data.xpos[feet]-pelvis)@R;kneelocal=(data.xpos[knees]-pelvis)@R
     clear=np.array([(data.xpos[f]+sole@data.xmat[f].reshape(3,3).T)[:,2].min() for f in feet])
     expected=torch.tensor([[index!=0,index!=1]])
     tensor=lambda x:torch.tensor(x,dtype=torch.float32)
     score=marching_shaping(tensor(footlocal)[None],tensor(kneelocal)[None],tensor(neutral_knee),tensor(clear)[None],torch.full((1,2),height),expected,expected,*[torch.zeros(1) for _ in range(4)])[0].item()
     results.append({'robot':robot,'side':side,'target_m':height,'policy_bounds':bounded,'achieved_m':float(clear[index]),'orientation_error_rad':float(np.linalg.norm(residual(fit.x)[2:5]/2)),'normalized_target_max':float(np.abs((fit.x-neutral)/delta).max()),'knee_credit':score,'knee_rise_m':float(kneelocal[index,2]-neutral_knee[index,2])})
 original_cfg=yaml.safe_load(subprocess.check_output(['git','show','HEAD:shared/q1-mujoco/resources/Q1.yaml'],cwd=ROOT,text=True))
 report={'neutral_unchanged':cfg['init_state']['default_joint_angles']==original_cfg['init_state']['default_joint_angles'],'config_sha256':hashlib.sha256(cfg_path.read_bytes()).hexdigest(),'kinematic_only':True,'results':results}
 Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
