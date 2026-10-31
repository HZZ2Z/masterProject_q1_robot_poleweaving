"""Read-only extraction parity check; executes inference and physics, never training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT=Path(__file__).resolve().parent

# Each interpreter imports only its own package. The original workspace is used
# exclusively for this comparison and is not an expert runtime dependency.
TRACE = r'''
import json,sys,pathlib
import numpy as np
import torch,mujoco
module,checkpoint,seeds,steps,output=sys.argv[1:]
torch.set_num_threads(2)
if module=='original':
 from play_mujoco import SoccerPlayer
else:
 from expert.runtime import SoccerPlayer
result={};summaries=[]
for seed in json.loads(seeds):
 p=SoccerPlayer(checkpoint,seed=seed,reset_mode='reference',task='Q1')
 obs=[];history=[];actions=[];states=[];torques=[]
 forward=p.policy.forward
 def policy(o,h):
  a=forward(o,h)
  obs.append(o.numpy().copy());history.append(h.numpy().copy());actions.append(a.numpy().copy())
  return a
 p.policy.forward=policy
 result[str(seed)+'_initial']=np.r_[p.data.qpos,p.data.qvel]
 cause=None
 for _ in range(int(steps)):
  cause=p.step(auto_reset=False)
  if cause:break
  states.append(np.r_[p.data.qpos,p.data.qvel].copy());torques.append(p.data.ctrl.copy())
 for name,values in [('observations',obs),('history',history),('raw_actions',actions),('states',states),('torques',torques)]:
  result[str(seed)+'_'+name]=np.array(values)
 summaries.append(dict(seed=seed,spawn_group=p.spawn_kind,physics_steps=p.physics_steps,terminal=cause))
np.savez(output,**result)
pathlib.Path(output+'.json').write_text(json.dumps(summaries))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/best.pth')
    parser.add_argument('--physics-steps',type=int,default=500)
    parser.add_argument('--output',type=Path,default=ROOT/'validation/parity.json')
    args=parser.parse_args()
    if args.physics_steps<1:parser.error('physics-steps must be positive')
    seeds=list(range(42,50))
    env=os.environ.copy()
    for key in ['PYTHONPATH','PYTHONHOME','LD_LIBRARY_PATH']:env.pop(key,None)
    env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    maxima={};entries=[]
    with tempfile.TemporaryDirectory(prefix='q1-expert-parity-') as temporary:
        paths=[]
        for mode,cwd in [('original',args.source_root),('extracted',ROOT.parent)]:
            path=Path(temporary)/(mode+'.npz');paths.append(path)
            subprocess.run([sys.executable,'-c',TRACE,mode,str(args.checkpoint.resolve()),json.dumps(seeds),
                            str(args.physics_steps),str(path)],cwd=cwd,env=env,check=True)
        with np.load(paths[0],allow_pickle=False) as old,np.load(paths[1],allow_pickle=False) as new:
            if old.files!=new.files:raise AssertionError('Trace schema differs')
            for name in old.files:
                if old[name].shape!=new[name].shape:raise AssertionError(f'Trace shape differs: {name}')
                difference=float(np.max(np.abs(old[name]-new[name]))) if old[name].size else 0.
                maxima[name.split('_',1)[1]]=max(difference,maxima.get(name.split('_',1)[1],0.))
                if not np.array_equal(old[name],new[name]):
                    raise AssertionError(f'Extraction changed {name}: max difference {difference}')
        entries=json.loads(Path(str(paths[0])+'.json').read_text())
        if entries!=json.loads(Path(str(paths[1])+'.json').read_text()):raise AssertionError('Terminal/cohort mismatch')
    if {r['spawn_group'] for r in entries}!={0,1,2}:raise AssertionError('Missing spawn cohort coverage')
    report=dict(passed=True,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                comparison='bitwise equality with original playback in separate interpreters',
                maximum_differences=maxima,cases=entries,training_started=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
