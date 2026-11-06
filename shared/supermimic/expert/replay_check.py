"""Verify expert physical transitions by replaying recorded substep torques."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from .runtime import ROOT


def check(root,max_episodes=3):
    root=Path(root);rows=[json.loads(line) for line in (root/'index.jsonl').read_text().splitlines()]
    representatives={}
    for row in rows:representatives.setdefault(row['termination'],row)
    selected=list(representatives.values())[:max_episodes]
    scene=ROOT/'resources/q1_description/q1_soccer.xml'
    manifest=json.loads((root/'manifest.json').read_text())
    if hashlib.sha256(scene.read_bytes()).hexdigest()!=manifest['scene_sha256']:
        raise ValueError('Scene differs from collection')
    results=[]
    for row in selected:
        meta=json.loads((root/(row['episode_id']+'.json')).read_text())
        model=mujoco.MjModel.from_xml_path(str(scene));model.opt.timestep=meta['physics_dt']
        data=mujoco.MjData(model)
        max_qpos=max_qvel=max_warmstart=0.
        with np.load(root/(row['episode_id']+'.npz'),allow_pickle=False) as arrays:
            data.qpos[:]=arrays['qpos'][0];data.qvel[:]=arrays['qvel'][0];data.time=float(arrays['time'][0])
            mujoco.mj_forward(model,data);data.qacc_warmstart[:]=arrays['qacc_warmstart'][0]
            for t,torques in enumerate(arrays['applied_torques']):
                for torque in torques:
                    data.ctrl[:]=torque;mujoco.mj_step(model,data)
                max_qpos=max(max_qpos,float(np.abs(data.qpos-arrays['qpos'][t+1]).max()))
                max_qvel=max(max_qvel,float(np.abs(data.qvel-arrays['qvel'][t+1]).max()))
                max_warmstart=max(max_warmstart,float(np.abs(data.qacc_warmstart-arrays['qacc_warmstart'][t+1]).max()))
        if max_qpos>1e-9 or max_qvel>1e-9:
            raise AssertionError(f"Physical replay differs: {row['episode_id']} qpos={max_qpos} qvel={max_qvel}")
        results.append(dict(episode=row['episode_id'],termination=row['termination'],steps=row['steps'],
                 max_qpos_difference=max_qpos,max_qvel_difference=max_qvel,max_warmstart_difference=max_warmstart))
    report=dict(passed=True,episodes=results,training_started=False)
    (root/'physics_replay_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path)
    parser.add_argument('--max-episodes',type=int,default=3);args=parser.parse_args()
    print(json.dumps(check(args.dataset,args.max_episodes),indent=2),flush=True)
