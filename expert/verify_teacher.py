"""Check recorded targets against the frozen teacher and causal observations."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import mujoco

from .runtime import ROOT
from .policy import ActorCritic
from .dataset import observation_history


def check(root,checkpoint=ROOT/'checkpoints/best.pth'):
    root=Path(root);checkpoint=Path(checkpoint)
    manifest=json.loads((root/'manifest.json').read_text())
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Teacher checkpoint differs from collection')
    config=yaml.safe_load((root/'config_snapshot.yaml').read_text())
    policy=ActorCritic(config);policy.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
    policy.eval();policy.requires_grad_(False)
    scene=mujoco.MjModel.from_xml_path(str(ROOT/'resources/q1_description/q1_soccer.xml'))
    limits=scene.jnt_range[scene.actuator_trnid[:,0]]
    rows=[json.loads(line) for line in (root/'index.jsonl').read_text().splitlines()]
    checked=0;max_difference=0.
    with torch.inference_mode():
        for row in rows:
            metadata=json.loads((root/(row['episode_id']+'.json')).read_text())
            names=metadata['joint_names'];defaults=config['init_state']['default_joint_angles']
            neutral=np.array([defaults.get(name,defaults['default']) for name in names],dtype=np.float32)
            scale=np.array(config['control']['action_scale'],dtype=np.float32)
            with np.load(root/(row['episode_id']+'.npz'),allow_pickle=False) as arrays:
                np.testing.assert_array_equal(np.clip(arrays['raw_actions'],-config['normalization']['clip_actions'],
                                                     config['normalization']['clip_actions']),arrays['actions'])
                targets=neutral+scale*arrays['actions']
                if manifest.get('control_contract')=='joint_target_and_torque_limits_v1':
                    targets=np.clip(targets,limits[:,0],limits[:,1]).astype(np.float32)
                np.testing.assert_array_equal(targets,arrays['pd_targets'])
                count=len(arrays['actions'])
                for index in sorted({0,count//2,count-1}) if count else []:
                    history=observation_history(arrays['observations'],index,arrays['initial_history'])
                    predicted=policy(torch.from_numpy(arrays['observations'][index:index+1]),torch.from_numpy(history[None]))[0].numpy()
                    difference=float(np.abs(predicted-arrays['raw_actions'][index]).max())
                    max_difference=max(max_difference,difference)
                    if not np.array_equal(predicted,arrays['raw_actions'][index]):
                        raise AssertionError(f'Teacher action differs: {row["episode_id"]} at {index}: {difference}')
                    checked+=1
    report=dict(passed=True,episodes=len(rows),checked_control_frames=checked,
                maximum_raw_action_difference=max_difference,history='causal newest-first 50 frames',training_started=False)
    (root/'teacher_action_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/best.pth');args=parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(check(args.dataset,args.checkpoint),indent=2),flush=True)
