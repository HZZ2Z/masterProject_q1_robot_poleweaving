"""Verify written expert shards, split separation and outcome counts."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from .dataset import validate_arrays,outcome


def check(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    rows=[json.loads(line) for line in (root/'index.jsonl').read_text().splitlines()]
    seeds=set();ids=set();splits=defaultdict(set);outcomes=Counter();cohorts=Counter();total=0;imitation=0;contact=0
    for row in rows:
        identifier=row['episode_id']
        if identifier in ids or row['seed'] in seeds:raise ValueError('Duplicate episode or seed')
        ids.add(identifier);seeds.add(row['seed']);splits[row['split']].add(row['seed'])
        metadata=json.loads((root/(identifier+'.json')).read_text())
        path=root/(identifier+'.npz')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['trajectory_sha256']:raise ValueError('Shard checksum mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            t=validate_arrays(arrays,metadata);total+=t
            expected=outcome(metadata['termination'],arrays['foot_ball_contact'],metadata['post_touch_max_progress_m'],
                             metadata['duration_s'],metadata['first_touch_s'])
            for key,value in expected.items():
                if metadata[key]!=value:raise ValueError('Outcome inconsistency: '+key)
        for key in row:
            if row[key]!=metadata[key]:raise ValueError('Index/metadata mismatch: '+key)
        if metadata['checkpoint_sha256']!=manifest['checkpoint_sha256']:raise ValueError('Mixed teacher weights')
        outcomes[metadata['termination']]+=1;cohorts[metadata['spawn_group']]+=1
        imitation+=int(metadata['policy_imitation_eligible']);contact+=int(metadata['contact'])
    if len(rows)!=manifest['completed_episodes']:raise ValueError('Episode count mismatch')
    if total!=manifest['total_transitions']:raise ValueError('Transition count mismatch')
    names=list(splits)
    for i,name in enumerate(names):
        for other in names[i+1:]:
            if splits[name]&splits[other]:raise ValueError('Episode leakage across dataset splits')
    report=dict(passed=True,episodes=len(rows),transitions=total,outcomes=dict(outcomes),spawn_groups=dict(cohorts),
                contact_episodes=contact,policy_imitation_episodes=imitation,
                splits={key:len(value) for key,value in splits.items()},training_started=False)
    (root/'dataset_check.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('dataset',type=Path);args=parser.parse_args()
    print(json.dumps(check(args.dataset),indent=2),flush=True)


if __name__=='__main__':main()
