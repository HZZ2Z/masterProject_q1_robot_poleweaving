"""Bounded GPU physics/learner checks. Run only on an authorized available GPU."""
from pathlib import Path
import os
import subprocess
import sys
import json
import torch
ROOT = Path(__file__).resolve().parents[3]


def run(*args, extra_env=None):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True,
                   env={**os.environ, **(extra_env or {})})


run('-m', 'pytest', '-q', 'tests', extra_env={'Q1_TEST_WARP': '1'})
for stage in ('balance', 'move', 'approach', 'kick', 'teamplay', 'match'):
    run('courses/soccer-4v4/train.py', '--backend', 'warp', '--device', 'cuda:0', '--no-wandb',
        '--num-envs', '4', '--horizon', '8', '--iterations', '1', '--epochs', '1',
        '--stage', stage, '--episode-steps', '4', '--job-type', 'smoke', '--output', 'logs/warp_stage_'+stage)
run('courses/soccer-4v4/train.py', '--backend', 'warp', '--device', 'cuda:0', '--no-wandb',
    '--num-envs', '4', '--horizon', '8', '--iterations', '1', '--epochs', '1',
    '--checkpoint', 'logs/warp_stage_balance/last.pt', '--output', 'logs/warp_resume', '--job-type', 'smoke')
ckpt = torch.load(ROOT/'logs/warp_resume/last.pt', weights_only=False, map_location='cpu')
assert ckpt['iteration'] == 2 and ckpt['backend'] == 'warp'
assert ckpt['policy_sharing']=='shared' and len(ckpt['teams'])==1
for team in ckpt['teams']:
    assert all(x['step'] == 4 for x in team['actor_optimizer']['state'].values())
    assert len(team['value_normalizer']) == 3
    assert team['value_normalizer']['debiasing_term'] > 0
run('courses/soccer-4v4/play.py', '--checkpoint', 'logs/warp_resume/last.pt', '--headless', '--steps', '30')
report = {'passed': True, 'tests': 29, 'backend': 'MuJoCo Warp CUDA', 'stages': 6,
          'resume_iteration': 2, 'formal_training_started': False}
(ROOT/'artifacts/warp_smoke_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
