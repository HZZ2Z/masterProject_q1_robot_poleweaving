"""Bounded offline checks only. Does not launch long-running training."""
from pathlib import Path
import subprocess
import sys
import json
import torch
ROOT = Path(__file__).resolve().parents[3]


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


run('-m', 'pytest', '-q', 'tests')
for stage in ('balance', 'move', 'approach', 'kick', 'teamplay', 'match'):
    run('courses/soccer-4v4/train.py', '--backend', 'cpu', '--device', 'cpu', '--no-wandb', '--num-envs', '2', '--horizon', '8', '--iterations', '1',
        '--epochs', '1', '--stage', stage, '--output', 'logs/smoke_'+stage,
        '--episode-steps', '4', '--job-type', 'smoke')
run('courses/soccer-4v4/train.py', '--backend', 'cpu', '--device', 'cpu', '--no-wandb', '--num-envs', '2', '--horizon', '8', '--iterations', '1',
    '--epochs', '1', '--checkpoint', 'logs/smoke_balance/last.pt',
    '--output', 'logs/smoke_resume', '--job-type', 'smoke')
ckpt = torch.load(ROOT/'logs/smoke_resume/last.pt', weights_only=False, map_location='cpu')
assert ckpt['iteration'] == 2
assert all(v['step'] == 2 for v in ckpt['teams'][0]['actor_optimizer']['state'].values())
run('courses/soccer-4v4/play.py', '--checkpoint', 'logs/smoke_resume/last.pt', '--headless', '--steps', '30')
if torch.cuda.is_available():
    run('courses/soccer-4v4/train.py', '--backend', 'cpu', '--device', 'cpu', '--no-wandb', '--num-envs', '1', '--horizon', '8', '--iterations', '1',
        '--epochs', '1', '--device', 'cuda:0', '--output', 'logs/smoke_cuda', '--job-type', 'smoke')
run('courses/soccer-4v4/play.py', '--image', 'artifacts/scene_4v4.png')
run('courses/soccer-4v4/play.py', '--video', 'artifacts/neutral_4v4.mp4', '--steps', '150')
report = {'passed': True, 'stages': 6, 'resume_iteration': 2,
          'cuda_learner_tested': torch.cuda.is_available(), 'physics': 'native MuJoCo CPU',
          'formal_training_started': False}
(ROOT/'artifacts/smoke_report.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
