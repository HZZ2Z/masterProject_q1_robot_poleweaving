"""Paired native-physics motor audit; uncertainty is over reset episodes only."""
import argparse
import json
from pathlib import Path
import mujoco
import torch
from train import evaluate
from q1_supermimic.physics import SCENE
from q1_supermimic.networks import MotorPolicy
from q1_supermimic.observations import MotorObservation
from q1_supermimic.statistics import paired_difference, wilson_interval


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--episodes',type=int,default=128)
    parser.add_argument('--seed',type=int,default=224173)
    parser.add_argument('--output',type=Path,default=Path('logs/motor_paired_audit.json'))
    args=parser.parse_args()
    if args.episodes<2: parser.error('Need at least two paired episodes')
    torch.set_num_threads(2)
    model=mujoco.MjModel.from_xml_path(str(SCENE));observe=MotorObservation(model)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    width=saved.get('config',{}).get('policy_width',128)
    policy=MotorPolicy(720,540,4,22,motor_widths=(width,width),visual_widths=(64,64))
    torch.nn.init.zeros_(policy.decoder[-1].weight);torch.nn.init.zeros_(policy.decoder[-1].bias)
    baseline=evaluate(model,policy,observe,args.seed,args.episodes,details=True)
    print('Baseline evaluation complete',flush=True)
    policy.load_state_dict(saved.get('accepted_policy',saved['policy']))
    candidate=evaluate(model,policy,observe,args.seed,args.episodes,details=True)
    result=dict(checkpoint=str(args.checkpoint),seed=args.seed,
                scope='5s neutral standing; only joint reset noise sigma=0.005; no visual or soccer evaluation',
                comparison=paired_difference(baseline['episode_returns'],candidate['episode_returns']),
                baseline=baseline,candidate=candidate)
    for metrics in (baseline,candidate):
        metrics['fall_rate_wilson_95_interval']=wilson_interval(sum(metrics['episode_falls']),args.episodes)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('baseline','candidate')}),flush=True)


if __name__=='__main__': main()
