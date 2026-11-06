"""Exercise a forced curriculum transition in an isolated, non-W&B smoke job.

The gate override tests scene rebuilding and checkpointing; it is not evidence
that the policy has mastered a course. Never resume formal training from its output.
Launch with torchrun exactly like train_distributed.py.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
# Import before Torch: this entrypoint isolates the worker's visible physical GPU.
from train_distributed import parser,run
from q1_4v4.parallel_curriculum import ParallelCurriculum


def main():
    p=parser()
    p.add_argument('--certify-course',required=True)
    options=p.parse_args()
    if options.job_type!='smoke' or options.wandb or options.iterations!=1:
        raise ValueError('Transition smoke requires --job-type smoke --no-wandb --iterations 1')
    if not options.checkpoint or options.curriculum_mode!='parallel':
        raise ValueError('Transition smoke requires a checkpoint and --curriculum-mode parallel')
    if (Path(options.output)/'last.pt').exists():
        raise ValueError('Use a fresh isolated output directory for the smoke test')
    def advance(curriculum):
        before=set(curriculum.ready)
        if options.certify_course not in before:
            raise ValueError('Requested test course must be ready and unmastered')
        curriculum.mastered.add(options.certify_course)
        for name in set(curriculum.ready)-before:
            curriculum.courses[name].recertify('smoke_only_forced_transition')
        return True
    ParallelCurriculum.advance=advance
    run(options)


if __name__=='__main__':main()
