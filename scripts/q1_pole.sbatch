#!/bin/bash
#SBATCH --job-name=q1-pole
#SBATCH --partition=gpu_batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/home/mc63588/course_projects/7104/master_project_q1_robot_soccer
#SBATCH --output=/home/mc63588/course_projects/7104/master_project_q1_robot_soccer/logs/slurm/%x-%j.out
#SBATCH --error=/home/mc63588/course_projects/7104/master_project_q1_robot_soccer/logs/slurm/%x-%j.err

# Automatically run GPU preflight -> 1-update smoke test -> full training.
# Any failed command or failed smoke validation stops the job.
# Submit from the repository root (create logs/slurm BEFORE sbatch):
#   mkdir -p logs/slurm
#   sbatch scripts/q1_pole.sbatch
# Override the wall-time request, if needed (gpu_batch maximum: 5 days):
#   sbatch --time=2-00:00:00 scripts/q1_pole.sbatch
# Smoke and full training start independently from the registered warmup.
# Full training uses the repository recipe and its stopping conditions.

set -euo pipefail
umask 027

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Submit this script with sbatch; do not run it with bash on a login node." >&2
    exit 2
fi
if (( $# != 0 )); then
    echo "Usage: sbatch scripts/q1_pole.sbatch (no script arguments)" >&2
    exit 2
fi

project_dir=/home/mc63588/course_projects/7104/master_project_q1_robot_soccer
config_path="$project_dir/courses/pole-weaving/pole_task/config.json"
log_dir="$project_dir/logs/slurm"

# Source Conda directly: batch shells need not load the interactive .bashrc.
source /home/mc63588/miniconda3/etc/profile.d/conda.sh
conda activate env_q1
cd "$project_dir"
mkdir -p "$log_dir"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

printf 'Job ID: %s\nNode: %s\nProject: %s\n' \
    "$SLURM_JOB_ID" "$(hostname)" "$PWD"
python -c 'import sys, torch; print("Python:", sys.executable); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())'
nvidia-smi

# The training entrypoint may exit 0 in preflight mode even without a GPU.
# Check hardware.ready explicitly before starting training.
preflight_report="$log_dir/preflight-$SLURM_JOB_ID.json"
printf '[1/3] GPU preflight. Report: %s\n' "$preflight_report"
srun --input=none python -u -m pole_task.training \
    --preflight-only --config "$config_path" > "$preflight_report"
python - "$preflight_report" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
report = json.loads(path.read_text())
ready = (
    report.get("training_ready", False)
    and report.get("hardware", {}).get("ready", False)
    and report.get("initialization", {}).get("strictly_loaded", False)
)
print("Preflight report:", path)
print("Preflight:", "PASS" if ready else "FAILED")
print("Hardware:", json.dumps(report.get("hardware", {})))
if not ready:
    raise SystemExit("Preflight failed; see the report above. Training was not started.")
PY

# The CLI prints its completed run directory as the last stdout line.
# Verify its identity and saved model tensors before advancing the pipeline.
validate_run() {
    python - "$1" "$2" "$3" "$project_dir" <<'PY'
import json
from pathlib import Path
import sys
import torch

stdout_path, summary_path = map(Path, sys.argv[1:3])
stage = sys.argv[3]
output_root = (Path(sys.argv[4]) / "logs/q1-pole-weaving").resolve()
lines = [line.strip() for line in stdout_path.read_text().splitlines() if line.strip()]
if not lines:
    raise SystemExit(f"{stage}: training produced no output")
run_dir = Path(lines[-1])
if not run_dir.is_absolute():
    raise SystemExit(f"{stage}: final output is not a run directory: {lines[-1]}")
run_dir = run_dir.resolve()
if not run_dir.is_relative_to(output_root) or not run_dir.is_dir():
    raise SystemExit(f"{stage}: unexpected training directory: {run_dir}")
if (run_dir / "failure.json").exists():
    raise SystemExit(f"{stage}: failure.json exists in {run_dir}")
identity = json.loads((run_dir / "identity.json").read_text())
completed = int(identity["completed_updates"])
if (stage == "smoke" and completed != 1) or completed < 1:
    raise SystemExit(f"{stage}: unexpected completed update count: {completed}")
checkpoint = run_dir / "latest.pt"
payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
if int(payload["update"]) != completed:
    raise SystemExit(f"{stage}: checkpoint update does not match identity.json")
for name in ("actor", "model", "critic"):
    for key, tensor in payload[name].items():
        if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
            raise SystemExit(f"{stage}: non-finite tensor {name}.{key}")
summary = {
    "stage": stage,
    "status": "PASS",
    "run_dir": str(run_dir),
    "completed_updates": completed,
    "latest_checkpoint": str(checkpoint),
    "stop_reason": identity.get("stop_reason"),
}
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY
}

smoke_log="$log_dir/smoke-$SLURM_JOB_ID.out"
printf '[2/3] Smoke test: one update. Log: %s\n' "$smoke_log"
srun --input=none python -u -m pole_task.training \
    --config "$config_path" --max-updates 1 | tee "$smoke_log"
validate_run "$smoke_log" "$log_dir/smoke-$SLURM_JOB_ID.json" smoke

full_log="$log_dir/train-$SLURM_JOB_ID.out"
printf '[3/3] Smoke test passed. Starting full training. Log: %s\n' "$full_log"
# Omit --max-updates and --resume-checkpoint: use the full original recipe.
srun --input=none python -u -m pole_task.training \
    --config "$config_path" | tee "$full_log"
validate_run "$full_log" "$log_dir/train-$SLURM_JOB_ID.json" full
printf 'Job %s finished. Check the training summary for its stopping reason.\n' "$SLURM_JOB_ID"
