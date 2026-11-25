#!/bin/bash
#SBATCH --job-name=xmad-p3
#SBATCH --partition=<partition>
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=p3_%j.out
# ---------------------------------------------------------------------------
# xmad/3_iteration/run_phase3.sbatch
#
# Runs `phase3.py run` as a batch job, so it does not depend on your SSH
# session. Submit from the 3_iteration folder:
#
#   cd /path/to/workspace/xmad/3_iteration
#   sbatch run_phase3.sbatch                          # must-runs
#   sbatch --export=ALL,PRIORITY=should run_phase3.sbatch   # must + should
#   sbatch --export=ALL,MODE=pretest run_phase3.sbatch      # pre-test only: smoke + all
#        homogeneous pairs with 30 questions (folders *_test) + sanity table; no re-submit
#   sbatch --export=ALL,MODE=judge,PRIORITY=should run_phase3.sbatch  # retry failed runs, then judge pass
#
# Watch:   squeue -u $USER        tail -f p3_<jobid>.out        cat results/OVERVIEW.md
#
# Time limit / preemption: 5 min before the limit (or on preemption) SLURM sends
# SIGTERM; phase3.py then stops all model servers and marks unfinished runs as
# "interrupted". If runs remain, the job submits itself again and continues
# where it stopped (max. MAX_RESUBMIT times in a row).
#
# Stop for good:   touch STOP   (prevents the re-submission)
#                  scancel --signal=TERM --batch <jobid>   (graceful stop now)
# Continue later:  rm STOP && sbatch run_phase3.sbatch
# ---------------------------------------------------------------------------
set -u

PRIORITY=${PRIORITY:-must}
MODE=${MODE:-run}
PROJECT=/path/to/workspace/xmad/3_iteration
VENV=/path/to/workspace/venvs/vllm-p3
MAX_RESUBMIT=20

cd "$PROJECT"
source "$VENV/bin/activate"
export HF_HOME=/path/to/workspace/hf_cache
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1

echo "[sbatch] job ${SLURM_JOB_ID:-?} on $(hostname), priority=$PRIORITY, $(date -Iseconds)"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader

# models of this priority (and the judge) without a smoke PASS yet are tested first
python smoke_test.py --priority "$PRIORITY" --missing --gpus 0,1,2,3,4,5,6,7 --port 9100 2>&1 | tee -a smoke.log

if [ "$MODE" = "pretest" ]; then
  python phase3.py run --priority "$PRIORITY" --homogeneous --n 30
  python phase3.py sanity --test | tee pretest_sanity.txt
  echo "[sbatch] pre-test finished -- see pretest_sanity.txt"
  exit 0
fi

if [ "$MODE" = "judge" ]; then
  # finish failed/open runs of this priority first, then the judge pass over all finished runs
  python phase3.py run --priority "$PRIORITY" --retry-failed
  python phase3.py judge --gpus 0,1,2,3
  python phase3.py status
  echo "[sbatch] judge pass finished"
  exit 0
fi

python phase3.py run --priority "$PRIORITY" &
PID=$!
trap 'echo "[sbatch] SIGTERM received -> stopping phase3 gracefully"; kill -TERM $PID' TERM
wait $PID
wait $PID 2>/dev/null      # second wait: the first one returns early when the trap fires

REMAIN=$(python phase3.py remaining --priority "$PRIORITY")
python phase3.py status
echo "[sbatch] $REMAIN runs remaining ($(date -Iseconds))"

COUNT_FILE=.resubmit_count
if [ "$REMAIN" -gt 0 ] && [ ! -f STOP ]; then
  COUNT=$(cat "$COUNT_FILE" 2>/dev/null || echo 0)
  if [ "$COUNT" -lt "$MAX_RESUBMIT" ]; then
    echo $((COUNT + 1)) > "$COUNT_FILE"
    echo "[sbatch] re-submitting ($((COUNT + 1))/$MAX_RESUBMIT)"
    sbatch --export=ALL,PRIORITY="$PRIORITY" "$PROJECT/run_phase3.sbatch"
  else
    echo "[sbatch] re-submit limit reached -- check OVERVIEW.md, then: rm $COUNT_FILE && sbatch run_phase3.sbatch"
  fi
else
  rm -f "$COUNT_FILE"
  echo "[sbatch] finished (no runs remaining or STOP file present)"
fi
