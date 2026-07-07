#!/usr/bin/env bash
set -uo pipefail

# 2x B200 100M ablation suite.
# Budget per run: 954 steps x world_size 2 x batch/GPU 256 x seq 2048 = 1.000341504B tokens.
# Total for 7 runs: 7.002390528B tokens.
#
# Usage on a 2x B200 host:
#   PYTHON=python scripts/ablations_100m/run_b200x2_1b_all.sh
#
# Extra CLI args are appended to every run.

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python}"
NPROC="${NPROC_PER_NODE:-2}"

RUNS=(
  100m_zipf_shared
  100m_zipf_no_shared
  100m_modulo_shared
  100m_random_shared
  100m_round_robin_shared
  100m_shared_only
  100m_dense_residual
)

mkdir -p runs/b200x2_100m_ablation_1b

run_one() {
  local name="$1"
  local run_name="b200x2-1b-${name}"
  local metrics="runs/${run_name}/metrics.csv"
  local log="runs/b200x2_100m_ablation_1b/${run_name}.log"

  if [[ -s "${metrics}" ]]; then
    echo "=== ${run_name} already has metrics; skipping ==="
    return 0
  fi

  echo "=== ${run_name} ==="
  set +e
  "${PYTHON_BIN}" -m torch.distributed.run --nproc_per_node="${NPROC}" \
    scripts/train_100m_o200k_tr_local.py \
    --config "configs/run_configs/ablations_100m/${name}.yaml" \
    --steps 954 \
    --batch-size 256 \
    --seq-len 2048 \
    --bf16 \
    --no-grad-ckpt \
    --eval-steps 250 \
    --eval-batches 16 \
    --log-steps 10 \
    --save-steps 0 \
    --save-total-limit 0 \
    --loss-chunk-tokens 1024 \
    --run-name "${run_name}" \
    --save-dir "/tmp/no-checkpoints-${run_name}" \
    "$@" \
    2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e

  rm -rf "/tmp/no-checkpoints-${run_name}" "checkpoints/${run_name}"

  if [[ ${rc} -ne 0 ]]; then
    if [[ -s "${metrics}" ]]; then
      echo "WARNING: ${run_name} exited rc=${rc} after writing metrics; continuing"
      return 0
    fi
    echo "ERROR: ${run_name} failed rc=${rc} and no metrics found; stopping"
    return "${rc}"
  fi
}

for name in "${RUNS[@]}"; do
  run_one "${name}" "$@" || exit $?
done

"${PYTHON_BIN}" - <<'PY'
import csv, math
from pathlib import Path
names = [
  '100m_zipf_shared',
  '100m_zipf_no_shared',
  '100m_modulo_shared',
  '100m_random_shared',
  '100m_round_robin_shared',
  '100m_shared_only',
  '100m_dense_residual',
]
summary_path = Path('runs/b200x2_100m_ablation_1b/loss_summary.csv')
summary_path.parent.mkdir(parents=True, exist_ok=True)
rows=[]
for name in names:
    run=f'b200x2-1b-{name}'
    path=Path('runs')/run/'metrics.csv'
    if not path.exists():
        rows.append({'name': name, 'status': 'missing', 'step': '', 'train_loss': '', 'last_eval_loss': '', 'best_eval_loss': '', 'tok_s': ''})
        continue
    data=list(csv.DictReader(path.open()))
    last=data[-1]
    evals=[]
    for r in data:
        try:
            v=float(r.get('eval_loss','nan'))
        except ValueError:
            continue
        if math.isfinite(v):
            evals.append(v)
    best=min(evals) if evals else float('nan')
    rows.append({
        'name': name,
        'status': 'complete',
        'step': last.get('step',''),
        'train_loss': last.get('loss', last.get('train_loss','')),
        'last_eval_loss': f'{evals[-1]:.6f}' if evals else '',
        'best_eval_loss': f'{best:.6f}' if math.isfinite(best) else '',
        'tok_s': last.get('tok_s',''),
    })
with summary_path.open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=['name','status','step','train_loss','last_eval_loss','best_eval_loss','tok_s'])
    writer.writeheader(); writer.writerows(rows)
print(summary_path)
print(summary_path.read_text())
PY
