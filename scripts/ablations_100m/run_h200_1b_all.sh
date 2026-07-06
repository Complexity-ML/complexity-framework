#!/usr/bin/env bash
set -euo pipefail

# 1x H200 100M ablation suite.
# Budget per run: 1908 steps x batch 256 x seq 2048 = 1.000341504B tokens.
# Total for 7 runs: 7.002390528B tokens.
#
# Rationale: 4B/run is expensive for ablation iteration, and batch 512 OOMs on
# a 143GB H200 for this model. Batch 256 has been smoke-tested on H200.
#
# Usage:
#   scripts/ablations_100m/run_h200_1b_all.sh
#   scripts/ablations_100m/run_h200_1b_all.sh --dataset text --text-file data/local/fineweb_sample.txt
#
# Extra CLI args are appended to every run, so you can override dataset/checkpointing/etc.

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON:-python}"

RUNS=(
  100m_zipf_shared
  100m_zipf_no_shared
  100m_modulo_shared
  100m_random_shared
  100m_round_robin_shared
  100m_shared_only
  100m_dense_residual
)

mkdir -p runs/h200_100m_ablation_1b

for name in "${RUNS[@]}"; do
  run_name="h200-1b-${name}"
  echo "=== ${run_name} ==="
  "${PYTHON_BIN}" scripts/train_100m_o200k_tr_local.py \
    --config "configs/run_configs/ablations_100m/${name}.yaml" \
    --steps 1908 \
    --batch-size 256 \
    --seq-len 2048 \
    --bf16 \
    --no-grad-ckpt \
    --eval-steps 250 \
    --eval-batches 16 \
    --log-steps 10 \
    --save-steps 1908 \
    --save-total-limit 2 \
    --loss-chunk-tokens 1024 \
    --run-name "${run_name}" \
    --save-dir "checkpoints/${run_name}" \
    "$@" \
    2>&1 | tee "runs/h200_100m_ablation_1b/${run_name}.log"
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
summary_path = Path('runs/h200_100m_ablation_1b/summary.csv')
summary_path.parent.mkdir(parents=True, exist_ok=True)
rows=[]
for name in names:
    run=f'h200-1b-{name}'
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
        'status': 'ok',
        'step': last.get('step',''),
        'train_loss': last.get('train_loss',''),
        'last_eval_loss': last.get('eval_loss',''),
        'best_eval_loss': f'{best:.6f}' if math.isfinite(best) else 'nan',
        'tok_s': last.get('tok_s',''),
    })
with summary_path.open('w', newline='') as f:
    writer=csv.DictWriter(f, fieldnames=['name','status','step','train_loss','last_eval_loss','best_eval_loss','tok_s'])
    writer.writeheader(); writer.writerows(rows)
print(summary_path)
print('name,status,step,train_loss,last_eval_loss,best_eval_loss,tok_s')
for r in rows:
    print(','.join(str(r[k]) for k in ['name','status','step','train_loss','last_eval_loss','best_eval_loss','tok_s']))
PY
