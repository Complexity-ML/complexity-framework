#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

RUNS=(
  100m_zipf_shared
  100m_random_shared
  100m_dense_residual
)

mkdir -p runs/local_100m_ablation_200

for name in "${RUNS[@]}"; do
  run_name="local-200-fixedtopk05-${name}"
  echo "=== ${run_name} ==="
  python3.13 scripts/train_100m_o200k_tr_local.py \
    --config "configs/run_configs/ablations_100m/${name}.yaml" \
    --dataset text \
    --text-file data/local/fineweb_sample.txt \
    --steps 200 \
    --batch-size 4 \
    --seq-len 256 \
    --eval-steps 50 \
    --eval-batches 2 \
    --log-steps 10 \
    --save-steps 0 \
    --run-name "${run_name}" \
    --save-dir "checkpoints/${run_name}" \
    --no-grad-ckpt \
    --loss-chunk-tokens 512 \
    --top-k-primary-weight 0.5 \
    --top-k-primary-weight-final 0.5 \
    --use-custom-kernels false \
    --cggr false \
    2>&1 | tee "runs/local_100m_ablation_200/${run_name}.log"
done

python3.13 - <<'PY'
import csv, math
from pathlib import Path
names = ['100m_zipf_shared', '100m_random_shared', '100m_dense_residual']
print('name,step,train_loss,last_eval_loss,best_eval_loss,tok_s')
for name in names:
    run=f'local-200-fixedtopk05-{name}'
    path=Path('runs')/run/'metrics.csv'
    if not path.exists():
        print(f'{name},MISSING,,,,')
        continue
    rows=list(csv.DictReader(path.open()))
    last=rows[-1]
    evals=[]
    for r in rows:
        try:
            v=float(r.get('eval_loss','nan'))
        except ValueError:
            continue
        if math.isfinite(v):
            evals.append(v)
    best=min(evals) if evals else float('nan')
    print(f"{name},{last['step']},{last['train_loss']},{last['eval_loss']},{best:.6f},{last.get('tok_s','')}")
PY
