from __future__ import annotations

import pytest


def test_cluster_plan_8b_32t_gb300_math():
    from complexity.training.cluster_plan import load_cluster_run_plan

    plan = load_cluster_run_plan("configs/run_configs/8b_o200k_tr_32t_gb300_4608.yaml")

    assert plan.parallel.tp_size == 8
    assert plan.parallel.pp_size == 8
    assert plan.parallel.dp_size == 72
    assert plan.parallel.world_size == 4608
    assert plan.parallel.model_replica_gpus == 64
    assert plan.parallel.batch_per_dp_replica == 128
    assert plan.parallel.global_batch == 9216
    assert plan.params == 8_201_527_360
    assert plan.tokens_per_step == 18_874_368
    assert plan.steps == 1_738_131
    assert plan.chinchilla_multiple == pytest.approx(200.0)


def test_cluster_plan_rejects_world_size_mismatch(tmp_path):
    from complexity.training.cluster_plan import load_cluster_run_plan

    path = tmp_path / "bad.yaml"
    path.write_text(
        """
world_size: 16
model:
  params: 8B
parallel:
  tp_size: 2
  pp_size: 2
  dp_size: 2
  micro_batch_size: 1
  gradient_accumulation: 1
run:
  run_name: bad
  target_tokens: 1B
  seq_len: 2048
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="world_size mismatch"):
        load_cluster_run_plan(path)


def test_cluster_config_blocks_experimental_pipeline_by_default():
    from complexity.parallel.cluster import ClusterConfig

    cfg = ClusterConfig(tp_size=1, pp_size=2, dp_size=1)
    with pytest.raises(NotImplementedError, match="Pipeline parallelism"):
        cfg.validate()
