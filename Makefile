# Backend-aware install targets.
#
# torch isn't in pyproject extras because pip can't pin an --index-url per
# extra (see pyproject.toml). scripts/install_backend.sh does the right
# thing per backend; these targets are thin aliases.

.PHONY: install install-rocm install-rocm6.4 install-rocm7.0 install-cuda install-cpu install-mps \
	install-vision-rocm install-vision-rocm6.4 install-vision-rocm7.0 \
	install-vision-cuda install-vision-cpu lint test test-core

PYTHON ?= python3
GLOO_SOCKET_IFNAME ?= $(shell if [ "$$(uname -s)" = Darwin ]; then echo lo0; else echo lo; fi)

lint:
	$(PYTHON) -m ruff check complexity --no-respect-gitignore

test:
	GLOO_SOCKET_IFNAME=$(GLOO_SOCKET_IFNAME) OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q

# Focused local contract gate. `make test` also checks historical pipelines.
test-core:
	GLOO_SOCKET_IFNAME=$(GLOO_SOCKET_IFNAME) OMP_NUM_THREADS=1 $(PYTHON) -m pytest -q \
		tests/test_models.py tests/test_tr_hash_model_integration.py \
		tests/test_training.py tests/test_training_boundaries.py \
		tests/test_pretrain_1b_ddp.py tests/test_pretrain_1b_fsdp_flow_control.py \
		tests/test_public_api_workflow.py tests/test_model_config_loading.py \
		tests/test_model_export_integrity.py \
		tests/test_checkpointing_controls.py \
		tests/test_pipeline_training_contract.py \
		tests/test_run_config.py tests/test_cli_workflows.py tests/test_vision_namespace.py \
		tests/test_cli_import_boundaries.py \
		tests/test_tr_hash_transformers_adapter.py tests/test_tr_hash_agentic_tokenizer.py \
		tests/test_lora.py tests/test_chat_template.py \
		tests/test_corpus_mixture.py tests/test_build_tr_hash_70b_replay_plan.py \
		tests/test_agentic_pretraining_50b.py tests/test_agentic_pretraining_corpus.py \
		tests/test_tokenize_agentic_pretraining_corpus.py \
		tests/test_pretraining_125b.py tests/test_pretraining_125b_candidates.py \
		tests/test_build_corrective_replay_plan.py \
		tests/test_tr_hash_text_lineage.py tests/test_tr_hash_agentic_100m_recipe.py \
		tests/test_agentic_100m_refinement_lr.py tests/test_tr_hash_200m_pretraining.py \
		tests/test_metrics_csv_to_tensorboard.py tests/test_tensorboard_metrics.py \
		tests/test_collect_200m_run_artifacts.py \
		tests/test_sft_preparation.py tests/test_compile_tr_hash_agentic_sft_bin.py \
		tests/test_sft_bin.py tests/test_sft_support_contracts.py \
		tests/test_prepare_tr_hash_agentic_sft.py tests/test_prepare_tr_hash_agentic_sft_250k.py \
		tests/test_prepare_tr_hash_agentic_sft_500k.py tests/test_prepare_tr_hash_agentic_sft_2m.py \
		tests/test_sft_curriculum_runner.py tests/test_sft_liger_dispatch.py tests/test_fused_ce.py \
		tests/test_training_runner_device_requirements.py tests/test_training_update_logging.py \
		tests/test_watch_and_restart_stalled_process.py tests/test_supervisor.py \
		tests/test_eval_mlx_zero_shot.py tests/test_eval_torch_arc_zero_shot.py \
		tests/test_choice_evaluation_workflow.py tests/test_generative_evaluation_workflow.py \
		tests/test_evaluator_label_alignment.py tests/test_sft_workflow.py tests/test_stack_sft.py \
		tests/test_sft_distributed.py tests/test_tr_hash_engine.py \
		tests/test_viz_pretrained_expert_tsne_3d.py \
		tests/test_vision_dependency_stack.py tests/test_documentation_contract.py

install:  ## Detect the backend for this machine
	./scripts/install_backend.sh auto

install-rocm:
	./scripts/install_backend.sh rocm

install-rocm6.4:
	./scripts/install_backend.sh rocm6.4

install-rocm7.0:
	./scripts/install_backend.sh rocm7.0

install-cuda:
	./scripts/install_backend.sh cuda

install-mps:
	./scripts/install_backend.sh mps

install-cpu:
	./scripts/install_backend.sh cpu

install-vision-rocm:
	./scripts/install_backend.sh rocm vision

install-vision-rocm6.4:
	./scripts/install_backend.sh rocm6.4 vision

install-vision-rocm7.0:
	./scripts/install_backend.sh rocm7.0 vision

install-vision-cuda:
	./scripts/install_backend.sh cuda vision

install-vision-cpu:
	./scripts/install_backend.sh cpu vision
