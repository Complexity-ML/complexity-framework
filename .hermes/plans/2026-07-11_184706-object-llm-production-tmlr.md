# Object LLM Production and TMLR Plan

> **For Hermes:** Execute task-by-task with strict TDD. Do not commit or push unless the user asks.

**Goal:** Turn the 98.2M attention-free causal-convolution + lexical-object + micro-expert prototype into a production-capable CUDA Graph inference/training framework, run controlled 1B-token NVIDIA experiments, and prepare an anonymous TMLR submission grounded only in verified artifacts.

**Architecture:** The canonical candidate uses a shared dilated causal convolutional sequence mixer, a tied o200k lexical object table, and narrow deterministic micro-expert residuals. QKV/GQA remains only as a matched scientific baseline. Production inference separates parallel prefill from fixed-state single-token decode and treats CUDA Graph compatibility as an explicit contract.

**Tech stack:** PyTorch 2.13, CUDA, CUDA Graphs, BF16, optional FP8 after BF16 validation, torch.distributed, FineWeb, o200k tokenizer, pytest, CSV metrics, LaTeX/TMLR/OpenReview.

---

## Current evidence and limits

- Canonical M5 diagnostic: 98,194,244 parameters, no QKV, 200 steps / 204,800 tokens, eval loss 7.557967, last throughput 6,145 tok/s.
- Matched saved dense Transformer diagnostic: eval loss 7.944988, last throughput 4,779 tok/s.
- M5 sequence-1024 matched comparison: convolution beats the persistent-state variant in loss and throughput.
- Cached convolutional decode matches full forward at `atol=rtol=1e-5`; state pointers remain stable.
- These are diagnostics, not paper-level claims. No CUDA capture has yet been executed. No independent 1B-token corpus has yet been wired into this worktree.
- Verda public prices observed 2026-07-11: H100 spot $1.14/h, H200 spot $1.40/h, B200 spot $2.14/h. Provisioning requires explicit user approval after smoke-derived ETA.

## Scientific contract

Primary matched-token variants:

1. Dense GQA + dense SwiGLU baseline.
2. Causal convolution + dense SwiGLU.
3. Causal convolution + tied lexical objects, no micro-experts.
4. Causal convolution + tied lexical objects + 4×32 micro-experts.

Hold fixed: tokenizer, training corpus and held-out split, optimizer, LR schedule, total tokens, seed, sequence length, effective batch tokens, precision, evaluation examples, and parameter budget within 0.1%. Report throughput separately; do not conflate matched-token quality with wall-clock efficiency.

---

### Task 1: Canonical architecture guardrails

**Objective:** Make the non-QKV production profile explicit and fail loud if QKV or unsupported state paths leak into it.

**Files:**
- Modify: `complexity/config/model_config.py`
- Modify: `complexity/models/builder.py`
- Test: `tests/test_causal_conv_model.py`

**Steps:**
1. Add failing tests asserting the canonical profile instantiates no `q_proj`, `k_proj`, `v_proj`, rotary embedding, or attention cache.
2. Add a config-level `architecture_family="object_conv"` (or equivalent minimal explicit discriminator) and validation for its allowed mixer/MLP pair.
3. Make model construction fail if the production profile resolves to GQA through parser defaults.
4. Verify with `pytest tests/test_causal_conv_model.py -q`.

### Task 2: Production prefill/decode API

**Objective:** Provide a typed API independent of the historical KV-centric inference engine.

**Files:**
- Create: `complexity/inference/object_engine.py`
- Modify: `complexity/inference/__init__.py`
- Modify: `complexity/models/builder.py`
- Test: `tests/test_object_inference.py`

**Steps:**
1. RED: specify `ObjectInferenceState`, `prefill(input_ids)`, and `decode_step(token_ids, state)` behavior.
2. Require prefill output logits `[B,1,V]`, per-layer fixed convolution states, and stable state pointers across decode calls.
3. Implement minimal wrappers around the already-verified model cache path.
4. Test full-forward versus prefill plus incremental decode over multiple prompt lengths and batches.
5. Test invalid batch-size/state-shape errors.
6. Run targeted tests, then `pytest tests/test_inference.py tests/test_object_inference.py tests/test_causal_conv_model.py -q`.

### Task 3: Static CUDA Graph decode runner

**Objective:** Capture and replay fixed-batch single-token decode on CUDA.

**Files:**
- Create: `complexity/inference/cuda_graph.py`
- Create: `scripts/verify_object_cuda_graph.py`
- Test: `tests/test_object_cuda_graph.py`

**Steps:**
1. RED on CUDA-capable hosts; skip with an explicit reason elsewhere.
2. Preallocate static token input, output logits, and per-layer states.
3. Warm up on a side stream.
4. Capture decode under `torch.cuda.graph` without `.item()`, host-dependent branches, or dynamic allocation.
5. Replay with changed token values copied into the static input.
6. Compare eager cached decode and graph replay logits and state over at least 32 steps.
7. Report eager tok/s, graph tok/s, latency p50/p95, allocated memory, and state bytes.
8. Ensure the script exits non-zero on capture failure or numeric mismatch.

### Task 4: Production generation loop

**Objective:** Build a minimal robust generator on top of prefill/decode without inheriting KV cache assumptions.

**Files:**
- Modify: `complexity/inference/object_engine.py`
- Modify: `complexity/api/inference.py`
- Test: `tests/test_object_inference.py`

**Steps:**
1. RED for greedy generation equality against `ComplexityModel.generate` on a tiny deterministic model.
2. Add sampling outside the captured model graph initially; keep graph scope to logits/state transition.
3. Support fixed batch, EOS masking, max tokens, temperature, top-k, and seeded sampling.
4. Do not add continuous batching or paged KV abstractions yet.
5. Test no cache reallocation and bounded state memory for generation longer than the receptive field.

### Task 5: Export and deployment boundary

**Objective:** Determine and document the supported production artifact instead of claiming unsupported ONNX compatibility.

**Files:**
- Create: `docs/object_model_deployment.md`
- Create: `scripts/export_object_prefill.py` only if `torch.export` succeeds
- Test: `tests/test_object_export.py`

**Steps:**
1. Test `torch.export` separately for prefill and decode wrappers.
2. Test ONNX only as an experiment; do not make it a requirement if dynamic tied embeddings/einsum/gather fail.
3. Record the canonical deployment as PyTorch CUDA Graph unless another path is actually verified.
4. Include exact supported shapes, dtypes, batch buckets, and state schema.

### Task 6: FineWeb 1B data pipeline

**Objective:** Replace the repeated 1.095M-token local diagnostic file with a reproducible, non-overlapping 1B-token corpus and held-out evaluation stream.

**Files:**
- Inspect/modify: `complexity/training/o200k/data.py`
- Create: `configs/data/fineweb_o200k_1b.yaml`
- Create: `scripts/prepare_fineweb_o200k_1b.py`
- Test: `tests/test_fineweb_1b_pipeline.py`

**Steps:**
1. RED for deterministic sharding, exact token accounting, document-boundary handling, and disjoint train/eval IDs.
2. Stream or materialize token shards with manifest, checksums, source revision, tokenizer provenance, and exact counts.
3. Target at least 1.05B available train tokens so the 1B run does not wrap.
4. Reserve a held-out validation split by source document identity, not by a second iterator over training data.
5. Add fail-loud checks preventing the local sample file from being used for a headline run.

### Task 7: Matched 100M experiment configs

**Objective:** Create the four reviewer-proof 1B-token conditions.

**Files:**
- Create: `configs/run_configs/object_1b/100m_dense_gqa.yaml`
- Create: `configs/run_configs/object_1b/100m_conv_dense.yaml`
- Create: `configs/run_configs/object_1b/100m_conv_lexical.yaml`
- Create: `configs/run_configs/object_1b/100m_conv_lexical_micro.yaml`
- Test: `tests/test_object_1b_configs.py`

**Steps:**
1. RED for production parser equivalence and realized model components.
2. Assert no-QKV conditions contain no QKV parameters.
3. Assert parameter counts within 0.1%, exact tokenizer vocab 200,019, exact token budget, seed, dataset manifest, and no checkpointing.
4. Use the same sequence length and effective batch tokens across variants unless a documented memory constraint forces gradient accumulation.
5. Dense baseline runs first on paid hardware.

### Task 8: Resilient H100/H200/B200 launcher

**Objective:** Smoke, estimate, and run without wasting paid compute.

**Files:**
- Create: `scripts/object_1b/run_verda_1b.sh`
- Create: `scripts/object_1b/summarize_object_1b.py`
- Test: `tests/test_object_1b_launcher.py`

**Steps:**
1. Detect GPU model, count, VRAM, CUDA/PyTorch versions, and available disk.
2. Run exact-shape 20-step smoke for the dense baseline first.
3. Abort on NaN/Inf, OOM, parser mismatch, missing dataset manifest, or CUDA Graph mismatch.
4. Calculate measured ETA and estimated cost from user-provided hourly rate.
5. Require an environment confirmation flag before crossing the smoke boundary.
6. Disable checkpoints; mirror `metrics.csv`, logs, run config, environment manifest, and summary after every run.
7. Treat a non-zero teardown as successful only if final-step metrics exist.

### Task 9: Verda provisioning

**Objective:** Provision only after explicit financial confirmation.

**Steps:**
1. User logs into Verda and configures payment/API access.
2. Prefer one H200 spot for the first smoke because public price is $1.40/h and 141GB should fit the existing 100M batch profile.
3. If spot interruption risk is unacceptable, present H200 on-demand and B200 spot alternatives with smoke-derived ETA.
4. Show GPU, region, storage, hourly rate, estimated run hours, estimated maximum cost, and shutdown policy.
5. Ask for explicit approval.
6. Provision, verify SSH, clone/copy the isolated worktree snapshot, install CUDA dependencies, and run smoke only.
7. Stop and delete the instance immediately on user instruction or failed invariants.

### Task 10: Evaluation suite

**Objective:** Produce evidence beyond local next-token loss.

**Files:**
- Create: `scripts/object_1b/evaluate_object_1b.py`
- Create: `configs/eval/object_100m.yaml`

**Steps:**
1. Evaluate held-out FineWeb loss at common checkpoints/token budgets.
2. Add compact public LM evaluations appropriate for 100M scale; avoid frontier comparisons.
3. Add long-range controlled tests that distinguish a 775-token receptive field from genuine longer memory.
4. Measure prefill and decode separately at batch 1 and serving-relevant fixed batches.
5. Record raw outputs and confidence intervals; do not report only plots.

### Task 11: Results ledger and paper skeleton

**Objective:** Make every paper claim traceable to an artifact before prose hardens.

**Files:**
- Create: `paper/claims.csv`
- Create: `paper/main.tex`
- Create: `paper/sections/method.tex`
- Create: `paper/sections/experiments.tex`
- Create: `paper/sections/limitations.tex`
- Create: `paper/references.bib`

**Steps:**
1. Build a claim-to-evidence ledger with metric file, config hash, dataset manifest, hardware, and script path.
2. Define equations for shared causal mixer, tied lexical object residual, and micro-expert routing exactly as implemented.
3. Describe QKV removal narrowly; do not claim universal replacement.
4. Frame M5 runs as diagnostics and NVIDIA 1B runs as headline evidence only if completed.
5. Include failed persistent-state ablation and numerical-stability lesson only if scientifically useful and supported by raw artifacts.
6. State limitations: single scale, limited seeds, fixed convolutional receptive field, provider-specific throughput, unverified general scaling if applicable.

### Task 12: Closest-prior-work review

**Objective:** Establish defensible novelty relative to causal CNNs, fixed/hash routing, shared-expert MoE, Mamba/SSMs, RWKV/retention, Hyena, and attention-free LMs.

**Steps:**
1. Search exact mechanisms, not only broad families.
2. Separate known components from the tested integration and deployment contract.
3. Avoid novelty claims for causal convolutions, deterministic routing, or shared residuals individually.
4. Position contribution as measured integration only if ablations support it.

### Task 13: Anonymous TMLR supplement

**Objective:** Package the exact verified harness without identity leaks.

**Files:**
- Create staging tree: `dist/tmlr_supplement/`
- Create: `dist/tmlr_supplement/README.md`

**Steps:**
1. Include only canonical model code, four configs, launcher, tokenizer stub/provenance, dataset manifest schema, tests, raw headline CSVs, and reproduction instructions.
2. Exclude `.git`, checkpoints, local samples, credentials, cloud IPs, caches, unrelated legacy profiles, and failed transient logs unless intentionally documented.
3. Replace `/Users/boris`, organization names, emails, repository URLs, and cloud identifiers.
4. Build ZIP, extract to scratch, scan identity strings and forbidden artifacts, inspect PDF metadata, and verify archive size.
5. Keep paper PDF separate from supplement ZIP unless TMLR explicitly asks for a combined artifact.

### Task 14: Final gates

**Objective:** Prevent premature submission or compute claims.

**Verification:**
- All targeted and full tests pass.
- `git diff --check` passes.
- Four configs instantiate intended architectures and matched budgets.
- Headline CSVs reach final step and contain finite metrics.
- CUDA Graph capture and replay pass on target CUDA hardware.
- Eager and graphed decode agree numerically.
- Dataset train/eval manifests are disjoint.
- Paper tables regenerate from raw CSVs.
- Every claim has evidence in `paper/claims.csv`.
- Anonymous ZIP extraction scan is clean.
- No commit/push/submission without user instruction.

## Risks and decisions

- **CUDA Graph:** graph-friendly design is not proof; capture must pass on target GPU.
- **Dataset:** current local FineWeb sample is invalid for 1B training.
- **Throughput:** M5 and NVIDIA numbers are not directly comparable; report per-device.
- **Spot instances:** cheaper but interruptible; preserve metrics frequently and make runs resumable only if checkpoints become necessary.
- **Paper strength:** one 1B seed per condition may support an exploratory architecture paper but not strong universal claims. Prefer a staged 200M multi-seed screen plus 1B finalists if budget is constrained.
- **Inference:** fixed receptive field is a real limitation. Do not re-add persistent state unless long-context evidence beats the simpler convolutional model.
- **Terminology:** rename historical `TransformerBlock/self_attn/past_key_values` interfaces in the production surface, while retaining compatibility internally until tests cover migration.
