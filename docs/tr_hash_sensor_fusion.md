# TR-Hash Robot Perception

> **Experimental perception system.** This position-routed sensor model is
> separate from the token-ID-routed 200M language-model release.

`TRHashSensorFusionClassifier` is a multimodal perception model: sensors ->
encoders -> TR-Hash fusion -> fused state -> classification head. It predicts
one of `num_classes` states from any available combination of depth, infrared,
thermal, IMU, mmWave radar, and skeleton streams.

For general use, construct it through the public API instead of importing the
class directly -- see [`docs/multimodal.md`](multimodal.md) for the full
`Robot` factory reference:

```python
from complexity.api import Robot

model = Robot.model(num_classes=40, num_experts=8, top_k=2)
```

This is the perception front-end of a robot stack, not a control policy: it
has no actuator, trajectory, or value head yet. `pooled_features` in its
output is the fused state a downstream policy/trajectory/value head would
consume.

## Architecture

- three visual streams (depth, IR, thermal) share one TR-Hash MoE vision
  tower (`HierarchicalTRHashVisionTower`), plus a frame-difference motion
  encoder and a per-modality residual adapter;
- IMU and skeleton use structured graph tokenizers (`IMUDeviceGraphTokenizer`,
  `SkeletonGraphTokenizer`); radar uses a generic dilated-temporal tokenizer;
- a fusion sequence (one CLS token plus a fixed token budget per modality)
  with learned modality and local-position embeddings;
- Transformer blocks using eight stored TR-Hash experts and top-2 routing,
  routing keyed by `(modality, local token position)`, never by content;
- a confidence-gated late-fusion path: a fixed TR-Hash route scores every
  `(modality, class)` pair and blends per-modality logits with the shared
  fused head;
- a residual class-hash head adding class-specialized capacity without
  replacing the shared classification head;
- a gradient-reversal subject-adversarial head for domain-invariance training;
- explicit sample-level modality masks for missing sensors.

The default configuration has about 11.6M parameters. There is no reduced or
legacy variant -- every capability above is always active.

For a promoted non-Vision lineage, training follows the same three boundaries
as the language stack: general sensor pretraining, a fresh-optimizer pass over
the exact same corpus for refinement, then labeled downstream SFT. The CUHK-X
command below remains an experimental task-training recipe; it is not evidence
that a complete three-stage release lineage was run.

## Input contract

```python
from complexity.generative.sensor_fusion import (
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)

model = TRHashSensorFusionClassifier(TRHashSensorFusionConfig())
output = model(
    {
        "depth": depth,        # [B, 3, T, H, W]
        "ir": ir,              # [B, 1, T, H, W]
        "thermal": thermal,    # [B, 3, T, H, W]
        "imu": imu,            # [B, T, 45]
        "radar": radar,        # [B, T, 16]
        "skeleton": skeleton,  # [B, T, 17, 3]
    },
    labels=labels,
)
```

IMU expects five devices with nine motion values each, radar sixteen
per-frame statistics, and skeleton seventeen three-dimensional joints (the
structured tokenizers are keyed to that layout). Visual resolution and
sequence duration may vary; each tokenizer adaptively emits a fixed number of
tokens.

## CUHK-X: dataset, training, and submission pipeline

The model was originally developed and benchmarked against the CUHK-X Small
Model Track (a HAR competition). The dataset loader, cross-subject fold
splitting, and Kaggle-style submission pipeline documented below remain
dataset-specific and are useful any time CUHK-X data is the training source,
independent of the competition itself.

### Local dataset and cross-subject split

Extract the licensed multi-volume `HAR.zip` archive locally. Point the trainer
at either the directory containing `HAR/data` or directly at `HAR/data`. The
first process builds a lightweight JSON manifest so the other DDP ranks do not
rescan hundreds of thousands of files.

The default validation people are users `8`, `9`, `23`, and `24`. They never
appear in the training split. Override them explicitly with
`--validation-users` to reproduce the original competition protocol.

### Eight-GPU training

```bash
torchrun --standalone --nproc_per_node=8 \
  -m complexity.generative.sensor_fusion.training \
  --data-root /workspace/datasets/CUHK-X/extracted \
  --output artifacts/tr_hash_robot_perception \
  --optimizer musgd \
  --epochs 50 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --workers 4 \
  --precision bf16 \
  --require-fused-cuda
```

`--batch-size` is per GPU. The trainer uses MuSGD, a separate learning-rate
group for routed expert parameters, inverse-square-root class weighting,
distributed non-padding validation, top-1/top-5/macro accuracy, and safetensors
model weights. It saves optimizer, scheduler, epoch, exact batch cursor, and one
RNG state per rank. `scripts/vast_train_cuhkx_sensor_fusion.sh` wraps this with
sensible defaults.

Resume with the same architecture, preprocessing, world size, batch size, and
total epoch target:

```bash
torchrun --standalone --nproc_per_node=8 \
  -m complexity.generative.sensor_fusion.training \
  --data-root /workspace/datasets/CUHK-X/extracted \
  --output artifacts/tr_hash_robot_perception \
  --resume artifacts/tr_hash_robot_perception/step_0000500 \
  --optimizer musgd \
  --epochs 50 \
  --batch-size 2 \
  --eval-batch-size 2 \
  --workers 4 \
  --precision bf16 \
  --require-fused-cuda
```

Training updates all parameters; LoRA is intentionally not used since the
visual, inertial, radar, skeleton, fusion, and classification layers all need
to learn the task jointly from random init.

An optional pretrained TR-Hash vision tower can be transferred into a fresh
run with `--vision-backbone-checkpoint`.

### Kaggle-style submission

Extract `small_model_track_test.zip` locally, then run inference from the
selected checkpoint. `test.csv` is the only source of test paths; the command
refuses a populated prediction column and never reads `sample_submission.csv`.

```bash
torchrun --standalone --nproc_per_node=8 \
  -m complexity.generative.sensor_fusion.submission \
  --checkpoint artifacts/tr_hash_robot_perception/best \
  --data-root /workspace/datasets/CUHK-X/test-extracted \
  --test-csv /workspace/datasets/CUHK-X/small-model-track/Small-Model-Track/Testing/test_file/test.csv \
  --output artifacts/tr_hash_robot_perception/submission_best.csv \
  --batch-size 8 \
  --workers 2 \
  --require-fused-cuda
```

Rank zero gathers and sorts all predictions, verifies the exact official path
order and class range, writes `path,prediction`, and stores a compressed
`submission_best.logits.npz`. Keep the logits: they allow checkpoint ensembles
without decoding the multimodal test clips again.

CUHK-X data must remain local. Its license forbids redistributing or mirroring
the raw data and derived data shards. Source code, configuration, model weights,
and aggregate evaluation results can be published separately.
