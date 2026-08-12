# TR-Hash Vision v6 ablation protocol

The first reference is the 50-epoch Pascal VOC v6 run with the ImageNet-1K
tower, PAN, P2, STAL and the one-to-one NMS-free auxiliary branch. Every
ablation within a dataset uses the same seed, data order, augmentations,
optimizer, learning-rate schedule, validation cadence and training budget.

Run one arm at a time:

```bash
bash scripts/vast_ablate_detector_v06.sh no-stal
```

Run the same controlled arm on COCO 2017 with the existing v6 COCO recipe:

```bash
DATASET=coco bash scripts/vast_ablate_detector_v06.sh no-stal
```

The COCO paths follow the existing launcher and must contain
`instances_{train,val}2017.json` plus `images/{train,val}2017`. The reusable
ImageNet-1K initialization checkpoint is published at
`AETHORIA-AI/TR-HASH-Vision-V6-ImageNet1K-Pretrain`; download it into the
default `artifacts/tr_hash_vision_v06_imagenet1k/best` path or set `BACKBONE`.

The supported arms isolate one change:

| Arm | Change from the full reference | Question |
|---|---|---|
| `o2m-only` | remove the one-to-one branch | Does O2O help or hurt the shared detector? |
| `no-stal` | disable STAL | Does STAL improve AP small? |
| `no-p2` | remove P2 | Does the fine level improve AP small enough to justify its cost? |
| `fpn` | replace PAN with additive FPN | Is bottom-up PAN fusion useful? |
| `no-neck` | remove cross-scale fusion | How much does any neck contribute? |

The full sequential suite is available, but should only be launched after the
reference run has finished because a concurrent run changes both throughput
and the experimental conditions:

```bash
bash scripts/vast_run_detector_v06_ablation_suite.sh
```

Set `DATASET=coco` for a COCO suite. Start with one arm and verify its manifest
before committing the full five-arm compute budget.

Collect a table including the existing reference:

```bash
python scripts/collect_detector_v06_ablations.py \
  --root artifacts/ablations/detector_v06_voc \
  --reference artifacts/detector_voc_v06_imagenet1k_nmsfree
```

Report both O2M+NMS and O2O NMS-free checkpoints where available. The primary
metric is mAP50-95; AP small, medium and large are mandatory secondary metrics.
Do not compare arms trained for different epochs or global batch sizes.

For external contributions, record the framework commit, GPU model/count,
complete command, wall time and peak VRAM alongside `summary.csv`. Submit raw
`validation.json` files and training metrics with the table so results remain
auditable.
