# TR-Hash Object Detection and Serving

The v6 detector is a compact, anchor-free, single-stage model. A TR-Hash
vision tower feeds a lightweight additive PAN with top-down and bottom-up
cross-scale fusion. Its one-to-many branch
provides dense supervision during training and class-aware batched NMS removes
duplicate predictions at inference. Dynamic STAL-style assignment sends small
objects to the finest grid. Decoupled branches predict stride-local LTRB box
distributions and unified sigmoid quality-class scores trained with DFL/QFL.

```text
TR-Hash vision tower
        |
        +-- initial P2/P3/P4/P5 maps
                 | top-down additive FPN
                 | bottom-up additive PAN
                 v
            fused P2/P3/P4/P5
                 |
                 +-- LTRB/DFL regression + quality-class QFL heads
```

For a 128 px input, width 128, four tower layers, and four routed experts, the
default P2 plus one-to-one configuration has 927,246 parameters with 20
classes (approximately 0.93 million). Disabling both optional branches reduces
it to 810,452 parameters. Parameter count is not a quality metric by itself:
compare accuracy, latency, memory, and deployment size using the same dataset
and evaluation protocol.

## Detection pretraining pipeline

The native release recipe is task-aligned and provenance-locked:

```text
random initialization -> COCO 2017 full-detector training -> validated best model
```

The complete tower, multi-scale neck, box regression, class-quality heads,
routing and specialization losses are trained jointly on COCO. The launcher
rejects detector and backbone transfer flags. Every checkpoint records a
`provenance.json`; strict resume accepts only native checkpoints descending
from random initialization on the same dataset.

```bash
bash scripts/vast_run_detector_coco_native.sh
```

## v8 mAP upgrade baseline

The v8 nano recipe changes one training contract and three small architectural
components while keeping the TR-Hash tower at four experts with top-2 routing:

- a nominal global batch of 64 drives automatic gradient accumulation and
  weight-decay normalization, independently of GPU count;
- warmup is expressed in optimizer epochs rather than a machine-specific fixed
  number of microbatches;
- independent depthwise spatial context is added before the box and class MLPs;
- each pyramid level learns a positive DFL-logit scale;
- two PAN passes use positive per-channel normalized fusion weights.

The baseline deliberately remains P3/P4/P5 and O2M + NMS. P2 is a separate
controlled ablation, so any small-object AP gain is measurable.

```bash
# Baseline: about 2.76M parameters.
bash scripts/vast_train_detector_coco_v08_nano.sh

# Same recipe plus P2: about 3.02M parameters.
bash scripts/vast_train_detector_coco_v08_nano_p2.sh
```

Both launchers select checkpoints with official COCO mAP50-95. Compare them at
the same source-image exposure, optimizer-step budget, seed, confidence and
maximum detections. Do not promote v8 based on training loss or mAP50 alone.

## Train on detection data

Synthetic images are useful for bounded smoke tests. Production fine-tuning
should use a YOLO-format dataset. Separate validation folders can be passed as
below; when omitted, the trainer creates a deterministic 80/20 split.

```bash
cf-detector-train \
  --output artifacts/detector_v06 \
  --yolo-images data/objects/images/train \
  --yolo-labels data/objects/labels/train \
  --validation-yolo-images data/objects/images/val \
  --validation-yolo-labels data/objects/labels/val \
  --num-classes 20 \
  --image-size 128 --patch-size 8 \
  --vision-hidden-size 128 --vision-layers 4 \
  --vision-stage-depths 1 1 2 --vision-window-size 8 \
  --vision-heads 4 --vision-expert-width 48 \
  --optimizer musgd \
  --lr 1e-2 --momentum 0.937 \
  --weight-decay 5e-4 --expert-lr-multiplier 1.5 \
  --device mps
```

Progressive loss balancing and STAL assignment are enabled by default.
Progressive loss begins with stronger quality supervision and half-strength
box regression, then linearly reaches the configured final weights. P2 and
strong photometric/geometric augmentation are also enabled by default. Use
`--no-progressive-loss`, `--no-stal`, `--no-p2-head`, or
`--augmentation light` for controlled ablations. Training uses SGD with
Nesterov momentum through the required `musgd` optimizer, with separate
learning-rate groups for routed experts and the optional one-to-one branch.

Training logs report one unambiguous loss total and its quality, box, and DFL
components. The validated inference path remains O2M with
`torchvision.ops.batched_nms`.

The trainer writes `metrics.jsonl`, a validated `best/` checkpoint, and a final
step checkpoint. Validation reports mAP50, mAP50-95, AP small/medium/large,
precision, recall, F1, the best F1, and its confidence threshold. Use
`--eval-every 5` to avoid a full validation pass after every epoch;
the final epoch is always evaluated. `--eval-batch-size` and
`--eval-max-detections` control validation throughput and memory. The service
uses the calibrated confidence threshold by default when `confidence` is
omitted.
`--expert-lr-multiplier 1.5` can give the routed expert tensors a higher
learning rate than the shared tower and detection heads; the default `1.0`
keeps a single learning rate for controlled comparisons.

## Fine-tune a detector on a new label set

`--detector-checkpoint` implements YOLO-style detector transfer. It keeps all
compatible tower, feature-pyramid, hidden-head, and LTRB/DFL regression
weights. When the class count changes, new class rows stay freshly initialized.
An optional JSON class map copies known class rows; keys are target class IDs
and values are source class IDs.

```json
{
  "0": 14,
  "1": 6
}
```

```bash
cf-detector-train \
  --detector-checkpoint artifacts/detector_v06/best \
  --class-map data/custom/class-map.json \
  --yolo-images data/custom/images/train \
  --yolo-labels data/custom/labels/train \
  --validation-yolo-images data/custom/images/val \
  --validation-yolo-labels data/custom/labels/val \
  --output artifacts/detector_custom \
  --num-classes 2 \
  --image-size 224 --patch-size 8 \
  --vision-hidden-size 128 --vision-layers 4 \
  --vision-stage-depths 1 1 2 --vision-window-size 8 \
  --vision-heads 4 --vision-expert-width 48 \
  --optimizer musgd \
  --device cuda
```

Without `--class-map`, localization transfer still occurs but every target
class row is initialized for the new dataset. With an unchanged class count,
all class rows transfer automatically.

For multi-GPU runs, use `torchrun --standalone --nproc_per_node N -m
complexity.generative.detection.training ...`. DDP records each rank's RNG
state and world size so exact resume keeps the same number of processes. The
tracked native COCO launcher provides the reproducible release recipe; ad-hoc
dataset launchers are intentionally not kept in `scripts/`.

## Run the local service

Install the serving dependencies and load one validated checkpoint into a
long-lived process:

```bash
pip install -e ".[serve]"
cf-detector-serve \
  --checkpoint artifacts/detector_coco_v06_native/best \
  --device mps \
  --host 127.0.0.1 --port 8000
```

The API provides:

- `GET /health` and `GET /v1/model`;
- `POST /v1/predict` with a file upload;
- `POST /v1/train`, `GET /v1/train/{id}`, and
  `GET /v1/train/{id}/logs`;
- `DELETE /v1/train/{id}` to cancel;
- `POST /v1/train/{id}/promote` to atomically load a completed job's best
  checkpoint without stopping the API.

Example inference:

```bash
curl -F file=@sample.jpg \
  'http://127.0.0.1:8000/v1/predict?confidence=0.25&iou_threshold=0.45'
```

Example synthetic training job:

```bash
curl -X POST http://127.0.0.1:8000/v1/train \
  -H 'Content-Type: application/json' \
  -d '{"dataset":"synthetic","epochs":20,"synthetic_samples":4096}'
```

Set `TR_HASH_API_KEY` before launching to require `X-API-Key` on model and job
endpoints. The built-in job manager deliberately runs one training process at
a time and persists metadata and logs. For a public or multi-machine service,
place the API behind TLS/authentication and replace the local process manager
with a durable queue and isolated workers.

## Other vision tasks

The same routed tower now exposes model variants for detection, instance
segmentation, semantic segmentation, monocular depth, classification, pose,
and oriented bounding boxes (OBB):

```python
from complexity.generative import TRHashDetectorConfig, create_vision_model

config = TRHashDetectorConfig(
    image_size=224,
    patch_size=8,
    vision_hidden_size=128,
    vision_layers=4,
    vision_heads=4,
    vision_expert_width=48,
    num_classes=20,
    p2_head=True,
)

detector = create_vision_model("detection", config)
instances = create_vision_model("instance_segmentation", config)
semantic = create_vision_model("semantic_segmentation", config, num_classes=21)
depth = create_vision_model("depth", config, max_depth=80.0)
classifier = create_vision_model("classification", config, num_classes=1000)
pose = create_vision_model("pose", config, num_keypoints=17)
obb = create_vision_model("obb", config)
```

The task heads and differentiable losses are implemented. Detection has the
complete YOLO/COCO trainer and serving path described above. The other task
families still need dataset-specific loaders, augmentation, metrics, export,
and serving endpoints before they should be called production pipelines.

## Publish a validated detector to Hugging Face

The publishing helper builds the model card, preprocessing metadata, class
names, metrics, provenance and `safetensors` checkpoint folder. A card-only
COCO draft can be created while training is active:

```bash
python scripts/publish_tr_hash_vision_hf.py \
  --repo-id AETHORIA-AI/TR-HASH-Vision-v6-1M-COCO \
  --dataset coco \
  --training --push
```

After selecting the final validated checkpoint, replace the draft with the
complete release. Repositories are private by default; add `--public` only when
the model card and metrics have been reviewed.

```bash
python scripts/publish_tr_hash_vision_hf.py \
  --repo-id AETHORIA-AI/TR-HASH-Vision-v6-1M-COCO \
  --dataset coco \
  --checkpoint artifacts/detector_coco_v06_native/best \
  --push --public
```

Consumers can use `load_detector_from_hub`, `preprocess_detector_image`, and
`restore_detector_boxes` to reproduce the training letterbox geometry and map
normalized predictions back to source-image pixels.
