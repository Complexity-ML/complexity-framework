# TR-Hash Object Detection and Serving

The v0.3 detector is a compact, anchor-free, single-stage model. A TR-Hash
vision tower feeds a lightweight prediction pyramid. Its one-to-many branch
provides dense supervision during training and class-aware batched NMS removes
duplicate predictions at inference. Dynamic STAL-style assignment sends small
objects to the finest grid and Varifocal objectness learns an IoU-aware score.

```text
TR-Hash vision tower
        |
        +-- P2 (optional) -- one-to-many head
        +-- P3            -- one-to-many head
        +-- P4            -- one-to-many head
        +-- P5            -- one-to-many head
```

For a 128 px input, width 128, four tower layers, and four routed experts, the
model has approximately 0.71 million parameters. Parameter count is not a
quality metric by itself: compare accuracy, latency, memory, and deployment
size using the same dataset and evaluation protocol.

## Pretrain the vision tower

Pretraining accepts CIFAR-10, a Hugging Face image-classification dataset, or
an ImageFolder containing `train/` and `val/` class directories. On Apple
Silicon, `--device mps` uses the Mac GPU and keeps the vision computation in
fp32 for stability. CUDA uses BF16 autocast, pinned persistent data workers,
and accelerated optimizer kernels when available.

```bash
cf-vision-pretrain \
  --cifar10 \
  --data-root artifacts/cifar10 \
  --output artifacts/tr_hash_vision_cifar10 \
  --image-size 128 --patch-size 8 \
  --hidden-size 128 --layers 4 --heads 4 --expert-width 48 \
  --device mps
```

For a stronger real-image initialization, ImageNet-100 provides roughly
127,000 training images across 100 classes:

```bash
pip install datasets 'huggingface_hub[hf_xet]'
HF_XET_HIGH_PERFORMANCE=1 cf-vision-pretrain \
  --hf-dataset clane9/imagenet-100 \
  --data-root artifacts/hf-cache \
  --output artifacts/tr_hash_vision_imagenet100 \
  --image-size 224 --patch-size 8 \
  --hidden-size 128 --layers 4 --heads 4 --expert-width 48 \
  --epochs 30 --batch-size 512 --workers 8 \
  --expert-lr-multiplier 1.5 --device cuda
```

## Train on detection data

Synthetic images are useful for bounded smoke tests. Production fine-tuning
should use a YOLO-format dataset. Separate validation folders can be passed as
below; when omitted, the trainer creates a deterministic 80/20 split.

```bash
cf-detector-train \
  --output artifacts/detector_v02_mps \
  --backbone-checkpoint artifacts/tr_hash_vision_cifar10/best \
  --yolo-images data/objects/images/train \
  --yolo-labels data/objects/labels/train \
  --validation-yolo-images data/objects/images/val \
  --validation-yolo-labels data/objects/labels/val \
  --image-size 128 --patch-size 8 \
  --vision-hidden-size 128 --vision-layers 4 \
  --vision-heads 4 --vision-expert-width 48 --p2-head \
  --optimizer sgd --lr 1e-2 --momentum 0.937 \
  --weight-decay 5e-4 --expert-lr-multiplier 1.5 \
  --device mps
```

Progressive loss balancing and STAL assignment are enabled by default.
Progressive loss begins
with stronger objectness supervision and half-strength box regression, then
linearly reaches the configured final weights. Old unversioned checkpoints
load with their legacy sigmoid center decoding. Use `--no-progressive-loss`
or `--no-stal` for controlled ablations. AdamW remains selectable only to reproduce an older baseline; the
v0.3 default is SGD with Nesterov momentum and a separate expert LR group.

The unsuccessful one-to-one/NMS-free experiment was removed after controlled
validation measured mAP50 0.0630 versus 0.0995 for the same checkpoint's
one-to-many branch. Legacy checkpoints containing `one_to_one_heads.*` remain
loadable: those obsolete tensors are discarded and inference uses the validated
one-to-many head with `torchvision.ops.batched_nms`.

Training logs report one unambiguous loss total and its objectness, box, and
class components. There is no hidden auxiliary loss added to the total.

The trainer writes `metrics.jsonl`, a validated `best/` checkpoint, and a final
step checkpoint. Validation reports mAP50, precision, recall, F1, the best F1,
and its confidence threshold. The service uses that calibrated threshold by
default when `confidence` is omitted.
`--expert-lr-multiplier 1.5` can give the routed expert tensors a higher
learning rate than the shared tower and detection heads; the default `1.0`
keeps a single learning rate for controlled comparisons.

## Fine-tune a detector on a new label set

`--detector-checkpoint` implements YOLO-style detector transfer. It keeps all
compatible tower, feature-pyramid, hidden-head, box-regression, and objectness
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
  --detector-checkpoint artifacts/detector_voc_5090_imagenet100/best \
  --class-map data/custom/class-map.json \
  --yolo-images data/custom/images/train \
  --yolo-labels data/custom/labels/train \
  --validation-yolo-images data/custom/images/val \
  --validation-yolo-labels data/custom/labels/val \
  --output artifacts/detector_custom \
  --image-size 224 --patch-size 8 \
  --vision-hidden-size 128 --vision-layers 4 \
  --vision-heads 4 --vision-expert-width 48 \
  --device cuda
```

Without `--class-map`, box/objectness transfer still occurs but every target
class row is initialized for the new dataset. With an unchanged class count,
all class rows transfer automatically.

For a practical public corpus, prepare Pascal VOC 2007+2012 (16,551 training
images, 4,952 validation images, 20 classes):

```bash
python scripts/prepare_voc_yolo.py --output artifacts/VOC

cf-detector-train \
  --output artifacts/detector_voc_mps \
  --backbone-checkpoint artifacts/detector_v02_mps/best \
  --yolo-images artifacts/VOC/images/train \
  --yolo-labels artifacts/VOC/labels/train \
  --validation-yolo-images artifacts/VOC/images/val \
  --validation-yolo-labels artifacts/VOC/labels/val \
  --image-size 224 --patch-size 8 \
  --vision-hidden-size 128 --vision-layers 4 \
  --vision-heads 4 --vision-expert-width 48 \
  --expert-lr-multiplier 1.5 --device mps
```

## Run the local service

Install the serving dependencies and load one validated checkpoint into a
long-lived process:

```bash
pip install -e ".[serve]"
cf-detector-serve \
  --checkpoint artifacts/detector_v02_mps/best \
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
complete VOC/YOLO trainer and serving path described above. The other task
families still need dataset-specific loaders, augmentation, metrics, export,
and serving endpoints before they should be called production pipelines.

## Publish a validated detector to Hugging Face

The publishing helper builds the model card, preprocessing metadata, VOC class
names, metrics, and `safetensors` checkpoint folder. A private card-only draft
can be created while a run is active:

```bash
python scripts/publish_tr_hash_vision_hf.py \
  --repo-id AETHORIA-AI/TR-HASH-Vision-0.8M-VOC \
  --training --push
```

After selecting the final validated checkpoint, replace the draft with the
complete release. Repositories are private by default; add `--public` only when
the model card and metrics have been reviewed.

```bash
python scripts/publish_tr_hash_vision_hf.py \
  --repo-id AETHORIA-AI/TR-HASH-Vision-0.8M-VOC \
  --checkpoint artifacts/detector_voc_5090_v03/best \
  --push --public
```

Consumers can use `load_detector_from_hub`, `preprocess_detector_image`, and
`restore_detector_boxes` to reproduce the training letterbox geometry and map
normalized predictions back to source-image pixels.
