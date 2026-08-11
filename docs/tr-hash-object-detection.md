# TR-Hash Object Detection and Serving

The detector is a compact, anchor-free, single-stage model. A TR-Hash vision
tower feeds three lightweight prediction grids. Dynamic positive assignment
selects several useful cells per object, and Varifocal objectness learns an
IoU-aware confidence score.

```text
TR-Hash vision tower
        |
        +-- 16 x 16 -- prediction head
        +--  8 x 8  -- prediction head
        +--  4 x 4  -- prediction head
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
and fused AdamW.

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
  --vision-heads 4 --vision-expert-width 48 \
  --device mps
```

The trainer writes `metrics.jsonl`, a validated `best/` checkpoint, and a final
step checkpoint. Validation reports mAP50, precision, recall, F1, the best F1,
and its confidence threshold. The service uses that calibrated threshold by
default when `confidence` is omitted.
`--expert-lr-multiplier 1.5` can give the routed expert tensors a higher
learning rate than the shared tower and detection heads; the default `1.0`
keeps a single learning rate for controlled comparisons.

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
