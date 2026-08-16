# TR-HASH Vision dependency stack

The detector uses the same public third-party computer-vision primitives as
the Ultralytics runtime where they are relevant to TR-HASH Vision:

- PyTorch and matching Torchvision wheels;
- OpenCV, Pillow, NumPy and Matplotlib;
- Albumentations for detection augmentation;
- pycocotools and faster-coco-eval for COCO evaluation;
- psutil, Polars and nvidia-ml-py for runtime measurements;
- ONNX, ONNX Runtime and onnxslim for portable export.

This is dependency-level interoperability only. The framework does not import
or copy Ultralytics model code. `ultralytics-platform` is intentionally omitted
because it is a hosted-platform client, and `ultralytics-thop` is unnecessary
because detector FLOPs are measured with PyTorch's native flop counter.

## Backend-safe installation

The installer resolves Torch and Torchvision together from the same backend
index. This prevents a ROCm installation from being silently replaced by a
CUDA wheel.

```bash
make install-vision-cuda
make install-vision-rocm
make install-vision-cpu
```

For an environment where the correct Torch stack is already installed:

```bash
pip install -e '.[detection,export]'
```

Verify a machine before training or export:

```bash
python scripts/check_vision_stack.py --strict
```

The reference dependency versions are tracked from the public Ultralytics
`pyproject.toml`; backend selection and model implementation remain native to
Complexity Framework.

## Active detector pipeline

The COCO launchers use the installed primitives directly rather than only
declaring them as optional packages:

- OpenCV decodes random-access COCO and YOLO images;
- Albumentations applies box-aware geometric and color transforms;
- `faster-coco-eval` is selected automatically, with `pycocotools` as the
  official reference fallback;
- distributed validation gathers each non-overlapping COCO shard and executes
  COCOeval once on rank zero;
- `best` and `best_nms_free` are selected by official COCO mAP50-95.

Official comparable evaluation retains at most 100 detections per image. The
framework's internal metric implementation remains available as a diagnostic,
but it is no longer used to select or publish a COCO checkpoint.

The specialized COCO launcher enables this stack by default. The equivalent
explicit training options are:

```bash
--image-backend opencv \
--augmentation-backend albumentations \
--eval-backend auto \
--eval-max-detections 100
```

Use the standalone evaluator to reproduce published metrics:

```bash
python scripts/evaluate_tr_hash_coco.py \
  artifacts/detector_coco_v06_native/best \
  --annotations artifacts/COCO/annotations/instances_val2017.json \
  --images artifacts/COCO/val2017 \
  --output artifacts/detector_coco_v06_native/evaluation \
  --eval-backend auto
```

Measure the decoder on the target machine before treating OpenCV as a speed
claim:

```bash
python scripts/check_detection_io_performance.py \
  --annotations artifacts/COCO/annotations/instances_train2017.json \
  --images artifacts/COCO/train2017 \
  --samples 500 --repeats 3
```
