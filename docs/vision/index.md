# Vision

Vision models, training, evaluation and export now live in one package tree:
`complexity.vision`. Start with one command family and choose the task; the
selected runtime retains its existing arguments and checkpoint contracts.

```bash
complexity vision list
complexity vision train detector -- --help
complexity vision evaluate detector -- --help
complexity vision evaluate onnx -- --help
complexity vision check coco -- --help
complexity vision export detector -- --help
```

The `--` separator forwards following arguments, including `--help`, to the
runtime. Listing or help does not train, download a model or start a server.
Operations need the dependency profile specified by their guide.

## Choose a task

| Task | Command family | Guide |
|---|---|---|
| Detection training/transfer/service | `vision train detector`, `vision serve detector` | [Object detection](tr-hash-object-detection.md) |
| Image generation | `vision train image`, `vision generate image` | [Text to image](tr-hash-text-to-image.md) |
| Instruction-guided image editing | `vision train image-edit`, `vision edit image` | [Image editor](tr-hash-image-editor.md) |
| Image + text to text | `vision train image-text` | [Vision-language model](tr-hash-image-text-to-text.md) |
| Backbone pretraining | `vision pretrain backbone` | [Dependency stack](vision-dependency-stack.md) and runtime help |
| Native/ONNX COCO evaluation | `vision evaluate detector`, `vision evaluate onnx` | [COCO accuracy gates](vision-v8-coco-accuracy-gates.md) |
| COCO report validation/merging | `vision check coco`, `vision merge coco` | [COCO accuracy gates](vision-v8-coco-accuracy-gates.md) |
| Detector ONNX export/deployment | `vision export detector` | [Deployment](onnx_deploy.md), [tensor contract](onnx_deploy_contract.md) |
| ONNX inference/TensorRT export | `vision infer onnx`, `vision export tensorrt` | [Deployment](onnx_deploy.md) |
| Quantization/benchmark/parity | `vision quantize onnx`, `vision benchmark onnx`, `vision check quantized-parity` | [Quantization gates](vision-v8-onnx-quantization.md) |
| Native ONNX parity/release manifest | `vision check onnx-parity`, `vision release onnx` | [Validation report](../onnx/tr_hash_v8_validation_report.md) |

All commands above use the `complexity` prefix, followed by forwarded runtime
arguments, for example `complexity vision train detector -- --optimizer musgd ...`.
The COCO/ONNX operations, report validators and release tooling are installed
with the framework. Their former 13 script files are removed. Defaults are
packaged JSON resources; these commands also work outside the source checkout.
Source provenance is resolved from the framework checkout itself. An installed
wheel without that checkout records `unknown`, rather than the commit of the
project using it. Build reproducible release artifacts from the pinned framework
checkout so their manifest can identify the source revision.

## Python tree

```text
complexity.vision
├── detection       detector, loss, data, train, evaluation, ONNX and service
├── image           text-to-image, editor, training and image inference
├── video           routed video tower
├── vision_language image/text model and visual pretraining
└── vision_tasks    segmentation, depth, classification, pose and OBB heads
```

```python
from complexity.vision import TRHashDetectorConfig, create_vision_model
from complexity.vision.vision_language import TRHashImageTextToText
```

Existing `complexity.generative` imports alias the same classes/modules for
checkpoint and caller compatibility. New work should use `complexity.vision`.
Audio and sensor fusion keep their own modules and are indexed in the broader
[multimodal guide](../multimodal.md).

## Evidence and dependencies

The v8 detector has its own reported release measurements and COCO protocol.
Image generation, editing, visual conversation and other task heads remain
experimental unless their guide identifies a released artifact. None inherits
the language-model release's PIQA results or the paused 1B training budget.

- [Install/backend/dependency contract](vision-dependency-stack.md)
- [COCO accuracy gates](vision-v8-coco-accuracy-gates.md)
- [ONNX quantization and runtime gates](vision-v8-onnx-quantization.md)
- [Historical detector specialization](TR_HASH_DETECTOR_SPECIALIZATION.md)
- [Framework boundaries](../framework-architecture.md)
- [General workflows](../workflows.md)

Use the exact tokenizer, configuration, geometry/provenance and checkpoint
manifest belonging to the selected family. A tiny structural test, one release
measurement and a production deployment are distinct validation scopes.
