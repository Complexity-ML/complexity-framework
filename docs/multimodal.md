# Multimodal prototypes

`complexity.multimodal` contains research prototypes for vision, audio, video,
fusion, and an omni wrapper.

For the supported hash-routed sensor-fusion path, use the public **TR-Hash
Robotics** factory instead of the historical fusion prototypes:

```python
from complexity.api import Robot

model = Robot.model(
    num_classes=40,
    num_experts=8,
    top_k=2,
)
```

`Robot` currently provides multimodal perception: depth, infrared, thermal,
IMU, radar, and skeleton observations are fused into a shared state
representation and classification head. It is suitable as the perception
front-end of a robotics stack. It does not yet include policy, trajectory,
value, or actuator-control heads.

There is a single architecture behind `Robot` (`TRHashSensorFusionClassifier`)
-- no version flag, no reduced/legacy variant. See
[`docs/tr_hash_sensor_fusion.md`](tr_hash_sensor_fusion.md) for the full
architecture reference and the CUHK-X dataset/training/submission pipeline it
was originally developed and benchmarked against.

These modules are separate from the text TR-GQA/TR-MHA evidence. Their routing
keys are positions, not tokenizer IDs:

| Module | Route key |
| --- | --- |
| vision | patch position |
| audio | time-step position |
| video | spatiotemporal position |
| fusion | query position |

Do not describe these as lexical TR-MoE without that qualification.

## Vision

```python
import torch

from complexity.multimodal import VisionEncoder

encoder = VisionEncoder(
    image_size=224,
    patch_size=16,
    hidden_size=384,
    num_layers=4,
    num_heads=6,
    num_experts=4,
)
pixels = torch.randn(2, 3, 224, 224)
features = encoder(pixels)
```

`num_experts=1` selects the dense MLP fallback. Values above one enable
position-routed MLPs.

## Audio

```python
from complexity.multimodal import AudioEncoder

encoder = AudioEncoder(
    n_mels=80,
    hidden_size=384,
    num_layers=4,
    num_heads=6,
    num_experts=4,
)
features = encoder(mel_spectrogram)
```

## Video

```python
from complexity.multimodal import VideoEncoder

encoder = VideoEncoder(num_experts=4)
features = encoder(video)
```

Expected video layout is documented by the module as
`[batch, channels, frames, height, width]`.

## Fusion

```python
from complexity.multimodal import MultimodalFusion

fusion = MultimodalFusion(num_experts=4)
```

Available components include cross-attention, gated fusion, concatenation,
Perceiver resampling, and a vision-language connector.

## Status

The multimodal package is experimental:

- it does not inherit the text-model training results;
- it has no production serving contract in this repository;
- position routing and token-ID routing answer different research questions;
- users should add modality-specific parity, masking, and benchmark tests
  before drawing conclusions.
