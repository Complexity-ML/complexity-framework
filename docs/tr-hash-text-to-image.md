# TR-Hash text-to-image

The image generator is a 203.5M-parameter latent rectified-flow transformer.
It is native to Complexity Framework and uses the canonical `TRHashEngine` in
every image block.

## Architecture

- output resolution: 256×256;
- frozen KL autoencoder: RGB → 4×32×32 latent;
- latent patching: 2×2, yielding 256 image tokens;
- caption encoder: 32k tokenizer, 4 transformer layers;
- flow backbone: 14 transformer layers, width 768;
- routing: spatial-position + timestep bucket, balanced independently by layer;
- experts: four stored, deterministic top-2 active, plus a shared SwiGLU path;
- conditioning: caption cross-attention and adaptive normalization;
- objective: rectified-flow velocity matching with 10% caption dropout for CFG.

The VAE is frozen and external to the 203.5M trainable parameters. The default
adapter uses `stabilityai/sd-vae-ft-mse`; another compatible four-channel,
8×-downsampling `AutoencoderKL` can be supplied.

## Install

```bash
source /venv/main/bin/activate
pip install -e '.[cuda,image]'
```

## Dataset

Download only the training shards from the public Atlas Images repository:

```bash
hf download Pacific-i64/complexity-atlas-images \
  --repo-type dataset \
  --include 'train/*.tar' \
  --local-dir /workspace/data/complexity-atlas-images
```

The loader reads `.webp`, `.txt`, and `.json` records directly inside each TAR.
It does not extract or duplicate the 6.7 GB image bank.

## Four-GPU training

Start with batch 2 per 16 GB GPU. Gradient checkpointing is enabled by default.

```bash
torchrun --standalone --nproc_per_node=4 \
  -m complexity.generative.image.training \
  --config configs/tr_hash_text_to_image_200m.yaml \
  --shards '/workspace/data/complexity-atlas-images/train/*.tar' \
  --tokenizer tokenizer/tokenizer.json \
  --output /workspace/artifacts/tr-hash-image-200m \
  --batch-size 8 \
  --keep-checkpoints 4 \
  --gradient-accumulation 2 \
  --epochs 1 \
  --bf16
```

The effective global batch in this example is 16 images. Checkpoints contain
`model.safetensors`, `config.json`, and a resumable AdamW training state.

## Smoke test

Use a tiny model to verify CUDA, DDP, dataset decoding, the VAE, and backward
before starting the full run:

```bash
python -m pytest -q tests/test_tr_hash_text_to_image.py
```

## Generation

```bash
python scripts/generate_tr_hash_image.py \
  /workspace/artifacts/tr-hash-image-200m/step_0005000 \
  --prompt 'An engraved astronomical instrument on a dark museum table' \
  --steps 30 \
  --guidance 4.0 \
  --output sample.png
```
