# TR-Hash instruction-guided image editor

> **Experimental modality.** This implementation has no released checkpoint or
> benchmark claim in the TR-HASH MoE 200M language-model lineage.

This model learns the direct mapping

```text
source image + edit instruction -> target image
```

It is not a captioning pipeline and it does not turn the source image into
text. The frozen VAE encodes both source and target images. The source latent
is injected spatially before and throughout the latent transformer, while the
target latent follows a rectified-flow noise path. Every MLP block uses the
canonical TR-Hash MoE engine.

## Training record format

Training uses WebDataset TAR shards. Every record must contain the following
members with an identical sample identifier:

```text
000001.source.webp   # image before the edit
000001.target.webp   # expected image after the edit
000001.txt           # imperative edit instruction
000001.json          # provenance, license, transformation metadata
```

The source and target must be a genuine aligned edit pair. Captions paired to
one image are not sufficient. Metadata should preserve the license and origin
of both images.

## Initialize from the refined text-to-image model

The editor can reuse a **refined** text-to-image checkpoint with the same
architecture. Direct initialization from stage-1 pretraining is outside the
release contract. New source-conditioning gates start at zero, preserving the
source model before edit SFT.

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/train_tr_hash_image_editor.py \
  --config configs/tr_hash_image_editor_200m.yaml \
  --shards '/workspace/data/atlas-edits/train/*.tar' \
  --samples-per-epoch 300000 \
  --tokenizer tokenizer/tokenizer.json \
  --init-text-to-image /workspace/checkpoints/text-to-image/step_0100000 \
  --source-stage refinement \
  --output /workspace/artifacts/tr-hash-image-editor-200m \
  --epochs 3 \
  --batch-size 8 \
  --bf16
```

Use `--max-steps` instead of `--samples-per-epoch` when the intended optimizer
step count is known exactly. Checkpoints contain model, optimizer, scheduler,
configuration, and step state; only the newest four are retained by default.

## Edit an image

```bash
python scripts/edit_tr_hash_image.py \
  /workspace/artifacts/tr-hash-image-editor-200m/step_0100000 \
  --source source.png \
  --instruction 'Turn the daytime sky into a moonlit night while preserving the buildings.' \
  --image-guidance 1.5 \
  --text-guidance 5.0 \
  --output edited.png
```

Image guidance controls structural fidelity to the source. Text guidance
controls adherence to the instruction. They are separate because an editor
must balance preservation and transformation rather than merely maximize text
conditioning.
