# TR-Hash detector specialization

The experimental specialization path extends the v6 COCO detector while
preserving its hierarchical TR-Hash tower.

## Image path

- Residual P2-P5 adapters are zero-initialized and exactly neutral at startup.
- The class x level gate hashes each identity through the configured TR-Hash
  experts, then applies a calibrated level prior to classification logits.
- Positive examples can be weighted by class x normalized size x scene density.
- Auxiliary objectives balance levels, calibrate the hash gate against actual
  assignments, and contrast positive object embeddings by class.

The release path begins directly with native detection training:

1. `scripts/vast_train_detector_specialized_coco.sh` initializes the complete
   detector randomly and trains it jointly on COCO 2017 at 640 px.
2. `scripts/vast_run_detector_coco_native.sh` resumes only native checkpoints
   carrying matching random-init COCO provenance.

External detector and backbone checkpoints are deliberately rejected. Tower,
PAN, box/class heads, one-to-one branch, adapters and hash gate are optimized
together.

## Video path

COCO-Video JSON uses standard COCO fields plus `video_id` and `frame_id` on
every image. A sample is an odd-length temporal clip. The center frame owns the
detection targets and boundary frames are repeated when necessary.

The spatial tower processes the center frame. Signed and absolute consecutive
frame differences form a motion residual injected into the same P2-P5 maps.
Repeated-frame clips are therefore exactly equivalent to static images.

Run `scripts/vast_train_detector_specialized_video.sh` after setting the four
dataset path variables documented by the script. Mosaic, MixUp, Copy-Paste and
random erasing are deliberately rejected for clips until synchronized temporal
implementations exist.

## Required ablations

1. v6 reference
2. adapters only
3. adapters + class-level hash gate
4. add 3D weighting
5. add level/gate auxiliary losses
6. add object contrastive loss
7. for a video dataset, repeat-frame control versus real temporal clips

Report COCO AP50-95 and AP small/medium/large with the same seed, data split,
global batch, number of optimizer steps, and validation filtering.

Set `ABLATION` to `baseline`, `adapters`, `hash-gate`, `weighting`, `auxiliary`,
or `full`.
`scripts/vast_run_detector_specialization_ablations.sh` executes those arms in
order after a true `baseline`, each transferred from the same intermediate
detector checkpoint. Interrupted arms resume exactly; completed arms are
skipped. The final collector refuses incomplete runs, uncontrolled config or
training-budget drift, then writes `summary.csv`, `summary.md` and
`protocol.json` with mAP50-95 and AP small/medium/large deltas.
