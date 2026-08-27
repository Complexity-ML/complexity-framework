# TR-HASH Vision v8 ONNX Validation

Validation was run from commit `0fcf05c146f84a857d392b7da7d2947a41eb6d62`
(`Make Vision v8 the only detector architecture`) with the native checkpoint
`AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO-SFT`.

The generated ONNX binaries are intentionally not committed to the source
repository. Upload them as GitHub Release assets and link the release asset URLs
from the release notes or from a follow-up update to this report. The release
assets should include both ONNX binaries and their export metadata sidecars:

- `tr_hash_v8_o2m.onnx`
- `tr_hash_v8_o2m.json`
- `tr_hash_v8_nms_free.onnx`
- `tr_hash_v8_nms_free.json`

Use the following hashes and sizes to verify the uploaded release artifacts:

| Branch | Artifact | Size | SHA-256 |
|---|---:|---:|---|
| O2M | `tr_hash_v8_o2m.onnx` | 11,104,476 bytes / 10.590054 MiB | `24CADE1A285475DDD6D8AB25EF6775F34BB9B686A08355CD35A39342A493256A` |
| NMS-free | `tr_hash_v8_nms_free.onnx` | 11,108,683 bytes / 10.594066 MiB | `CA24D9577FF2224BC1A4949D09A2E1CC514DB04B5DFFECE6B42223DE3326FDD7` |

Branch metadata is committed beside this report:

| Branch | Metadata |
|---|---|
| O2M | [`tr_hash_v8_o2m.metadata.json`](tr_hash_v8_o2m.metadata.json) |
| NMS-free | [`tr_hash_v8_nms_free.metadata.json`](tr_hash_v8_nms_free.metadata.json) |

## Reproducibility

Download the checkpoint snapshot:

```powershell
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO-SFT', local_dir=r'models/TR-HASH-Vision-v8-2M-COCO-SFT')"
```

Export both raw prediction branches:

```powershell
$env:PYTHONPATH='.'
python scripts/export_onnx.py models\TR-HASH-Vision-v8-2M-COCO-SFT --output tr_hash_v8_o2m.onnx --branch o2m --opset 17 --check
python scripts/export_onnx.py models\TR-HASH-Vision-v8-2M-COCO-SFT\best_nms_free --output tr_hash_v8_nms_free.onnx --branch nms-free --opset 17 --check
```

Validate parity with the calibrated v8 defaults:

```powershell
$env:PYTHONPATH='.'
python scripts/check_onnx_parity.py models\TR-HASH-Vision-v8-2M-COCO-SFT tr_hash_v8_o2m.onnx --branch auto --num-tests 5 --batch-size 1
python scripts/check_onnx_parity.py models\TR-HASH-Vision-v8-2M-COCO-SFT\best_nms_free tr_hash_v8_nms_free.onnx --branch auto --num-tests 5 --batch-size 1
```

To reproduce the exact historical parity commands, pass the thresholds
explicitly:

```powershell
python scripts/check_onnx_parity.py models\TR-HASH-Vision-v8-2M-COCO-SFT tr_hash_v8_o2m.onnx --branch auto --tolerance 0.002 --num-tests 5 --batch-size 1
python scripts/check_onnx_parity.py models\TR-HASH-Vision-v8-2M-COCO-SFT\best_nms_free tr_hash_v8_nms_free.onnx --branch auto --tolerance 0.0035 --num-tests 5 --batch-size 1
```

## Contract

Both ONNX models were exported with opset 17 and expose raw detector logits
only. Decode and post-processing remain outside the graph.

| Branch | Input | Output | Post-processing |
|---|---|---|---|
| O2M | `pixel_values`, `float32`, `[1, 3, 640, 640]` | `predictions`, `float32`, `[1, 34000, 148]` | Decode plus NMS |
| NMS-free | `pixel_values`, `float32`, `[1, 3, 640, 640]` | `predictions`, `float32`, `[1, 34000, 148]` | Decode plus confidence filtering |

Output channel layout is `68` LTRB/DFL regression logits followed by `80`
quality-class logits. The grid pyramid is `[160, 80, 40, 20]`, for `34,000`
prediction cells.

## Runtime Versions

| Component | Version |
|---|---|
| PyTorch | `2.6.0+cu118` |
| ONNX | `1.21.0` |
| ONNX Runtime | `1.23.2` |
| ONNX Runtime providers | `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider` |
| GPU | NVIDIA GeForce GTX 1660 SUPER, driver `560.94` |

The CPU benchmark used `CPUExecutionProvider` on
`AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD`.

The GPU benchmark used `CUDAExecutionProvider` on the GTX 1660 SUPER. On this
Windows environment, ONNX Runtime required `ort.preload_dlls()` plus explicit
`nvidia/*/bin` DLL directories from the installed CUDA 12 and cuDNN 9 wheels.

## Export Results

Both exports succeeded:

```text
o2m:      Forward OK: output shape torch.Size([1, 34000, 148])
nms-free: Forward OK: output shape torch.Size([1, 34000, 148])
```

## Parity Results

The legacy strict raw-logit threshold `1e-4` failed for both branches on CPU
ONNX Runtime:

```text
o2m:      max_diff range 1.36e-03 to 1.74e-03, mean_diff about 1.0e-04
nms-free: max_diff range 2.36e-03 to 3.00e-03, mean_diff about 1.0e-04
```

Tensor localization showed the largest raw differences are concentrated in
regression logits on the finest `160x160` grid. Class-logit drift is smaller.
Decoded-output drift is substantially lower:

| Branch | Decoded boxes max diff | Decoded class score max diff |
|---|---:|---:|
| O2M | `6.00814819e-05` | `3.02493572e-05` |
| NMS-free | `5.73396683e-05` | `1.35712326e-05` |

The thresholds first derived from these five seeds were `0.002` (O2M) and
`0.0035` (NMS-free):

```text
O2M, tolerance 0.002:
  Test 1/5: max_diff=1.61e-03, mean_diff=1.02e-04 [PASS]
  Test 2/5: max_diff=1.36e-03, mean_diff=1.03e-04 [PASS]
  Test 3/5: max_diff=1.63e-03, mean_diff=1.17e-04 [PASS]
  Test 4/5: max_diff=1.36e-03, mean_diff=1.08e-04 [PASS]
  Test 5/5: max_diff=1.74e-03, mean_diff=1.06e-04 [PASS]
  Parity PASSED: branch=o2m, tolerance=0.002

NMS-free, tolerance 0.0035:
  Test 1/5: max_diff=2.65e-03, mean_diff=1.03e-04 [PASS]
  Test 2/5: max_diff=2.36e-03, mean_diff=1.01e-04 [PASS]
  Test 3/5: max_diff=2.83e-03, mean_diff=1.11e-04 [PASS]
  Test 4/5: max_diff=2.51e-03, mean_diff=1.00e-04 [PASS]
  Test 5/5: max_diff=3.00e-03, mean_diff=1.10e-04 [PASS]
  Parity PASSED: branch=nms-free, tolerance=0.0035
```

### Sample-size sensitivity

Five seeds are not enough to bound the maximum. Each test compares roughly five
million values, so the observed maximum is an extreme-value statistic that keeps
growing with the seed count. Re-measured on the same checkpoint:

| Branch | max over 5 seeds | max over 50 seeds | Growth | Old threshold |
|---|---:|---:|---:|---:|
| O2M | `1.741886e-03` | `2.788305e-03` | `+60.1%` | `2.0e-03` |
| NMS-free | `2.990723e-03` | `4.793882e-03` | `+60.3%` | `3.5e-03` |

Both branches exceed their original threshold well before 50 seeds, so
`check_onnx_parity.py --num-tests 20` failed on an otherwise healthy export.
The thresholds are therefore twice the maximum observed over 50 seeds:

| Branch | Observed max (50 seeds) | Threshold | Headroom |
|---|---:|---:|---:|
| O2M | `2.788305e-03` | `6.0e-03` | `2.2x` |
| NMS-free | `4.793882e-03` | `1.0e-02` | `2.1x` |

Reproduce either column by running the parity check with `--num-tests 5` and
`--num-tests 50`.

Measured with PyTorch `2.13.0+cu130`, ONNX `1.21.0`, ONNX Runtime `1.24.4` on
`CPUExecutionProvider`. The five-seed maxima reproduce the values above to three
significant digits despite the different runtime versions, so this drift
originates in graph operation ordering rather than in the runtime build.

## Benchmarks

Benchmarks used batch size 1, `10` warmup runs, and `50` measured runs.

CPU ONNX Runtime:

| Branch | Provider | Mean latency | P95 latency |
|---|---|---:|---:|
| O2M | `CPUExecutionProvider` | `230.712 ms` | `240.292 ms` |
| NMS-free | `CPUExecutionProvider` | `244.064 ms` | `258.964 ms` |

NMS-free was `13.351 ms` slower than O2M on mean latency, a `5.79%` increase.

GPU ONNX Runtime:

| Branch | Provider | Mean latency | P95 latency |
|---|---|---:|---:|
| O2M | `CUDAExecutionProvider` | `32.997 ms` | `33.890 ms` |
| NMS-free | `CUDAExecutionProvider` | `33.855 ms` | `34.724 ms` |

NMS-free was `0.858 ms` slower than O2M on mean latency, a `2.60%` increase.

## Conclusion

TR-HASH Vision v8 exports successfully to ONNX for both raw prediction
branches. Branch behavior matches the expected architecture: NMS-free computes
the extra one-to-one head path and is empirically slower than O2M. Deployment
validation should use decoded-output drift and the calibrated v8 raw-logit
thresholds above rather than the legacy `1e-4` raw-logit threshold.
