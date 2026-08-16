"""Convert a TR-Hash ONNX model to a TensorRT engine and benchmark latency.

The TensorRT engine is compiled for the specific GPU it runs on. Each user
must run this script on their own hardware — the resulting .engine file is
not portable across GPU architectures.

FP16 support uses NVIDIA ModelOpt AutoCast to convert the ONNX model to
mixed precision before building, as required by TensorRT 11.x (which removed
the legacy BuilderFlag.FP16 in favor of strongly-typed networks).

Usage:
    python export_tensorrt.py model.onnx
    python export_tensorrt.py model.onnx --fp16
    python export_tensorrt.py model.onnx --fp16 --benchmark
    python export_tensorrt.py model.onnx --benchmark --warmup 100 --iterations 500
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    raise ImportError("TensorRT not installed. Run: pip install tensorrt")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("onnx_model", type=Path, help="Input ONNX model path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .engine path (default: derived from input name)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert to mixed-precision FP16 via ModelOpt AutoCast before building",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=1024,
        help="Max workspace size in MB (default: %(default)s)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark after building",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Benchmark warmup iterations (default: %(default)s)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Benchmark measurement iterations (default: %(default)s)",
    )
    parser.add_argument("--min-batch", type=int, default=1, help="Dynamic profile minimum batch")
    parser.add_argument("--opt-batch", type=int, default=1, help="Dynamic profile optimum batch")
    parser.add_argument("--max-batch", type=int, default=8, help="Dynamic profile maximum batch")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch used by the benchmark (must be inside the dynamic profile)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="TensorRT vs original ONNX max error (default: 1e-4 FP32, 1e-2 FP16)",
    )
    return parser.parse_args()


def convert_to_fp16(onnx_path: Path) -> Path:
    """Convert an FP32 ONNX model to mixed-precision FP16 using ModelOpt."""

    try:
        import modelopt.onnx.autocast as autocast
        import onnx
    except ImportError:
        raise ImportError(
            "ModelOpt not installed. Run: "
            "pip install nvidia-modelopt[onnx] --extra-index-url https://pypi.nvidia.com"
        )

    print("Converting to mixed-precision FP16 via ModelOpt AutoCast...")
    converted = autocast.convert_to_mixed_precision(
        onnx_path=str(onnx_path),
        low_precision_type="fp16",
        keep_io_types=True,
    )

    fp16_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
    onnx.save(converted, str(fp16_path))
    size_mb = fp16_path.stat().st_size / (1024 * 1024)
    print(f"FP16 ONNX saved: {fp16_path} ({size_mb:.1f} MB)")
    return fp16_path


def build_engine(
    onnx_path: Path,
    output_path: Path,
    *,
    max_workspace_mb: int = 1024,
    min_batch: int = 1,
    opt_batch: int = 1,
    max_batch: int = 8,
) -> Path:
    """Build a TensorRT engine from an ONNX model."""

    print(f"Loading ONNX: {onnx_path}")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX model")

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)
    print(f"  Input:  {input_tensor.name} {input_tensor.shape}")
    print(f"  Output: {output_tensor.name} {output_tensor.shape}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, max_workspace_mb * (1 << 20)
    )

    network_shape = tuple(input_tensor.shape)
    dynamic_dimensions = [index for index, value in enumerate(network_shape) if value == -1]
    if dynamic_dimensions:
        if dynamic_dimensions != [0]:
            raise ValueError(
                "only a dynamic batch dimension is supported; "
                f"received input shape {network_shape}"
            )
        if not 0 < min_batch <= opt_batch <= max_batch:
            raise ValueError("dynamic batches must satisfy 0 < min <= opt <= max")
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            (min_batch, *network_shape[1:]),
            (opt_batch, *network_shape[1:]),
            (max_batch, *network_shape[1:]),
        )
        config.add_optimization_profile(profile)
        print(f"  Dynamic batch profile: min={min_batch} opt={opt_batch} max={max_batch}")

    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Engine saved: {output_path} ({size_mb:.1f} MB)")

    return output_path


def benchmark_engine(
    engine_path: Path,
    *,
    warmup: int = 50,
    iterations: int = 200,
    batch_size: int = 1,
    reference_onnx: Path | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """Verify the engine and measure raw-network inference latency."""

    import torch

    print(f"\nBenchmarking: {engine_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    tensor_names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    input_names = [
        name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]
    output_names = [
        name for name in tensor_names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
    ]
    if len(input_names) != 1 or len(output_names) != 1:
        raise ValueError(
            "the benchmark expects exactly one input and one output, got "
            f"{len(input_names)} inputs and {len(output_names)} outputs"
        )
    input_name, output_name = input_names[0], output_names[0]
    engine_input_shape = tuple(engine.get_tensor_shape(input_name))

    stream = torch.cuda.Stream()
    if -1 in engine_input_shape:
        input_shape = (batch_size, *engine_input_shape[1:])
        context.set_optimization_profile_async(0, stream.cuda_stream)
        if not context.set_input_shape(input_name, input_shape):
            raise ValueError(f"batch size {batch_size} is outside the engine profile")
    else:
        input_shape = engine_input_shape
        if batch_size != input_shape[0]:
            raise ValueError(
                f"static engine batch is {input_shape[0]}, received --batch-size {batch_size}"
            )
    output_shape = tuple(context.get_tensor_shape(output_name))
    if any(dimension < 0 for dimension in output_shape):
        raise RuntimeError(f"TensorRT could not resolve output shape {output_shape}")

    torch_dtypes = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    try:
        input_dtype = torch_dtypes[engine.get_tensor_dtype(input_name)]
        output_dtype = torch_dtypes[engine.get_tensor_dtype(output_name)]
    except KeyError as error:
        raise TypeError(f"unsupported TensorRT I/O dtype: {error.args[0]}") from error

    with torch.cuda.stream(stream):
        input_tensor = torch.randn(*input_shape, dtype=input_dtype, device="cuda")
        output_tensor = torch.empty(*output_shape, dtype=output_dtype, device="cuda")
    stream.synchronize()

    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    print(f"  Input:  {input_name} {list(input_shape)}")
    print(f"  Output: {output_name} {list(output_shape)}")

    print(f"  Warmup: {warmup} iterations...")
    for _ in range(warmup):
        if context.execute_async_v3(stream_handle=stream.cuda_stream) is False:
            raise RuntimeError("TensorRT warmup execution failed")
    stream.synchronize()

    parity = None
    if reference_onnx is not None:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "TensorRT parity requires numpy and onnxruntime; install the export extra"
            ) from error
        reference_session = ort.InferenceSession(
            str(reference_onnx), providers=["CPUExecutionProvider"]
        )
        reference_input = reference_session.get_inputs()[0].name
        reference_output = reference_session.get_outputs()[0].name
        input_array = input_tensor.detach().float().cpu().numpy()
        expected = reference_session.run([reference_output], {reference_input: input_array})[0]
        actual = output_tensor.detach().float().cpu().numpy()
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"TensorRT output shape {actual.shape} differs from ONNX {expected.shape}"
            )
        absolute_difference = np.abs(actual - expected)
        max_difference = float(absolute_difference.max())
        mean_difference = float(absolute_difference.mean())
        if max_difference > tolerance:
            raise RuntimeError(
                f"TensorRT parity failed: max_diff={max_difference:.3e} > {tolerance:.3e}"
            )
        parity = {
            "max_difference": max_difference,
            "mean_difference": mean_difference,
            "tolerance": tolerance,
        }
        print(
            f"  Parity: max_diff={max_difference:.3e}, "
            f"mean_diff={mean_difference:.3e} [PASS]"
        )

    print(f"  Measuring: {iterations} iterations...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        if context.execute_async_v3(stream_handle=stream.cuda_stream) is False:
            raise RuntimeError("TensorRT benchmark execution failed")
    stream.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms_per_batch = (elapsed / iterations) * 1000
    latency_ms_per_image = latency_ms_per_batch / batch_size
    images_per_second = batch_size * iterations / elapsed

    results = {
        "raw_latency_ms_per_batch": round(latency_ms_per_batch, 3),
        "raw_latency_ms_per_image": round(latency_ms_per_image, 3),
        "raw_images_per_second": round(images_per_second, 1),
        "batch_size": batch_size,
        "iterations": iterations,
        "engine": str(engine_path),
        "parity": parity,
    }

    print("\n  Raw network results (pre/post-processing excluded):")
    print(f"    Latency: {latency_ms_per_batch:.3f} ms/batch")
    print(f"    Latency: {latency_ms_per_image:.3f} ms/image")
    print(f"    Throughput: {images_per_second:.1f} images/s")

    return results


def main() -> None:
    args = parse_args()

    onnx_path = args.onnx_model

    # FP16 conversion via ModelOpt AutoCast (TRT 11.x workflow)
    if args.fp16:
        onnx_path = convert_to_fp16(onnx_path)

    # Output path
    output = args.output
    if output is None:
        suffix = "_fp16" if args.fp16 else ""
        output = args.onnx_model.with_name(args.onnx_model.stem + suffix + ".engine")

    build_engine(
        onnx_path,
        output,
        max_workspace_mb=args.workspace,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
    )

    if args.benchmark:
        benchmark_engine(
            output,
            warmup=args.warmup,
            iterations=args.iterations,
            batch_size=args.batch_size,
            reference_onnx=args.onnx_model,
            tolerance=(1e-2 if args.fp16 else 1e-4)
            if args.tolerance is None
            else args.tolerance,
        )


if __name__ == "__main__":
    main()
