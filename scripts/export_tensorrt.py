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
) -> dict:
    """Measure inference latency of a TensorRT engine."""

    import torch

    print(f"\nBenchmarking: {engine_path}")

    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    input_tensor = torch.randn(*input_shape, dtype=torch.float32, device="cuda")
    output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")

    context.set_tensor_address(input_name, input_tensor.data_ptr())
    context.set_tensor_address(output_name, output_tensor.data_ptr())

    stream = torch.cuda.Stream()

    print(f"  Input:  {input_name} {list(input_shape)}")
    print(f"  Output: {output_name} {list(output_shape)}")

    print(f"  Warmup: {warmup} iterations...")
    for _ in range(warmup):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    print(f"  Measuring: {iterations} iterations...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        context.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / iterations) * 1000
    fps = iterations / elapsed

    results = {
        "latency_ms": round(latency_ms, 3),
        "fps": round(fps, 1),
        "iterations": iterations,
        "engine": str(engine_path),
    }

    print(f"\n  Results:")
    print(f"    Latency: {latency_ms:.3f} ms/image")
    print(f"    Throughput: {fps:.1f} FPS")

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
    )

    if args.benchmark:
        benchmark_engine(
            output,
            warmup=args.warmup,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
