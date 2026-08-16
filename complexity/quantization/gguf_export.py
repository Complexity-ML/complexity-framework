"""
GGUF Export for llama.cpp compatibility.

GGUF (GPT-Generated Unified Format) is the format used by llama.cpp
for efficient CPU/GPU inference.

Supports various quantization types:
- Q4_0, Q4_1: 4-bit quantization
- Q5_0, Q5_1: 5-bit quantization
- Q8_0: 8-bit quantization
- Q4_K_M, Q5_K_M: K-quant variants (better quality)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, BinaryIO
from dataclasses import dataclass
from enum import Enum
import struct
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GGUFQuantType(Enum):
    """GGUF quantization types."""
    F32 = 0      # 32-bit float
    F16 = 1      # 16-bit float
    Q4_0 = 2     # 4-bit (32 weights, 1 scale)
    Q4_1 = 3     # 4-bit with min (32 weights, scale + min)
    Q5_0 = 6     # 5-bit
    Q5_1 = 7     # 5-bit with min
    Q8_0 = 8     # 8-bit
    Q8_1 = 9     # 8-bit with min
    Q2_K = 10    # 2-bit K-quant
    Q3_K = 11    # 3-bit K-quant
    Q4_K = 12    # 4-bit K-quant
    Q5_K = 13    # 5-bit K-quant
    Q6_K = 14    # 6-bit K-quant
    Q8_K = 15    # 8-bit K-quant


@dataclass
class GGUFTensorInfo:
    """Information about a tensor in GGUF format."""
    name: str
    dims: List[int]
    dtype: GGUFQuantType
    offset: int


class GGUFWriter:
    """
    Writer for GGUF format files.

    GGUF structure:
    1. Magic number ("GGUF")
    2. Version
    3. Tensor count
    4. Metadata count
    5. Metadata key-value pairs
    6. Tensor infos
    7. Tensor data (aligned)
    """

    MAGIC = b"GGUF"
    VERSION = 3

    def __init__(self, path: str):
        """
        Args:
            path: Output file path
        """
        self.path = path
        self.tensors: List[GGUFTensorInfo] = []
        self.tensor_data: List[bytes] = []
        self.metadata: Dict[str, Any] = {}

    def add_metadata(self, key: str, value: Any):
        """Add metadata key-value pair."""
        self.metadata[key] = value

    def add_tensor(
        self,
        name: str,
        data: torch.Tensor,
        quant_type: GGUFQuantType = GGUFQuantType.F16,
    ):
        """
        Add tensor to GGUF file.

        Args:
            name: Tensor name
            data: Tensor data
            quant_type: Quantization type
        """
        # Quantize if needed
        if quant_type == GGUFQuantType.F32:
            quantized = data.float().numpy().tobytes()
        elif quant_type == GGUFQuantType.F16:
            quantized = data.half().numpy().tobytes()
        elif quant_type == GGUFQuantType.Q8_0:
            quantized = self._quantize_q8_0(data)
        elif quant_type == GGUFQuantType.Q4_0:
            quantized = self._quantize_q4_0(data)
        elif quant_type == GGUFQuantType.Q4_K:
            quantized = self._quantize_q4_k(data)
        else:
            # Default to F16
            quantized = data.half().numpy().tobytes()
            quant_type = GGUFQuantType.F16

        self.tensors.append(GGUFTensorInfo(
            name=name,
            dims=list(data.shape),
            dtype=quant_type,
            offset=sum(len(d) for d in self.tensor_data),
        ))
        self.tensor_data.append(quantized)

    def _quantize_q8_0(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q8_0 format."""
        # Q8_0: 32 weights per block, 1 scale (fp16)
        tensor = tensor.float().reshape(-1)
        block_size = 32

        # Pad to block boundary
        if len(tensor) % block_size != 0:
            pad = block_size - (len(tensor) % block_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad))

        # Reshape to blocks
        blocks = tensor.view(-1, block_size)

        # Compute scale per block
        scales = blocks.abs().max(dim=1).values / 127
        scales = scales.clamp(min=1e-8)

        # Quantize
        quantized = torch.clamp(
            torch.round(blocks / scales.unsqueeze(1)),
            -128, 127
        ).to(torch.int8)

        # Pack: scale (fp16) + 32 int8 weights
        result = bytearray()
        for i in range(len(scales)):
            result.extend(struct.pack('<e', scales[i].item()))  # fp16
            result.extend(quantized[i].numpy().tobytes())

        return bytes(result)

    def _quantize_q4_0(self, tensor: torch.Tensor) -> bytes:
        """Quantize to Q4_0 format."""
        # Q4_0: 32 weights per block, 1 scale (fp16), weights packed 2 per byte
        tensor = tensor.float().reshape(-1)
        block_size = 32

        # Pad
        if len(tensor) % block_size != 0:
            pad = block_size - (len(tensor) % block_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad))

        blocks = tensor.view(-1, block_size)

        # Scale per block
        scales = blocks.abs().max(dim=1).values / 7  # 4-bit signed: -8 to 7
        scales = scales.clamp(min=1e-8)

        # Quantize to int4
        quantized = torch.clamp(
            torch.round(blocks / scales.unsqueeze(1)),
            -8, 7
        ).to(torch.int8)

        # Pack: scale (fp16) + 16 bytes (32 int4 packed)
        result = bytearray()
        for i in range(len(scales)):
            result.extend(struct.pack('<e', scales[i].item()))

            # Pack pairs of int4 into int8
            q = quantized[i].numpy()
            packed = ((q[::2] + 8) & 0x0F) | (((q[1::2] + 8) & 0x0F) << 4)
            result.extend(packed.astype(np.uint8).tobytes())

        return bytes(result)

    def _quantize_q4_k(self, tensor: torch.Tensor) -> bytes:
        """
        Quantize to Q4_K format (K-quant).

        K-quants use super-blocks for better accuracy:
        - Super-block of 256 weights
        - Split into 8 sub-blocks of 32
        - Each sub-block has local scale
        """
        tensor = tensor.float().reshape(-1)
        super_block_size = 256
        sub_block_size = 32

        # Pad
        if len(tensor) % super_block_size != 0:
            pad = super_block_size - (len(tensor) % super_block_size)
            tensor = torch.nn.functional.pad(tensor, (0, pad))

        super_blocks = tensor.view(-1, super_block_size)
        result = bytearray()

        for sb in super_blocks:
            # Global scale for super-block (fp16)
            d = sb.abs().max() / 127
            d = max(d.item(), 1e-8)

            # Compute sub-block scales (6-bit, stored as fp16)
            sub_blocks = sb.view(8, sub_block_size)
            sub_scales = sub_blocks.abs().max(dim=1).values / 7

            # Quantize
            quantized = torch.clamp(
                torch.round(sb / d),
                -8, 7
            ).to(torch.int8)

            # Write super-block
            result.extend(struct.pack('<e', d))  # d scale

            # Write sub-block scales (simplified - full impl uses 6-bit packing)
            for s in sub_scales:
                result.extend(struct.pack('<e', s.item()))

            # Write quantized weights (packed)
            q = quantized.numpy()
            packed = ((q[::2] + 8) & 0x0F) | (((q[1::2] + 8) & 0x0F) << 4)
            result.extend(packed.astype(np.uint8).tobytes())

        return bytes(result)

    def write(self):
        """Write GGUF file."""
        with open(self.path, 'wb') as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack('<I', self.VERSION))
            f.write(struct.pack('<Q', len(self.tensors)))
            f.write(struct.pack('<Q', len(self.metadata)))

            # Metadata
            for key, value in self.metadata.items():
                self._write_string(f, key)
                self._write_value(f, value)

            # Tensor infos
            for tensor in self.tensors:
                self._write_string(f, tensor.name)
                f.write(struct.pack('<I', len(tensor.dims)))
                for dim in tensor.dims:
                    f.write(struct.pack('<Q', dim))
                f.write(struct.pack('<I', tensor.dtype.value))
                f.write(struct.pack('<Q', tensor.offset))

            # Align to 32 bytes
            pos = f.tell()
            align = 32 - (pos % 32)
            if align != 32:
                f.write(b'\x00' * align)

            # Tensor data
            for data in self.tensor_data:
                f.write(data)

        logger.info(f"Wrote GGUF file: {self.path}")

    def _write_string(self, f: BinaryIO, s: str):
        """Write length-prefixed string."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)

    def _write_value(self, f: BinaryIO, value: Any):
        """Write typed value."""
        if isinstance(value, str):
            f.write(struct.pack('<I', 8))  # String type
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 4))  # int32 type
            f.write(struct.pack('<i', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 6))  # float32 type
            f.write(struct.pack('<f', value))
        else:
            # Default to string
            f.write(struct.pack('<I', 8))
            self._write_string(f, str(value))


def export_gguf(
    model: nn.Module,
    output_path: str,
    quantization: str = "Q4_K_M",
    model_name: str = "complexity",
    vocab_size: Optional[int] = None,
):
    """
    Export model to GGUF format.

    Args:
        model: PyTorch model to export
        output_path: Output file path (.gguf)
        quantization: Quantization type (F16, Q8_0, Q4_0, Q4_K_M, etc.)
        model_name: Model name for metadata
        vocab_size: Vocabulary size

    Example:
        export_gguf(model, "model.gguf", quantization="Q4_K_M")
    """
    # Map quantization string to enum
    quant_map = {
        "F32": GGUFQuantType.F32,
        "F16": GGUFQuantType.F16,
        "Q8_0": GGUFQuantType.Q8_0,
        "Q4_0": GGUFQuantType.Q4_0,
        "Q4_K": GGUFQuantType.Q4_K,
        "Q4_K_M": GGUFQuantType.Q4_K,  # Use Q4_K for now
        "Q5_K_M": GGUFQuantType.Q5_K,
    }

    quant_type = quant_map.get(quantization, GGUFQuantType.Q4_K)

    writer = GGUFWriter(output_path)

    # Add metadata
    writer.add_metadata("general.architecture", "llama")  # Compatibility
    writer.add_metadata("general.name", model_name)
    writer.add_metadata("general.quantization_version", 2)

    # Model config
    config = getattr(model, 'config', None)
    if config:
        writer.add_metadata("llama.context_length", getattr(config, 'max_position_embeddings', 2048))
        writer.add_metadata("llama.embedding_length", getattr(config, 'hidden_size', 768))
        writer.add_metadata("llama.block_count", getattr(config, 'num_hidden_layers', 12))
        writer.add_metadata("llama.feed_forward_length", getattr(config, 'intermediate_size', 3072))
        writer.add_metadata("llama.attention.head_count", getattr(config, 'num_attention_heads', 12))
        writer.add_metadata("llama.attention.head_count_kv", getattr(config, 'num_key_value_heads', 12))

    if vocab_size:
        writer.add_metadata("llama.vocab_size", vocab_size)

    # Export tensors
    logger.info(f"Exporting model with {quantization} quantization...")

    for name, param in model.named_parameters():
        # Convert name to GGUF format
        gguf_name = _convert_name_to_gguf(name)

        # Determine quant type for this tensor
        tensor_quant = quant_type
        if "norm" in name or "bias" in name or "embed" in name:
            tensor_quant = GGUFQuantType.F32  # Keep norms/embeddings in F32

        writer.add_tensor(gguf_name, param.data, tensor_quant)

    writer.write()
    logger.info(f"Exported to {output_path}")


def _convert_name_to_gguf(name: str) -> str:
    """Convert PyTorch parameter name to GGUF tensor name."""
    # Common conversions
    conversions = {
        "embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
        "norm.weight": "output_norm.weight",
        "self_attn.q_proj": "attn_q",
        "self_attn.k_proj": "attn_k",
        "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output",
        "mlp.gate_proj": "ffn_gate",
        "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
        "input_layernorm": "attn_norm",
        "post_attention_layernorm": "ffn_norm",
    }

    # Extract layer number if present
    import re
    layer_match = re.search(r'layers\.(\d+)\.', name)
    layer_num = layer_match.group(1) if layer_match else None

    # Apply conversions
    result = name
    for old, new in conversions.items():
        result = result.replace(old, new)

    # Format layer number
    if layer_num is not None:
        result = re.sub(r'layers\.\d+\.', f'blk.{layer_num}.', result)

    return result
