"""
Tokenization utilities for LLM training.

Provides unified interface for different tokenizers:
- BPE (Byte Pair Encoding)
- SentencePiece
- Tiktoken (OpenAI's tokenizer)
- HuggingFace tokenizers
- Chat templates (Complexity/ChatML/Nemotron style)
- Tool calling support
- Reasoning/thinking mode support
- Multi-modal tokens (vision, audio, robotics)

INL Complexity Format Features:
- 2048 reserved special tokens (0-2047)
- Native tool calling with <|tool|> tags
- Chain-of-thought <|reason|> tags
- Multi-modal support <|vision|>, <|audio|>, <|action|>
- Robotics integration <|state|>, <|goal|>, <|trajectory|>
- Code execution <|exec|>, <|result|>
- Memory/context <|memory|>, <|retrieve|>
"""

import torch
from typing import List, Optional, Union, Dict, Any, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import re


class Tokenizer(ABC):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Vocabulary size."""
        pass

    @property
    @abstractmethod
    def bos_token_id(self) -> Optional[int]:
        """Beginning of sequence token ID."""
        pass

    @property
    @abstractmethod
    def eos_token_id(self) -> Optional[int]:
        """End of sequence token ID."""
        pass

    @property
    @abstractmethod
    def pad_token_id(self) -> Optional[int]:
        """Padding token ID."""
        pass

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Encode multiple texts."""
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_lists: List[List[int]]) -> List[str]:
        """Decode multiple token lists."""
        return [self.decode(tokens) for tokens in token_lists]


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding tokenizer.

    Simple BPE implementation for educational purposes.
    For production, use tiktoken or sentencepiece.
    """

    def __init__(
        self,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            vocab_path: Path to vocabulary file (JSON)
            merges_path: Path to merges file
            special_tokens: Dictionary of special tokens
        """
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.vocab = json.load(f)

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

        # Load merges
        self.merges = {}
        with open(merges_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) == 2:
                    self.merges[(parts[0], parts[1])] = i

        # Special tokens
        self._special_tokens = special_tokens or {
            "<|endoftext|>": len(self.vocab),
            "<|padding|>": len(self.vocab) + 1,
        }

        self._bos_id = self._special_tokens.get("<|startoftext|>")
        self._eos_id = self._special_tokens.get("<|endoftext|>", len(self.vocab))
        self._pad_id = self._special_tokens.get("<|padding|>", len(self.vocab) + 1)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + len(self._special_tokens)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_id

    def encode(self, text: str) -> List[int]:
        """Encode text using BPE."""
        # Convert to bytes
        tokens = list(text.encode('utf-8'))

        # Apply BPE merges
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

            # Find best merge
            best_pair = None
            best_rank = float('inf')

            for pair in pairs:
                pair_str = (chr(pair[0]) if isinstance(pair[0], int) else pair[0],
                           chr(pair[1]) if isinstance(pair[1], int) else pair[1])
                if pair_str in self.merges:
                    if self.merges[pair_str] < best_rank:
                        best_rank = self.merges[pair_str]
                        best_pair = pair

            if best_pair is None:
                break

            # Apply merge
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                    merged = chr(tokens[i]) + chr(tokens[i+1]) if isinstance(tokens[i], int) else tokens[i] + tokens[i+1]
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        # Convert to IDs
        ids = []
        for token in tokens:
            if isinstance(token, int):
                token = chr(token)
            if token in self.vocab:
                ids.append(self.vocab[token])

        return ids

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        text_parts = []
        for token_id in tokens:
            if token_id in self.inverse_vocab:
                text_parts.append(self.inverse_vocab[token_id])
            elif token_id == self._eos_id:
                break  # Stop at EOS
        return ''.join(text_parts)


class SentencePieceTokenizer(Tokenizer):
    """
    SentencePiece tokenizer wrapper.

    Used by Llama, Mistral, and other models.
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to SentencePiece model file (.model)
        """
        try:
            import sentencepiece as spm
        except ImportError:
            raise ImportError("sentencepiece required. Install with: pip install sentencepiece")

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.sp.bos_id() if self.sp.bos_id() >= 0 else None

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.sp.eos_id() if self.sp.eos_id() >= 0 else None

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.sp.pad_id() if self.sp.pad_id() >= 0 else None

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self.sp.Encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.Decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch encode."""
        return self.sp.Encode(texts)


class TiktokenWrapper(Tokenizer):
    """
    Tiktoken tokenizer wrapper (OpenAI's tokenizer).

    Used by GPT-4, Claude, and other modern LLMs.
    Very fast BPE implementation.
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",  # GPT-4 encoding
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            encoding_name: Tiktoken encoding name
                - "cl100k_base": GPT-4, ChatGPT
                - "p50k_base": GPT-3
                - "r50k_base": Older models
            special_tokens: Additional special tokens
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError("tiktoken required. Install with: pip install tiktoken")

        self.encoding = tiktoken.get_encoding(encoding_name)

        # Add special tokens if needed
        self._special_tokens = special_tokens or {}
        self._eos_id = self.encoding.eot_token

    @property
    def vocab_size(self) -> int:
        return self.encoding.n_vocab

    @property
    def bos_token_id(self) -> Optional[int]:
        return None  # Tiktoken doesn't have BOS by default

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return None  # Tiktoken doesn't have PAD by default

    def encode(self, text: str) -> List[int]:
        """Encode text."""
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens."""
        return self.encoding.decode(tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """Batch encode (optimized)."""
        return self.encoding.encode_batch(texts)


class HuggingFaceTokenizer(Tokenizer):
    """
    Wrapper for HuggingFace tokenizers.

    Supports any tokenizer from the transformers library.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_fast: bool = True,
        **kwargs,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model name or path
            use_fast: Use fast Rust tokenizer if available
            **kwargs: Additional arguments for AutoTokenizer
        """
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise ImportError("transformers required. Install with: pip install transformers")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=use_fast,
            **kwargs,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id

    def encode(self, text: str) -> List[int]:
        """Encode text."""
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens: List[int]) -> str:
        """Decode tokens."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def __call__(
        self,
        text: Union[str, List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Full tokenization with padding and attention masks."""
        return self.tokenizer(text, return_tensors="pt", **kwargs)


def get_tokenizer(
    tokenizer_type: str,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tokenizer:
    """
    Get tokenizer by type.

    Args:
        tokenizer_type: One of "tiktoken", "sentencepiece", "huggingface"
        model_path: Path to tokenizer model (for sentencepiece/huggingface)
        **kwargs: Additional arguments

    Returns:
        Configured tokenizer
    """
    if tokenizer_type == "tiktoken":
        return TiktokenWrapper(
            encoding_name=kwargs.get("encoding_name", "cl100k_base"),
        )
    elif tokenizer_type == "sentencepiece":
        if model_path is None:
            raise ValueError("model_path required for sentencepiece")
        return SentencePieceTokenizer(model_path)
    elif tokenizer_type == "huggingface":
        if model_path is None:
            raise ValueError("model_path required for huggingface")
        return HuggingFaceTokenizer(model_path, **kwargs)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


# =============================================================================
# INL Complexity Special Tokens
# =============================================================================

@dataclass
class ComplexityTokens:
    """
    INL Complexity format special tokens.

    Reserves 2048 special tokens (0-2047) organized by category:
    - 0-31: Core tokens (unk, bos, eos, pad, masks)
    - 32-63: Chat/conversation tokens
    - 64-127: Tool calling tokens
    - 128-191: Reasoning/thinking tokens
    - 192-255: Code execution tokens
    - 256-383: Multi-modal tokens (vision, audio, video)
    - 384-511: Robotics tokens (state, action, trajectory)
    - 512-767: Memory/retrieval tokens
    - 768-1023: Task-specific tokens
    - 1024-2047: Reserved for future use
    """

    # =========================================================================
    # Core tokens (0-31)
    # =========================================================================
    unk_token: str = "<|unk|>"              # 0
    bos_token: str = "<|begin|>"            # 1  - INL style
    eos_token: str = "<|end|>"              # 2
    pad_token: str = "<|pad|>"              # 3
    sep_token: str = "<|sep|>"              # 4
    cls_token: str = "<|cls|>"              # 5
    mask_token: str = "<|mask|>"            # 6
    newline_token: str = "<|n|>"            # 7

    # =========================================================================
    # Chat/Conversation tokens (32-63) - Complexity style
    # =========================================================================
    turn_start: str = "<|turn|>"            # 32 - Start of turn
    turn_end: str = "<|/turn|>"             # 33
    system_start: str = "<|system|>"        # 34
    system_end: str = "<|/system|>"         # 35
    user_start: str = "<|user|>"            # 36
    user_end: str = "<|/user|>"             # 37
    assistant_start: str = "<|assistant|>"  # 38
    assistant_end: str = "<|/assistant|>"   # 39
    context_start: str = "<|context|>"      # 40
    context_end: str = "<|/context|>"       # 41

    # =========================================================================
    # Tool calling tokens (64-127) - Complexity style
    # =========================================================================
    tools_start: str = "<|tools|>"          # 64
    tools_end: str = "<|/tools|>"           # 65
    tool_def: str = "<|tool_def|>"          # 66 - Tool definition
    tool_call: str = "<|call|>"             # 67 - Tool invocation
    tool_call_end: str = "<|/call|>"        # 68
    tool_args: str = "<|args|>"             # 69
    tool_args_end: str = "<|/args|>"        # 70
    tool_result: str = "<|result|>"         # 71
    tool_result_end: str = "<|/result|>"    # 72
    tool_error: str = "<|error|>"           # 73
    tool_error_end: str = "<|/error|>"      # 74

    # =========================================================================
    # Reasoning/Thinking tokens (128-191) - Complexity CoT
    # =========================================================================
    reason_start: str = "<|reason|>"        # 128 - Chain of thought
    reason_end: str = "<|/reason|>"         # 129
    step_start: str = "<|step|>"            # 130 - Reasoning step
    step_end: str = "<|/step|>"             # 131
    conclude: str = "<|conclude|>"          # 132 - Final conclusion
    conclude_end: str = "<|/conclude|>"     # 133
    reflect: str = "<|reflect|>"            # 134 - Self-reflection
    reflect_end: str = "<|/reflect|>"       # 135
    verify: str = "<|verify|>"              # 136 - Verification step
    verify_end: str = "<|/verify|>"         # 137
    plan: str = "<|plan|>"                  # 138 - Planning
    plan_end: str = "<|/plan|>"             # 139

    # =========================================================================
    # Code execution tokens (192-255)
    # =========================================================================
    code_start: str = "<|code|>"            # 192
    code_end: str = "<|/code|>"             # 193
    exec_start: str = "<|exec|>"            # 194 - Execute code
    exec_end: str = "<|/exec|>"             # 195
    output_start: str = "<|output|>"        # 196
    output_end: str = "<|/output|>"         # 197
    lang_python: str = "<|python|>"         # 198
    lang_bash: str = "<|bash|>"             # 199
    lang_sql: str = "<|sql|>"               # 200
    lang_json: str = "<|json|>"             # 201

    # =========================================================================
    # Multi-modal tokens (256-383) - Vision, Audio, Video
    # =========================================================================
    # Vision (256-287)
    vision_start: str = "<|vision|>"        # 256
    vision_end: str = "<|/vision|>"         # 257
    image: str = "<|image|>"                # 258
    image_end: str = "<|/image|>"           # 259
    patch: str = "<|patch|>"                # 260 - Image patch token
    bbox: str = "<|bbox|>"                  # 261 - Bounding box
    segment: str = "<|segment|>"            # 262 - Segmentation

    # Audio (288-319)
    audio_start: str = "<|audio|>"          # 288
    audio_end: str = "<|/audio|>"           # 289
    speech: str = "<|speech|>"              # 290
    music: str = "<|music|>"                # 291
    sound: str = "<|sound|>"                # 292
    transcribe: str = "<|transcribe|>"      # 293

    # Video (320-351)
    video_start: str = "<|video|>"          # 320
    video_end: str = "<|/video|>"           # 321
    frame: str = "<|frame|>"                # 322
    clip: str = "<|clip|>"                  # 323
    temporal: str = "<|temporal|>"          # 324

    # =========================================================================
    # Robotics tokens (384-511) - INL Embodied AI
    # =========================================================================
    # State/Observation (384-415)
    state_start: str = "<|state|>"          # 384
    state_end: str = "<|/state|>"           # 385
    proprio: str = "<|proprio|>"            # 386 - Proprioception
    sensor: str = "<|sensor|>"              # 387
    observation: str = "<|obs|>"            # 388

    # Action (416-447)
    action_start: str = "<|action|>"        # 416
    action_end: str = "<|/action|>"         # 417
    gripper: str = "<|gripper|>"            # 418
    joint: str = "<|joint|>"                # 419
    velocity: str = "<|velocity|>"          # 420
    torque: str = "<|torque|>"              # 421

    # Goal/Task (448-479)
    goal_start: str = "<|goal|>"            # 448
    goal_end: str = "<|/goal|>"             # 449
    task: str = "<|task|>"                  # 450
    subtask: str = "<|subtask|>"            # 451
    waypoint: str = "<|waypoint|>"          # 452

    # Trajectory (480-511)
    trajectory_start: str = "<|trajectory|>"  # 480
    trajectory_end: str = "<|/trajectory|>"   # 481
    path: str = "<|path|>"                    # 482
    motion: str = "<|motion|>"                # 483

    # =========================================================================
    # Memory/Retrieval tokens (512-767)
    # =========================================================================
    memory_start: str = "<|memory|>"        # 512
    memory_end: str = "<|/memory|>"         # 513
    retrieve: str = "<|retrieve|>"          # 514 - RAG retrieval
    retrieve_end: str = "<|/retrieve|>"     # 515
    store: str = "<|store|>"                # 516 - Store in memory
    store_end: str = "<|/store|>"           # 517
    forget: str = "<|forget|>"              # 518
    recall: str = "<|recall|>"              # 519
    cite: str = "<|cite|>"                  # 520 - Citation
    cite_end: str = "<|/cite|>"             # 521
    source: str = "<|source|>"              # 522
    source_end: str = "<|/source|>"         # 523

    # =========================================================================
    # Task tokens (768-1023)
    # =========================================================================
    task_qa: str = "<|qa|>"                 # 768 - Q&A task
    task_summarize: str = "<|summarize|>"   # 769
    task_translate: str = "<|translate|>"   # 770
    task_classify: str = "<|classify|>"     # 771
    task_generate: str = "<|generate|>"     # 772
    task_edit: str = "<|edit|>"             # 773
    task_analyze: str = "<|analyze|>"       # 774

    # Reserved tokens count
    num_reserved: int = 2048

    def get_token_map(self) -> Dict[str, int]:
        """Get mapping of special tokens to IDs."""
        tokens = {
            # Core (0-31)
            self.unk_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.pad_token: 3,
            self.sep_token: 4,
            self.cls_token: 5,
            self.mask_token: 6,
            self.newline_token: 7,

            # Chat (32-63)
            self.turn_start: 32,
            self.turn_end: 33,
            self.system_start: 34,
            self.system_end: 35,
            self.user_start: 36,
            self.user_end: 37,
            self.assistant_start: 38,
            self.assistant_end: 39,
            self.context_start: 40,
            self.context_end: 41,

            # Tools (64-127)
            self.tools_start: 64,
            self.tools_end: 65,
            self.tool_def: 66,
            self.tool_call: 67,
            self.tool_call_end: 68,
            self.tool_args: 69,
            self.tool_args_end: 70,
            self.tool_result: 71,
            self.tool_result_end: 72,
            self.tool_error: 73,
            self.tool_error_end: 74,

            # Reasoning (128-191)
            self.reason_start: 128,
            self.reason_end: 129,
            self.step_start: 130,
            self.step_end: 131,
            self.conclude: 132,
            self.conclude_end: 133,
            self.reflect: 134,
            self.reflect_end: 135,
            self.verify: 136,
            self.verify_end: 137,
            self.plan: 138,
            self.plan_end: 139,

            # Code (192-255)
            self.code_start: 192,
            self.code_end: 193,
            self.exec_start: 194,
            self.exec_end: 195,
            self.output_start: 196,
            self.output_end: 197,
            self.lang_python: 198,
            self.lang_bash: 199,
            self.lang_sql: 200,
            self.lang_json: 201,

            # Vision (256-287)
            self.vision_start: 256,
            self.vision_end: 257,
            self.image: 258,
            self.image_end: 259,
            self.patch: 260,
            self.bbox: 261,
            self.segment: 262,

            # Audio (288-319)
            self.audio_start: 288,
            self.audio_end: 289,
            self.speech: 290,
            self.music: 291,
            self.sound: 292,
            self.transcribe: 293,

            # Video (320-351)
            self.video_start: 320,
            self.video_end: 321,
            self.frame: 322,
            self.clip: 323,
            self.temporal: 324,

            # Robotics - State (384-415)
            self.state_start: 384,
            self.state_end: 385,
            self.proprio: 386,
            self.sensor: 387,
            self.observation: 388,

            # Robotics - Action (416-447)
            self.action_start: 416,
            self.action_end: 417,
            self.gripper: 418,
            self.joint: 419,
            self.velocity: 420,
            self.torque: 421,

            # Robotics - Goal (448-479)
            self.goal_start: 448,
            self.goal_end: 449,
            self.task: 450,
            self.subtask: 451,
            self.waypoint: 452,

            # Robotics - Trajectory (480-511)
            self.trajectory_start: 480,
            self.trajectory_end: 481,
            self.path: 482,
            self.motion: 483,

            # Memory (512-767)
            self.memory_start: 512,
            self.memory_end: 513,
            self.retrieve: 514,
            self.retrieve_end: 515,
            self.store: 516,
            self.store_end: 517,
            self.forget: 518,
            self.recall: 519,
            self.cite: 520,
            self.cite_end: 521,
            self.source: 522,
            self.source_end: 523,

            # Tasks (768-1023)
            self.task_qa: 768,
            self.task_summarize: 769,
            self.task_translate: 770,
            self.task_classify: 771,
            self.task_generate: 772,
            self.task_edit: 773,
            self.task_analyze: 774,
        }

        # Add reserved tokens for gaps
        used_ids = set(tokens.values())
        reserved_idx = 0
        for i in range(self.num_reserved):
            if i not in used_ids:
                tokens[f"<|reserved_{reserved_idx}|>"] = i
                reserved_idx += 1

        return tokens

    @classmethod
    def for_robotics(cls) -> "ComplexityTokens":
        """Get tokens optimized for robotics applications."""
        return cls()

    @classmethod
    def for_multimodal(cls) -> "ComplexityTokens":
        """Get tokens optimized for multi-modal applications."""
        return cls()


# Legacy alias for backward compatibility
SpecialTokens = ComplexityTokens


# =============================================================================
# Chat Template - INL Complexity Format
# =============================================================================

@dataclass
class Message:
    """Chat message for Complexity format."""
    role: str  # system, user, assistant, tool
    content: str
    name: Optional[str] = None  # For tool messages
    tool_calls: Optional[List[Dict[str, Any]]] = None
    reasoning: Optional[str] = None  # Chain-of-thought reasoning
    steps: Optional[List[str]] = None  # Reasoning steps
    conclusion: Optional[str] = None  # Final conclusion


@dataclass
class ComplexityTemplate:
    """
    INL Complexity chat template.

    Native format for the Complexity framework with:
    - Structured reasoning (<|reason|>, <|step|>, <|conclude|>)
    - Tool calling (<|call|>, <|args|>, <|result|>)
    - Multi-modal support (<|vision|>, <|audio|>, <|action|>)
    - Robotics integration (<|state|>, <|goal|>, <|trajectory|>)

    Also supports legacy formats for compatibility:
    - chatml: OpenAI ChatML format
    - llama: Llama/Mistral instruction format
    """
    format: str = "complexity"  # complexity, chatml, llama
    tokens: ComplexityTokens = field(default_factory=ComplexityTokens)
    enable_reasoning: bool = False
    enable_steps: bool = False  # Show step-by-step reasoning

    def apply(
        self,
        messages: List[Message],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: List of chat messages
            add_generation_prompt: Add prompt for model generation

        Returns:
            Formatted string
        """
        if self.format == "complexity":
            return self._apply_complexity(messages, add_generation_prompt)
        elif self.format == "chatml":
            return self._apply_chatml(messages, add_generation_prompt)
        elif self.format == "llama":
            return self._apply_llama(messages, add_generation_prompt)
        else:
            raise ValueError(f"Unknown format: {self.format}")

    def _apply_complexity(
        self,
        messages: List[Message],
        add_generation_prompt: bool,
    ) -> str:
        """
        INL Complexity native format.

        Structure:
        <|begin|>
        <|system|>System prompt<|/system|>
        <|turn|>
        <|user|>User message<|/user|>
        <|assistant|>
        <|reason|>
        <|step|>Step 1<|/step|>
        <|step|>Step 2<|/step|>
        <|conclude|>Conclusion<|/conclude|>
        <|/reason|>
        Response content
        <|/assistant|>
        <|/turn|>
        <|end|>
        """
        t = self.tokens
        result = [t.bos_token, "\n"]

        # System message first
        for msg in messages:
            if msg.role == "system":
                result.append(f"{t.system_start}\n{msg.content}\n{t.system_end}\n")
                break

        # Conversation turns
        for msg in messages:
            if msg.role == "system":
                continue

            result.append(f"{t.turn_start}\n")

            if msg.role == "user":
                result.append(f"{t.user_start}\n{msg.content}\n{t.user_end}\n")

            elif msg.role == "assistant":
                result.append(f"{t.assistant_start}\n")

                # Add reasoning if present
                if msg.reasoning or msg.steps:
                    result.append(f"{t.reason_start}\n")

                    if msg.steps:
                        for step in msg.steps:
                            result.append(f"{t.step_start}{step}{t.step_end}\n")

                    if msg.reasoning:
                        result.append(msg.reasoning)
                        result.append("\n")

                    if msg.conclusion:
                        result.append(f"{t.conclude}{msg.conclusion}{t.conclude_end}\n")

                    result.append(f"{t.reason_end}\n")

                # Add tool calls if present
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        result.append(f"{t.tool_call}\n")
                        result.append(f"{t.tool_args}\n{json.dumps(tc, indent=2)}\n{t.tool_args_end}\n")
                        result.append(f"{t.tool_call_end}\n")

                result.append(f"{msg.content}\n{t.assistant_end}\n")

            elif msg.role == "tool":
                result.append(f"{t.tool_result}\n{msg.content}\n{t.tool_result_end}\n")

            result.append(f"{t.turn_end}\n")

        # Add generation prompt
        if add_generation_prompt:
            result.append(f"{t.turn_start}\n{t.assistant_start}\n")
            if self.enable_reasoning:
                result.append(f"{t.reason_start}\n")
                if self.enable_steps:
                    result.append(f"{t.step_start}")

        return "".join(result)

    def _apply_chatml(
        self,
        messages: List[Message],
        add_generation_prompt: bool,
    ) -> str:
        """ChatML format (OpenAI style) using Complexity tokens."""
        t = self.tokens
        result = []

        for msg in messages:
            # Use turn markers as im_start/im_end equivalent
            result.append(f"{t.turn_start}{msg.role}\n{msg.content}{t.turn_end}\n")

        if add_generation_prompt:
            result.append(f"{t.turn_start}assistant\n")
            if self.enable_reasoning:
                result.append(f"{t.reason_start}\n")

        return "".join(result)

    def _apply_llama(
        self,
        messages: List[Message],
        add_generation_prompt: bool,
    ) -> str:
        """Llama/Mistral instruction format."""
        t = self.tokens
        result = [t.bos_token]

        system_msg = None
        for msg in messages:
            if msg.role == "system":
                system_msg = msg.content
                break

        for msg in messages:
            if msg.role == "system":
                continue
            elif msg.role == "user":
                if system_msg:
                    result.append(f"[INST]{t.system_start}\n{system_msg}\n{t.system_end}\n\n{msg.content}[/INST]")
                    system_msg = None
                else:
                    result.append(f"[INST]{msg.content}[/INST]")
            elif msg.role == "assistant":
                result.append(f" {msg.content} {t.eos_token}")

        if add_generation_prompt:
            result.append(" ")

        return "".join(result)

    def parse_response(self, text: str) -> Message:
        """
        Parse model response into Message.

        Extracts reasoning, steps, tool calls, and content.
        """
        t = self.tokens

        reasoning = None
        steps = []
        conclusion = None
        tool_calls = []
        content = text

        # Extract reasoning block
        reason_pattern = re.compile(
            f"{re.escape(t.reason_start)}(.*?){re.escape(t.reason_end)}",
            re.DOTALL
        )
        reason_match = reason_pattern.search(text)
        if reason_match:
            reason_block = reason_match.group(1)

            # Extract steps
            step_pattern = re.compile(
                f"{re.escape(t.step_start)}(.*?){re.escape(t.step_end)}",
                re.DOTALL
            )
            steps = [m.group(1).strip() for m in step_pattern.finditer(reason_block)]

            # Extract conclusion
            conclude_pattern = re.compile(
                f"{re.escape(t.conclude)}(.*?){re.escape(t.conclude_end)}",
                re.DOTALL
            )
            conclude_match = conclude_pattern.search(reason_block)
            if conclude_match:
                conclusion = conclude_match.group(1).strip()

            # Remaining reasoning text
            reasoning = step_pattern.sub("", reason_block)
            reasoning = conclude_pattern.sub("", reasoning).strip()
            if not reasoning:
                reasoning = None

            content = reason_pattern.sub("", content)

        # Extract tool calls
        tool_pattern = re.compile(
            f"{re.escape(t.tool_call)}(.*?){re.escape(t.tool_call_end)}",
            re.DOTALL
        )
        for match in tool_pattern.finditer(text):
            args_pattern = re.compile(
                f"{re.escape(t.tool_args)}(.*?){re.escape(t.tool_args_end)}",
                re.DOTALL
            )
            args_match = args_pattern.search(match.group(1))
            if args_match:
                try:
                    tc = json.loads(args_match.group(1).strip())
                    tool_calls.append(tc)
                except json.JSONDecodeError:
                    pass
        content = tool_pattern.sub("", content)

        # Clean up content
        content = content.strip()
        for token in [t.assistant_end, t.turn_end, t.eos_token]:
            if content.endswith(token):
                content = content[:-len(token)].strip()

        return Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            reasoning=reasoning,
            steps=steps if steps else None,
            conclusion=conclusion,
        )

    def format_for_robotics(
        self,
        state: Dict[str, Any],
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> str:
        """
        Format input for robotics model.

        Args:
            state: Robot state dict (proprio, vision, etc.)
            goal: Goal description
            instruction: Natural language instruction

        Returns:
            Formatted prompt
        """
        t = self.tokens
        result = [t.bos_token, "\n"]

        # Add state
        result.append(f"{t.state_start}\n")
        if "proprio" in state:
            result.append(f"{t.proprio}{json.dumps(state['proprio'])}\n")
        if "vision" in state:
            result.append(f"{t.vision_start}{t.image}{t.vision_end}\n")
        result.append(f"{t.state_end}\n")

        # Add goal
        if goal:
            result.append(f"{t.goal_start}\n{goal}\n{t.goal_end}\n")

        # Add instruction as user message
        if instruction:
            result.append(f"{t.user_start}\n{instruction}\n{t.user_end}\n")

        # Prompt for action
        result.append(f"{t.assistant_start}\n{t.action_start}\n")

        return "".join(result)

    def parse_robotics_action(self, text: str) -> Dict[str, Any]:
        """Parse action from model output."""
        t = self.tokens

        action_pattern = re.compile(
            f"{re.escape(t.action_start)}(.*?){re.escape(t.action_end)}",
            re.DOTALL
        )
        match = action_pattern.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        return {}


# Legacy alias
ChatTemplate = ComplexityTemplate


# =============================================================================
# INL Complexity Tokenizer
# =============================================================================

class ComplexityTokenizer:
    """
    INL Complexity tokenizer with native framework integration.

    Features:
    - Chat template formatting (Complexity native format)
    - Structured reasoning with steps
    - Tool calling support
    - Robotics action encoding
    - Multi-modal token handling
    - Memory/retrieval tokens
    """

    def __init__(
        self,
        base_tokenizer: Tokenizer,
        template: Optional[ComplexityTemplate] = None,
        tokens: Optional[ComplexityTokens] = None,
    ):
        """
        Args:
            base_tokenizer: Underlying tokenizer (BPE, SentencePiece, etc.)
            template: Chat template configuration
            tokens: Special tokens configuration
        """
        self.tokenizer = base_tokenizer
        self.tokens = tokens or ComplexityTokens()
        self.template = template or ComplexityTemplate(tokens=self.tokens)

        # Build special token ID map
        self._token_ids = self.tokens.get_token_map()
        self._id_to_token = {v: k for k, v in self._token_ids.items()}

    def encode_chat(
        self,
        messages: List[Union[Message, Dict[str, str]]],
        add_generation_prompt: bool = True,
        enable_reasoning: bool = False,
        enable_steps: bool = False,
    ) -> List[int]:
        """
        Encode chat messages to token IDs.

        Args:
            messages: List of messages (Message objects or dicts)
            add_generation_prompt: Add generation prompt
            enable_reasoning: Enable reasoning mode (<|reason|>)
            enable_steps: Enable step-by-step mode (<|step|>)

        Returns:
            Token IDs
        """
        # Convert dicts to Message objects
        msgs = []
        for m in messages:
            if isinstance(m, dict):
                # Handle both old 'thinking' and new 'reasoning' keys
                if 'thinking' in m and 'reasoning' not in m:
                    m = dict(m)
                    m['reasoning'] = m.pop('thinking')
                msgs.append(Message(**{k: v for k, v in m.items()
                                       if k in Message.__dataclass_fields__}))
            else:
                msgs.append(m)

        # Apply template
        self.template.enable_reasoning = enable_reasoning
        self.template.enable_steps = enable_steps
        text = self.template.apply(msgs, add_generation_prompt)

        return self.tokenizer.encode(text)

    def decode_chat(
        self,
        tokens: List[int],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode tokens to text."""
        text = self.tokenizer.decode(tokens)

        if skip_special_tokens:
            for token in self._token_ids.keys():
                text = text.replace(token, "")

        return text

    def parse_response(self, text: str) -> Message:
        """Parse model response into structured Message."""
        return self.template.parse_response(text)

    def encode_robotics(
        self,
        state: Dict[str, Any],
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[int]:
        """
        Encode robotics input.

        Args:
            state: Robot state (proprio, vision, etc.)
            goal: Goal description
            instruction: Natural language instruction

        Returns:
            Token IDs
        """
        text = self.template.format_for_robotics(state, goal, instruction)
        return self.tokenizer.encode(text)

    def decode_action(self, text: str) -> Dict[str, Any]:
        """Decode robotics action from model output."""
        return self.template.parse_robotics_action(text)

    def get_token_id(self, token: str) -> Optional[int]:
        """Get ID for a special token."""
        return self._token_ids.get(token)

    def get_token(self, token_id: int) -> Optional[str]:
        """Get token string from ID."""
        return self._id_to_token.get(token_id)

    # =========================================================================
    # Token ID properties for common tokens
    # =========================================================================

    @property
    def bos_id(self) -> int:
        return self._token_ids[self.tokens.bos_token]

    @property
    def eos_id(self) -> int:
        return self._token_ids[self.tokens.eos_token]

    @property
    def pad_id(self) -> int:
        return self._token_ids[self.tokens.pad_token]

    @property
    def reason_start_id(self) -> int:
        return self._token_ids[self.tokens.reason_start]

    @property
    def reason_end_id(self) -> int:
        return self._token_ids[self.tokens.reason_end]

    @property
    def step_start_id(self) -> int:
        return self._token_ids[self.tokens.step_start]

    @property
    def step_end_id(self) -> int:
        return self._token_ids[self.tokens.step_end]

    @property
    def action_start_id(self) -> int:
        return self._token_ids[self.tokens.action_start]

    @property
    def action_end_id(self) -> int:
        return self._token_ids[self.tokens.action_end]

    @property
    def tool_call_id(self) -> int:
        return self._token_ids[self.tokens.tool_call]

    @property
    def user_start_id(self) -> int:
        return self._token_ids[self.tokens.user_start]

    @property
    def assistant_start_id(self) -> int:
        return self._token_ids[self.tokens.assistant_start]

    # =========================================================================
    # Utility methods
    # =========================================================================

    def add_reasoning_prompt(self, text: str) -> str:
        """Add reasoning prompt to text."""
        return f"{text}{self.tokens.reason_start}\n{self.tokens.step_start}"

    def wrap_code(self, code: str, language: str = "python") -> str:
        """Wrap code in code tokens."""
        t = self.tokens
        lang_token = getattr(t, f"lang_{language}", t.lang_python)
        return f"{t.code_start}{lang_token}\n{code}\n{t.code_end}"

    def wrap_tool_call(self, name: str, args: Dict[str, Any]) -> str:
        """Format a tool call."""
        t = self.tokens
        return (
            f"{t.tool_call}\n"
            f"{t.tool_args}\n"
            f'{{"name": "{name}", "arguments": {json.dumps(args)}}}\n'
            f"{t.tool_args_end}\n"
            f"{t.tool_call_end}"
        )

    def wrap_memory(self, content: str, action: str = "store") -> str:
        """Wrap content in memory tokens."""
        t = self.tokens
        if action == "store":
            return f"{t.store}\n{content}\n{t.store_end}"
        elif action == "retrieve":
            return f"{t.retrieve}\n{content}\n{t.retrieve_end}"
        elif action == "cite":
            return f"{t.cite}{content}{t.cite_end}"
        return content


# Legacy alias for backward compatibility
ChatTokenizer = ComplexityTokenizer


# =============================================================================
# Format Compatibility Mappings
# =============================================================================

@dataclass
class TokenFormatMapping:
    """
    Mapping between INL Complexity tokens and other formats.

    Provides bidirectional conversion between:
    - Complexity format (INL native)
    - Llama 3 format
    - Mistral format
    - ChatML (GPT/OpenAI style)
    - Gemma format
    """

    # Mapping: complexity_token -> {format: token}
    MAPPINGS = {
        # BOS/EOS
        "<|begin|>": {
            "llama3": "<|begin_of_text|>",
            "mistral": "<s>",
            "chatml": "<|im_start|>",
            "gemma": "<bos>",
        },
        "<|end|>": {
            "llama3": "<|end_of_text|>",
            "mistral": "</s>",
            "chatml": "<|im_end|>",
            "gemma": "<eos>",
        },

        # Role tokens
        "<|system|>": {
            "llama3": "<|start_header_id|>system<|end_header_id|>",
            "mistral": "[INST]",
            "chatml": "<|im_start|>system",
            "gemma": "<start_of_turn>user",  # Gemma uses user for system
        },
        "<|user|>": {
            "llama3": "<|start_header_id|>user<|end_header_id|>",
            "mistral": "[INST]",
            "chatml": "<|im_start|>user",
            "gemma": "<start_of_turn>user",
        },
        "<|assistant|>": {
            "llama3": "<|start_header_id|>assistant<|end_header_id|>",
            "mistral": "",  # Mistral just ends [INST] and starts response
            "chatml": "<|im_start|>assistant",
            "gemma": "<start_of_turn>model",
        },
        "<|/user|>": {
            "llama3": "<|eot_id|>",
            "mistral": "[/INST]",
            "chatml": "<|im_end|>",
            "gemma": "<end_of_turn>",
        },
        "<|/assistant|>": {
            "llama3": "<|eot_id|>",
            "mistral": "</s>",
            "chatml": "<|im_end|>",
            "gemma": "<end_of_turn>",
        },

        # Tool calling (Llama 3.1+ style)
        "<|call|>": {
            "llama3": "<|python_tag|>",  # Llama 3.1 uses this for tools
            "mistral": "[TOOL_CALLS]",
            "chatml": "<|im_start|>assistant\n{\"tool_calls\":",
            "gemma": "<start_of_turn>model\n```tool_call",
        },
        "<|result|>": {
            "llama3": "<|start_header_id|>ipython<|end_header_id|>",
            "mistral": "[TOOL_RESULTS]",
            "chatml": "<|im_start|>tool",
            "gemma": "<start_of_turn>tool",
        },

        # Thinking/reasoning (for models that support it)
        "<|reason|>": {
            "llama3": "",  # No native support
            "mistral": "",
            "chatml": "",
            "gemma": "",
            "deepseek": "<think>",
            "qwen": "<think>",
        },
        "<|/reason|>": {
            "llama3": "",
            "mistral": "",
            "chatml": "",
            "gemma": "",
            "deepseek": "</think>",
            "qwen": "</think>",
        },
    }

    @classmethod
    def to_format(cls, text: str, target_format: str) -> str:
        """
        Convert Complexity format text to target format.

        Args:
            text: Text with Complexity tokens
            target_format: Target format (llama3, mistral, chatml, gemma)

        Returns:
            Converted text
        """
        result = text
        for complexity_token, mappings in cls.MAPPINGS.items():
            if target_format in mappings:
                target_token = mappings[target_format]
                result = result.replace(complexity_token, target_token)
        return result

    @classmethod
    def from_format(cls, text: str, source_format: str) -> str:
        """
        Convert from target format to Complexity format.

        Args:
            text: Text with source format tokens
            source_format: Source format (llama3, mistral, chatml, gemma)

        Returns:
            Text with Complexity tokens
        """
        result = text
        for complexity_token, mappings in cls.MAPPINGS.items():
            if source_format in mappings:
                source_token = mappings[source_format]
                if source_token:  # Only replace non-empty tokens
                    result = result.replace(source_token, complexity_token)
        return result

    @classmethod
    def get_special_tokens_for_format(cls, target_format: str) -> Dict[str, str]:
        """Get the special tokens needed for a target format."""
        tokens = {}
        for complexity_token, mappings in cls.MAPPINGS.items():
            if target_format in mappings:
                tokens[complexity_token] = mappings[target_format]
        return tokens


class CompatibleTokenizer:
    """
    Tokenizer wrapper that can work with multiple model formats.

    Internally uses Complexity format but can:
    - Load Llama/Mistral/GPT tokenizers
    - Convert between formats
    - Train new tokenizers compatible with existing models

    Usage:
        # Load a Llama tokenizer but use Complexity format
        tok = CompatibleTokenizer.from_pretrained(
            "meta-llama/Llama-3-8B",
            target_format="complexity",  # Use our format internally
        )

        # The tokenizer will automatically map tokens
        encoded = tok.encode_chat(messages)

        # Or convert existing Complexity text to Llama format
        llama_text = tok.to_format(complexity_text, "llama3")
    """

    # Predefined vocab sizes and base models
    PRESETS = {
        "llama3": {
            "vocab_size": 128256,
            "base_model": "meta-llama/Meta-Llama-3-8B",
            "special_tokens_start": 128000,
        },
        "llama2": {
            "vocab_size": 32000,
            "base_model": "meta-llama/Llama-2-7b-hf",
            "special_tokens_start": 32000,
        },
        "mistral": {
            "vocab_size": 32768,
            "base_model": "mistralai/Mistral-7B-v0.1",
            "special_tokens_start": 32000,
        },
        "gpt2": {
            "vocab_size": 50257,
            "base_model": "gpt2",
            "special_tokens_start": 50257,
        },
        "gemma": {
            "vocab_size": 256000,
            "base_model": "google/gemma-7b",
            "special_tokens_start": 256000,
        },
        "qwen2": {
            "vocab_size": 151936,
            "base_model": "Qwen/Qwen2-7B",
            "special_tokens_start": 151643,
        },
    }

    def __init__(
        self,
        base_tokenizer: Tokenizer,
        source_format: str = "auto",
        add_complexity_tokens: bool = True,
    ):
        """
        Args:
            base_tokenizer: Base tokenizer (from HF, tiktoken, etc.)
            source_format: Source format of the tokenizer
            add_complexity_tokens: Add Complexity special tokens
        """
        self.tokenizer = base_tokenizer
        self.source_format = source_format
        self.complexity_tokens = ComplexityTokens()

        # Build token mapping
        self._complexity_to_source = {}
        self._source_to_complexity = {}

        if source_format != "complexity" and source_format in TokenFormatMapping.MAPPINGS.get("<|begin|>", {}):
            mapping = TokenFormatMapping.get_special_tokens_for_format(source_format)
            for complexity_tok, source_tok in mapping.items():
                if source_tok:
                    self._complexity_to_source[complexity_tok] = source_tok
                    self._source_to_complexity[source_tok] = complexity_tok

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        source_format: str = "auto",
        add_complexity_tokens: bool = True,
        **kwargs,
    ) -> "CompatibleTokenizer":
        """
        Load a pretrained tokenizer with Complexity compatibility.

        Args:
            model_name_or_path: HuggingFace model name or path
            source_format: Source format (auto-detected if "auto")
            add_complexity_tokens: Add Complexity special tokens

        Returns:
            CompatibleTokenizer instance
        """
        # Auto-detect format from model name
        if source_format == "auto":
            name_lower = model_name_or_path.lower()
            if "llama-3" in name_lower or "llama3" in name_lower:
                source_format = "llama3"
            elif "llama" in name_lower:
                source_format = "llama2"
            elif "mistral" in name_lower or "mixtral" in name_lower:
                source_format = "mistral"
            elif "gemma" in name_lower:
                source_format = "gemma"
            elif "qwen" in name_lower:
                source_format = "qwen2"
            elif "gpt" in name_lower:
                source_format = "gpt2"
            else:
                source_format = "chatml"  # Default to ChatML

        # Load base tokenizer
        base_tok = HuggingFaceTokenizer(model_name_or_path, **kwargs)

        return cls(base_tok, source_format, add_complexity_tokens)

    @classmethod
    def create_compatible(
        cls,
        base_format: str,
        vocab_size: Optional[int] = None,
        add_complexity_tokens: bool = True,
    ) -> Dict[str, Any]:
        """
        Get configuration to train a tokenizer compatible with a base format.

        Args:
            base_format: Target compatibility format
            vocab_size: Override vocab size
            add_complexity_tokens: Reserve space for Complexity tokens

        Returns:
            Dict with training configuration
        """
        if base_format not in cls.PRESETS:
            raise ValueError(f"Unknown format: {base_format}. Use: {list(cls.PRESETS.keys())}")

        preset = cls.PRESETS[base_format]

        # Calculate vocab size
        if vocab_size is None:
            vocab_size = preset["vocab_size"]

        # Reserve space for Complexity tokens (2048)
        complexity_tokens = ComplexityTokens()
        complexity_reserved = complexity_tokens.num_reserved if add_complexity_tokens else 0

        # Get all special tokens
        special_tokens = []
        if add_complexity_tokens:
            token_map = complexity_tokens.get_token_map()
            special_tokens = list(token_map.keys())

        return {
            "vocab_size": vocab_size,
            "special_tokens": special_tokens,
            "reserved_tokens": complexity_reserved,
            "compatible_with": base_format,
            "base_model": preset["base_model"],
        }

    def encode(self, text: str, use_complexity_format: bool = True) -> List[int]:
        """
        Encode text, optionally converting from Complexity format.

        Args:
            text: Input text
            use_complexity_format: Text is in Complexity format

        Returns:
            Token IDs
        """
        if use_complexity_format and self.source_format != "complexity":
            # Convert Complexity tokens to source format
            text = TokenFormatMapping.to_format(text, self.source_format)

        return self.tokenizer.encode(text)

    def decode(self, tokens: List[int], to_complexity_format: bool = True) -> str:
        """
        Decode tokens, optionally converting to Complexity format.

        Args:
            tokens: Token IDs
            to_complexity_format: Convert result to Complexity format

        Returns:
            Decoded text
        """
        text = self.tokenizer.decode(tokens)

        if to_complexity_format and self.source_format != "complexity":
            text = TokenFormatMapping.from_format(text, self.source_format)

        return text

    def encode_chat(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """
        Encode chat messages using Complexity template, then convert.

        Args:
            messages: Chat messages
            add_generation_prompt: Add generation prompt

        Returns:
            Token IDs
        """
        # Format with Complexity template
        template = ComplexityTemplate(format="complexity")
        msgs = [Message(**m) if isinstance(m, dict) else m for m in messages]
        text = template.apply(msgs, add_generation_prompt)

        # Convert to source format and encode
        return self.encode(text, use_complexity_format=True)

    def to_format(self, text: str, target_format: str) -> str:
        """Convert Complexity format text to another format."""
        return TokenFormatMapping.to_format(text, target_format)

    def from_format(self, text: str, source_format: str) -> str:
        """Convert from another format to Complexity format."""
        return TokenFormatMapping.from_format(text, source_format)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.tokenizer.pad_token_id


# =============================================================================
# Tool Calling Support
# =============================================================================

@dataclass
class ToolDefinition:
    """Definition of a callable tool/function."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


@dataclass
class ToolCall:
    """A tool call from the model."""
    id: str
    name: str
    arguments: Dict[str, Any]


class ToolCallingMixin:
    """
    Mixin for tool calling support in tokenizers.

    Formats tool definitions and parses tool calls from model output.
    """

    def format_tools(
        self,
        tools: List[ToolDefinition],
        special_tokens: SpecialTokens,
    ) -> str:
        """Format tool definitions for model input."""
        st = special_tokens

        result = [f"{st.available_tools_start}\n"]
        for tool in tools:
            result.append(json.dumps(tool.to_dict(), indent=2))
            result.append("\n")
        result.append(f"{st.available_tools_end}\n")

        return "".join(result)

    def parse_tool_calls(
        self,
        text: str,
        special_tokens: SpecialTokens,
    ) -> List[ToolCall]:
        """Parse tool calls from model output."""
        st = special_tokens
        calls = []

        pattern = re.compile(
            f"{re.escape(st.tool_call_start)}(.*?){re.escape(st.tool_call_end)}",
            re.DOTALL
        )

        for match in pattern.finditer(text):
            try:
                data = json.loads(match.group(1).strip())
                calls.append(ToolCall(
                    id=data.get("id", f"call_{len(calls)}"),
                    name=data.get("name", ""),
                    arguments=data.get("arguments", {}),
                ))
            except (json.JSONDecodeError, KeyError):
                continue

        return calls

    def format_tool_response(
        self,
        tool_call_id: str,
        response: Any,
        special_tokens: SpecialTokens,
    ) -> str:
        """Format tool response for model input."""
        st = special_tokens

        return (
            f"{st.tool_response_start}\n"
            f'{{"tool_call_id": "{tool_call_id}", "output": {json.dumps(response)}}}\n'
            f"{st.tool_response_end}\n"
        )
