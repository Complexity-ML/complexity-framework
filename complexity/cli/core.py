"""
Core CLI utilities and state management.
"""

import torch
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelState:
    """Loaded model state."""
    model: Any = None
    tokenizer: Any = None
    name: str = ""
    device: str = "cpu"
    dtype: str = "float32"


@dataclass
class ChatHistory:
    """Chat conversation history."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    system: Optional[str] = None

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def clear(self):
        self.messages = []

    def to_list(self) -> List[Dict[str, str]]:
        result = []
        if self.system:
            result.append({"role": "system", "content": self.system})
        result.extend(self.messages)
        return result


class ComplexityCLI:
    """
    Core CLI manager.

    Handles model loading, inference, and state management.
    """

    def __init__(self):
        self.state = ModelState()
        self.history = ChatHistory()
        self._verbose = False

    def load_model(
        self,
        model_name: str = "complexity-7b",
        device: str = "auto",
        dtype: str = "auto",
        quantization: Optional[str] = None,
    ) -> bool:
        """
        Load a model.

        Args:
            model_name: Model name or path
            device: Device (auto/cuda/cpu/mps)
            dtype: Data type (auto/float16/bfloat16/float32)
            quantization: Quantization method (int8/int4/None)

        Returns:
            True if successful
        """
        # Skip if already loaded
        if self.state.model is not None and self.state.name == model_name:
            return True

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        # Determine dtype
        if dtype == "auto":
            if device == "cuda":
                dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                dtype = "float32"

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.float32)

        try:
            from complexity import ComplexityModel, get_preset, ModelConfig

            # Load config
            if model_name.startswith("complexity-"):
                try:
                    config = get_preset(model_name)
                except KeyError:
                    config = ModelConfig()

                model = ComplexityModel(config)
            else:
                # Assume path
                model = ComplexityModel.from_pretrained(model_name)

            # Quantization
            if quantization:
                from complexity.quantization import quantize_model
                model = quantize_model(model, method=quantization)

            # Move to device
            model = model.to(device=device, dtype=torch_dtype)
            model.eval()

            # Update state
            self.state.model = model
            self.state.name = model_name
            self.state.device = device
            self.state.dtype = dtype

            return True

        except Exception as e:
            if self._verbose:
                print(f"Error loading model: {e}")
            return False

    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        reasoning: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Chat with the model.

        Args:
            message: User message
            system: System prompt
            reasoning: Enable reasoning mode
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with response, reasoning, steps, etc.
        """
        from complexity.data import ComplexityTemplate, ComplexityTokens, Message

        # Update history
        if system and system != self.history.system:
            self.history.system = system
            self.history.clear()

        # Build messages
        messages = []
        if self.history.system:
            messages.append(Message(role="system", content=self.history.system))

        for msg in self.history.messages:
            messages.append(Message(role=msg["role"], content=msg["content"]))

        messages.append(Message(role="user", content=message))

        # Format
        template = ComplexityTemplate(
            tokens=ComplexityTokens(),
            enable_reasoning=reasoning,
            enable_steps=reasoning,
        )
        prompt = template.apply(messages, add_generation_prompt=True)

        # Generate
        output_text = self._generate(prompt, max_tokens, temperature)

        # Parse
        parsed = template.parse_response(output_text)

        # Update history
        self.history.add_user(message)
        self.history.add_assistant(parsed.content)

        return {
            "content": parsed.content,
            "reasoning": parsed.reasoning,
            "steps": parsed.steps,
            "conclusion": parsed.conclusion,
            "tool_calls": parsed.tool_calls,
        }

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.8,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Text prompt
            max_tokens: Max tokens
            temperature: Temperature

        Returns:
            Generated text
        """
        return self._generate(prompt, max_tokens, temperature)

    def _generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Internal generation method."""
        if self.state.model is None:
            return "[Model not loaded]"

        try:
            from complexity.inference import InferenceEngine, GenerationConfig

            engine = InferenceEngine(self.state.model)
            config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )
            return engine.generate_text(prompt, config)

        except Exception as e:
            if self._verbose:
                return f"[Error: {e}]"
            return f"[Generation error - model: {self.state.name}]"

    def predict_action(
        self,
        state: Dict[str, Any],
        goal: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict robotics action.

        Args:
            state: Robot state
            goal: Goal description
            instruction: Natural language instruction

        Returns:
            Action dict
        """
        from complexity.data import ComplexityTemplate, ComplexityTokens

        template = ComplexityTemplate(tokens=ComplexityTokens())
        prompt = template.format_for_robotics(state, goal, instruction)

        output = self._generate(prompt, max_tokens=64, temperature=0.0)

        return template.parse_robotics_action(output)

    def format_tokens(
        self,
        text: str,
        format: str = "complexity",
    ) -> str:
        """
        Show tokenization format.

        Args:
            text: Input text
            format: Format (complexity/chatml/llama)

        Returns:
            Formatted string
        """
        from complexity.data import ComplexityTemplate, ComplexityTokens, Message

        template = ComplexityTemplate(
            tokens=ComplexityTokens(),
            format=format,
        )
        messages = [Message(role="user", content=text)]
        return template.apply(messages)

    def clear_history(self):
        """Clear chat history."""
        self.history.clear()

    def set_system(self, prompt: str):
        """Set system prompt."""
        self.history.system = prompt

    @staticmethod
    def list_models() -> List[str]:
        """List available models."""
        return [
            "complexity-1b",
            "complexity-7b",
            "complexity-13b",
            "complexity-70b",
            "complexity-robotics-7b",
        ]

    @staticmethod
    def model_info(name: str) -> Dict[str, Any]:
        """Get model info."""
        info = {
            "complexity-1b": {
                "parameters": "1.3B",
                "context": "8K",
                "features": ["chat", "reasoning"],
                "vram": "~3GB",
            },
            "complexity-7b": {
                "parameters": "7B",
                "context": "32K",
                "features": ["chat", "reasoning", "tools", "code"],
                "vram": "~14GB",
            },
            "complexity-13b": {
                "parameters": "13B",
                "context": "32K",
                "features": ["chat", "reasoning", "tools", "code", "multimodal"],
                "vram": "~26GB",
            },
            "complexity-70b": {
                "parameters": "70B",
                "context": "128K",
                "features": ["chat", "reasoning", "tools", "code", "multimodal", "agents"],
                "vram": "~140GB",
            },
            "complexity-robotics-7b": {
                "parameters": "7B",
                "context": "16K",
                "features": ["robotics", "action", "trajectory", "vision"],
                "vram": "~14GB",
            },
        }
        return info.get(name, {"error": f"Unknown model: {name}"})


# Global CLI instance
cli = ComplexityCLI()
