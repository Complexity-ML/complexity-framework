"""
Registry system for framework-complexity.

This allows users to register custom components (attention, mlp, normalization, etc.)
and use them in their models via config.

Example:
    @register_attention("my_custom_attention")
    class MyAttention(AttentionBase):
        ...

    # Then in config:
    config = ModelConfig(attention_type="my_custom_attention", ...)
"""

from typing import Dict, Type, TypeVar, Callable, Optional
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)


class Registry:
    """Generic registry for neural network components."""

    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[nn.Module]] = {}

    def register(self, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component.

        Args:
            name: The name to register the component under.

        Returns:
            Decorator function.

        Example:
            @ATTENTION_REGISTRY.register("gqa")
            class GroupedQueryAttention(AttentionBase):
                ...
        """
        def decorator(cls: Type[T]) -> Type[T]:
            if name in self._registry:
                raise ValueError(
                    f"Component '{name}' already registered in {self.name} registry. "
                    f"Existing: {self._registry[name]}, New: {cls}"
                )
            self._registry[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type[nn.Module]:
        """Get a registered component by name.

        Args:
            name: The registered name of the component.

        Returns:
            The component class.

        Raises:
            KeyError: If component not found.
        """
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        return self._registry[name]

    def build(self, name: str, *args, **kwargs) -> nn.Module:
        """Build a component by name.

        Args:
            name: The registered name of the component.
            *args: Positional arguments for the component constructor.
            **kwargs: Keyword arguments for the component constructor.

        Returns:
            Instantiated component.
        """
        cls = self.get(name)
        return cls(*args, **kwargs)

    def list(self) -> list:
        """List all registered component names."""
        return list(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"Registry(name={self.name}, components={self.list()})"


# Global registries for each component type
ATTENTION_REGISTRY = Registry("attention")
MLP_REGISTRY = Registry("mlp")
NORMALIZATION_REGISTRY = Registry("normalization")
POSITION_REGISTRY = Registry("position")
MODEL_REGISTRY = Registry("model")


# Convenience decorators
def register_attention(name: str):
    """Register an attention implementation."""
    return ATTENTION_REGISTRY.register(name)


def register_mlp(name: str):
    """Register an MLP implementation."""
    return MLP_REGISTRY.register(name)


def register_normalization(name: str):
    """Register a normalization implementation."""
    return NORMALIZATION_REGISTRY.register(name)


def register_position(name: str):
    """Register a position embedding implementation."""
    return POSITION_REGISTRY.register(name)


def register_model(name: str):
    """Register a model architecture."""
    return MODEL_REGISTRY.register(name)
