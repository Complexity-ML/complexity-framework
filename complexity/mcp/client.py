"""Official Model Context Protocol SDK client helpers.

No tools are reimplemented here. ``OfficialMCPStdioClient`` only wraps the
public Python MCP SDK stdio client/session APIs and normalizes tool metadata into
small dataclasses that Complexity policies can consume.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class MCPTool:
    """Normalized MCP tool metadata."""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    raw: Any = None


@dataclass(frozen=True)
class MCPToolResult:
    """Normalized MCP tool call result."""

    content: Any
    is_error: bool = False
    raw: Any = None


@dataclass(frozen=True)
class OfficialMCPStdioConfig:
    """Configuration for launching an MCP server through official stdio transport."""

    command: str
    args: Sequence[str] = field(default_factory=tuple)
    env: Optional[Mapping[str, str]] = None
    cwd: Optional[str] = None


class OfficialMCPStdioClient:
    """Small async wrapper around the official MCP Python SDK stdio client."""

    def __init__(self, config: OfficialMCPStdioConfig):
        self.config = config

    async def list_tools(self) -> List[MCPTool]:
        """Launch/connect to the MCP server and return normalized tool metadata."""
        async with self._session() as session:
            result = await session.list_tools()
        return [self._normalize_tool(tool) for tool in getattr(result, "tools", [])]

    async def call_tool(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> MCPToolResult:
        """Call one MCP tool via official SDK and normalize the result."""
        async with self._session() as session:
            result = await session.call_tool(name, arguments=dict(arguments or {}))
        return MCPToolResult(
            content=getattr(result, "content", result),
            is_error=bool(getattr(result, "isError", False)),
            raw=result,
        )

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Any]:
        ClientSession, StdioServerParameters, stdio_client = _load_official_stdio_sdk()
        server_parameters = StdioServerParameters(
            command=self.config.command,
            args=list(self.config.args),
            env=dict(self.config.env) if self.config.env is not None else None,
            cwd=self.config.cwd,
        )
        async with stdio_client(server_parameters) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session

    @staticmethod
    def _normalize_tool(raw_tool: Any) -> MCPTool:
        name = _field(raw_tool, "name")
        description = _field(raw_tool, "description", "")
        input_schema = _field(raw_tool, "inputSchema", None)
        if input_schema is None:
            input_schema = _field(raw_tool, "input_schema", {"type": "object"})
        return MCPTool(
            name=name,
            description=description,
            input_schema=input_schema,
            raw=raw_tool,
        )


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _load_official_stdio_sdk() -> tuple[Any, Any, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise ImportError(
            "Official MCP SDK is required for complexity.mcp. "
            "Install with `pip install 'complexity-framework[tools]'` or `pip install mcp`."
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client
