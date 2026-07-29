"""Official MCP SDK integration for Complexity.

This package intentionally does not reimplement tools. It bridges Complexity's
lexical / Token-Routed model stack to the official Model Context Protocol SDK.
The MCP SDK is optional; install with ``complexity-framework[tools]``.
"""

from .client import MCPTool, MCPToolResult, OfficialMCPStdioClient, OfficialMCPStdioConfig

__all__ = [
    "MCPTool",
    "MCPToolResult",
    "OfficialMCPStdioClient",
    "OfficialMCPStdioConfig",
]
