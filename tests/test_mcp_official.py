import asyncio
import sys
import types
from pathlib import Path

import pytest


def test_mcp_package_import_is_lazy_without_sdk_installed():
    import complexity.mcp as cmcp

    assert cmcp.OfficialMCPStdioClient is not None
    assert cmcp.OfficialMCPStdioConfig is not None


def test_pyproject_exposes_optional_tools_extra_for_official_mcp_sdk():
    pyproject = Path("pyproject.toml").read_text()

    assert "tools = [" in pyproject
    assert '"mcp>=' in pyproject


def test_real_official_mcp_sdk_imports_when_installed():
    pytest.importorskip("mcp")

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    assert ClientSession is not None
    assert StdioServerParameters is not None
    assert stdio_client is not None


def install_fake_mcp_sdk(monkeypatch):
    mcp_module = types.ModuleType("mcp")
    client_module = types.ModuleType("mcp.client")
    stdio_module = types.ModuleType("mcp.client.stdio")

    class StdioServerParameters:
        def __init__(self, command, args=None, env=None, cwd=None):
            self.command = command
            self.args = args
            self.env = env
            self.cwd = cwd

    class FakeTool:
        name = "filesystem.read"
        description = "Read a file"
        inputSchema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    class FakeListToolsResult:
        tools = [FakeTool()]

    class FakeCallToolResult:
        content = [{"type": "text", "text": "hello"}]
        isError = False

    class ClientSession:
        def __init__(self, read, write):
            self.read = read
            self.write = write

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return FakeListToolsResult()

        async def call_tool(self, name, arguments=None):
            assert name == "filesystem.read"
            assert arguments == {"path": "README.md"}
            return FakeCallToolResult()

    class FakeStdioClient:
        def __init__(self, server_parameters):
            self.server_parameters = server_parameters

        async def __aenter__(self):
            assert self.server_parameters.command == "node"
            assert self.server_parameters.args == ["server.js"]
            return "read", "write"

        async def __aexit__(self, exc_type, exc, tb):
            return False

    def stdio_client(server_parameters):
        return FakeStdioClient(server_parameters)

    mcp_module.ClientSession = ClientSession
    mcp_module.StdioServerParameters = StdioServerParameters
    stdio_module.stdio_client = stdio_client

    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.client", client_module)
    monkeypatch.setitem(sys.modules, "mcp.client.stdio", stdio_module)


def test_official_mcp_stdio_client_lists_tools(monkeypatch):
    install_fake_mcp_sdk(monkeypatch)
    from complexity.mcp import OfficialMCPStdioClient, OfficialMCPStdioConfig

    client = OfficialMCPStdioClient(OfficialMCPStdioConfig(command="node", args=["server.js"]))

    tools = asyncio.run(client.list_tools())

    assert len(tools) == 1
    assert tools[0].name == "filesystem.read"
    assert tools[0].description == "Read a file"
    assert tools[0].input_schema["required"] == ["path"]


def test_official_mcp_stdio_client_calls_tool(monkeypatch):
    install_fake_mcp_sdk(monkeypatch)
    from complexity.mcp import OfficialMCPStdioClient, OfficialMCPStdioConfig

    client = OfficialMCPStdioClient(OfficialMCPStdioConfig(command="node", args=["server.js"]))

    result = asyncio.run(client.call_tool("filesystem.read", {"path": "README.md"}))

    assert result.is_error is False
    assert result.content == [{"type": "text", "text": "hello"}]
