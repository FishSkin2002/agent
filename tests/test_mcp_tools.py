import io
import json
import os
import socket
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import main
from tools.mcp_client import MCPConfigError, register_mcp_tools
from tools.registry import ToolRegistry


ECHO_SERVER = """\
import os
from pathlib import Path
from mcp.server import MCPServer

server = MCPServer("test-server")

@server.tool()
def echo(text: str) -> dict[str, str]:
    \"\"\"Echo text from the agent.\"\"\"
    return {"echo": text, "prefix": os.environ["MCP_TEST_PREFIX"], "cwd": Path.cwd().name}

server.run(transport="stdio")
"""


class MCPToolTests(unittest.TestCase):
    def test_chat_discovers_and_calls_a_local_mcp_tool(self):
        # AnyIO's portal uses local socket I/O to wake the background event loop.
        try:
            sender, receiver = socket.socketpair()
            with sender, receiver:
                sender.sendall(b"x")
        except PermissionError:
            self.skipTest("local socket I/O is unavailable")

        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "server.py").write_text(ECHO_SERVER, encoding="utf-8")
            (directory / ".env").write_text("MODEL=deepseek-flash\nAPI_KEY=test-key\n", encoding="utf-8")
            env = {"MCP_TEST_PREFIX": "fixture"}
            if "PYTHONPATH" in os.environ:
                env["PYTHONPATH"] = os.environ["PYTHONPATH"]
            command = {"command": sys.executable, "args": ["server.py"], "env": env}
            config = {"mcpServers": {
                "missing": {"command": str(directory / "missing-server")},
                "test": command,
                "another": command,
            }}
            (directory / "mcp.json").write_text(json.dumps(config), encoding="utf-8")

            def next_message(url, model, api_key, messages, registry, **kwargs):
                latest = messages[-1]
                if latest["role"] == "user":
                    return self.call("list_tools")
                result = json.loads(latest["content"])
                if latest["tool_call_id"] == "list_tools":
                    echo_tools = [tool for tool in result["tools"] if tool["name"].startswith("mcp_")]
                    self.assertEqual(len(echo_tools), 2, errors.getvalue())
                    self.assertEqual(len({tool["name"] for tool in echo_tools}), 2)
                    tool = next(tool for tool in echo_tools if tool["name"].startswith("mcp_test__echo"))
                    self.assertEqual(tool["parameters"]["properties"]["text"]["type"], "string")
                    return self.call(tool["name"], text="hello from myAgent")
                self.assertFalse(result["is_error"])
                self.assertEqual(result["structured_content"], {
                    "echo": "hello from myAgent",
                    "prefix": "fixture",
                    "cwd": directory.name,
                })
                return {"role": "assistant", "content": "MCP result received"}

            with (
                patch.object(main, "__file__", str(directory / "main.py")),
                patch.object(main, "call_model", side_effect=next_message) as model,
                patch("builtins.input", side_effect=["请测试 echo 工具", "/exit"]),
                redirect_stdout(io.StringIO()) as output,
                redirect_stderr(io.StringIO()) as errors,
            ):
                self.assertEqual(main.main(), 0)
                self.assertEqual(model.call_count, 3)
                self.assertIn("MCP result received", output.getvalue())
                self.assertIn("MCP missing 连接失败", errors.getvalue())
                self.assertIn("MCP test 已连接", errors.getvalue())

    def test_malformed_config_reports_the_json_location(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mcp.json"
            path.write_text('{\n  "mcpServers":\n}', encoding="utf-8")
            with self.assertRaisesRegex(MCPConfigError, "mcp.json 第 3 行"):
                with register_mcp_tools(path, ToolRegistry()):
                    self.fail("Malformed MCP configuration was accepted")

    @staticmethod
    def call(name, **arguments):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": name,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }],
        }
