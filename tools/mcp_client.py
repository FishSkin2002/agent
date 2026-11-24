from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from .base import Tool
from .registry import ToolRegistry

if TYPE_CHECKING:
    from anyio.from_thread import BlockingPortal
    from mcp import Client
    from mcp_types import Tool as MCPToolDefinition


class MCPConfigError(ValueError):
    """An MCP configuration file or server entry is invalid."""


class MCPTool(Tool):
    def __init__(
        self,
        server: str,
        definition: MCPToolDefinition,
        client: Client,
        portal: BlockingPortal,
    ) -> None:
        self._client = client
        self._portal = portal
        self._remote_name = definition.name
        safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", f"mcp_{server}__{definition.name}")
        identity = json.dumps([server, definition.name]).encode("ascii")
        suffix = hashlib.blake2s(identity, digest_size=5).hexdigest()
        self.name = f"{safe_name[:53]}_{suffix}"
        self.description = f"[MCP {server}] {definition.description or definition.name}"
        self.parameters = definition.input_schema

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            result = self._portal.call(self._client.call_tool, self._remote_name, arguments)
        except Exception as error:
            return {"error": f"MCP 工具 {self._remote_name} 调用失败（{_error_type(error)}）"}

        content = [
            block.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                exclude={"meta": True, "data": True, "resource": {"meta", "blob"}},
            )
            for block in result.content
        ]
        output: dict[str, Any] = {"content": content, "is_error": result.is_error}
        if result.structured_content is not None:
            output["structured_content"] = result.structured_content
        return output


def _error_type(error: BaseException) -> str:
    while isinstance(error, BaseExceptionGroup):
        error = error.exceptions[0]
    return type(error).__name__


def _read_config(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise MCPConfigError(f"mcp.json 第 {error.lineno} 行第 {error.colno} 列的 JSON 格式有误。") from error
    except (OSError, UnicodeError) as error:
        raise MCPConfigError("请确认 mcp.json 是可读的 UTF-8 文件。") from error
    if not isinstance(data, dict) or not isinstance(data.get("mcpServers"), dict):
        raise MCPConfigError("mcp.json 必须包含一个 mcpServers 对象。")
    return data["mcpServers"]


def _string_map(value: Any, field: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value.values()):
        raise MCPConfigError(f"{field} 必须是一个值为字符串的对象。")
    return value


@asynccontextmanager
async def _connect(config: Any, directory: Path) -> AsyncIterator[tuple[Client, list[MCPToolDefinition]]]:
    from mcp import Client, StdioServerParameters, stdio_client

    if not isinstance(config, dict):
        raise MCPConfigError("服务器配置必须是 JSON 对象。")
    transport = config.get("type", "http" if "url" in config else "stdio")

    async with AsyncExitStack() as stack:
        if transport == "stdio":
            if config.keys() - {"type", "command", "args", "env", "cwd"}:
                raise MCPConfigError("stdio 服务器可配置 command、args、env、cwd 和 type。")
            command = config.get("command")
            args = config.get("args", [])
            cwd = config.get("cwd", ".")
            if (
                not isinstance(command, str) or not command.strip()
                or not isinstance(args, list) or any(not isinstance(arg, str) for arg in args)
                or not isinstance(cwd, str) or not cwd.strip()
            ):
                raise MCPConfigError("stdio 服务器需要 command 字符串，args 为字符串数组，cwd 为路径。")
            parameters = StdioServerParameters(
                command=command,
                args=args,
                env=_string_map(config.get("env", {}), "env"),
                cwd=directory / cwd,
            )
            server = stdio_client(parameters, errlog=sys.__stderr__)
        elif transport in ("http", "streamable-http"):
            from httpx2 import AsyncClient, Timeout
            from mcp.client.streamable_http import streamable_http_client

            if config.keys() - {"type", "url", "headers"}:
                raise MCPConfigError("HTTP 服务器可配置 url、headers 和 type。")
            url = config.get("url")
            try:
                address = urlsplit(url) if isinstance(url, str) else None
            except ValueError as error:
                raise MCPConfigError("url 必须是有效的 HTTP 或 HTTPS MCP 端点。") from error
            if address is None or address.scheme not in ("http", "https") or not address.hostname:
                raise MCPConfigError("url 必须是有效的 HTTP 或 HTTPS MCP 端点。")
            http = await stack.enter_async_context(AsyncClient(
                headers=_string_map(config.get("headers", {}), "headers"),
                timeout=Timeout(30.0, read=None),
            ))
            server = streamable_http_client(url, http_client=http)
        else:
            raise MCPConfigError("type 支持 stdio、http 或 streamable-http。")

        client = await stack.enter_async_context(Client(server, read_timeout_seconds=60.0))
        page = await client.list_tools()
        tools = {tool.name: tool for tool in page.tools}
        cursors = set()
        while page.next_cursor is not None:
            cursor = page.next_cursor
            if cursor in cursors:
                raise RuntimeError("MCP 服务器重复返回同一分页游标。")
            cursors.add(cursor)
            page = await client.list_tools(cursor=cursor)
            tools.update((tool.name, tool) for tool in page.tools)
        yield client, list(tools.values())


@contextmanager
def register_mcp_tools(path: Path, registry: ToolRegistry) -> Iterator[None]:
    servers = _read_config(path)
    if not servers:
        yield
        return

    from anyio.from_thread import start_blocking_portal

    with ExitStack() as stack:
        portal = stack.enter_context(start_blocking_portal())
        for server, config in servers.items():
            try:
                client, definitions = stack.enter_context(
                    portal.wrap_async_context_manager(_connect(config, path.resolve().parent))
                )
            except MCPConfigError as error:
                print(f"MCP {server} 配置错误（{error}）", file=sys.stderr)
                continue
            except Exception as error:
                print(f"MCP {server} 连接失败（{_error_type(error)}）", file=sys.stderr)
                continue
            for definition in definitions:
                registry.register(MCPTool(server, definition, client, portal))
            print(f"MCP {server} 已连接，注册 {len(definitions)} 个工具。", file=sys.stderr)
        try:
            yield
        finally:
            # Exit transport contexts normally even if the chat exits on EOF.
            stack.close()
