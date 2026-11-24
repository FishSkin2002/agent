# 论文知识检索 Agent

这个项目用于学习如何用 Python 手写 Agent 的对话和工具调用流程，并按用户问题在线查找论文证据。

## 项目目录

```text
.
├── docs/          # Git 工作约定、命令规则和 README 编写约定
├── scripts/       # 仓库 Git 维护脚本及其隔离测试
├── tests/         # 论文检索、记忆、MCP 与对话入口的测试
├── tools/         # Agent 工具注册表、论文知识库、记忆工具和 MCP 适配器
├── .gitignore     # 本地笔记、配置、数据库及缓存的忽略规则
├── AGENTS.md      # 项目协作与教学方式
├── README.md      # 项目用途与目录
├── chat.py        # 命令行多轮对话入口
├── memory.py      # SQLite 记忆存储与按轮次淘汰
└── requirements.txt # Python 依赖
```

## 输入与输出

在对话中输入“我把 ReAct 的论文标题记成了 Synergjzing Reasoning and Acting，请帮我找到原论文并给出版本链接。”终端会显示模型生成的查询式，回答会给出找到的论文，例如 [ReAct 的 arXiv 版本](https://arxiv.org/abs/2210.03629v3)。

可以继续问“ReAct 在 HotpotQA 和 FEVER 上分别用了几个 few-shot 示例？请引用原文页码。”答案可在 ReAct v3 的 PDF 第 4 页核对。

## 检索结果的核验

打开回答中的 arXiv 版本链接，核对标题、作者和摘要。首次提交日期以 arXiv 记录为准。

核对正文证据时，打开带 `#page=` 的版本链接并查看该页原文。

Thank you to arXiv for use of its open access interoperability.

## 环境准备

项目使用名为 `agent` 的 Conda 环境。首次配置时运行以下命令。

```bash
conda create -n agent python=3.12 pip -y
conda activate agent
python -m pip install -r requirements.txt
```

日常开发用 `conda activate agent` 进入环境。

## 主要脚本使用说明

`chat.py` 通过 GLM 或 DeepSeek API 进行命令行多轮对话。每次调用 `search_papers` 接收一个 `query`，执行一次 arXiv 查询并返回最多 5 篇论文。模型可以根据结果改写查询，再发起一次工具调用。

模型自主决定直接回答或调用工具。输入“请用工具查看你当前有哪些工具”，可以让模型通过 `list_tools` 查询工具名、用途和参数。

工具模块与模型使用的名称对应。例如 `tools/search_papers.py` 提供 `SearchPapersTool`。新增工具时继承 `Tool`，定义名称、描述和参数，实现 `run()`，再在 `chat.py` 中用 `ToolRegistry.register()` 注册实例。模型根据工具自身的描述和参数挑选工具。

在仓库根目录准备 `.env`，选择模型并填入对应服务的密钥。例如使用 DeepSeek 时填写如下。

```dotenv
MODEL=deepseek-flash
API_KEY=你的_API_Key
```

- `MODEL=deepseek-flash` 使用 DeepSeek API Key。
- `MODEL=glm-4.7-flash` 使用智谱开放平台通用 API Key。

```bash
python chat.py
```

在「用户 ❯」提示后输入消息，回复以「助手 ❯」开头。终端会显示正在等待模型、正在执行的工具和工具返回状态。arXiv 搜索还会显示查询式和返回的论文数，查询字段使用中文全称。每轮最多执行 10 次工具调用。输入 `/exit` 结束对话。

多步任务中，模型可选择制定计划，终端用 `[计划]` 显示当前步骤。调用了内置论文检索工具的轮次会在答复前显示 `[复核]`，模型会根据检索结果核对草稿。发现关键证据缺失时，Agent 带着已有结果继续检索。计划、检索和补查共享当轮的 10 次工具调用额度，额度用尽或补查未取得新证据时，答复会说明证据缺口。

### 添加 MCP 服务器

在与 `chat.py` 同目录的 `mcp.json` 中，把每个服务器添加到 `mcpServers` 对象。文件由 Git 忽略。首次添加后重新运行 `python chat.py`，输入“请用工具查看你当前有哪些工具”来检查注册结果。

本地服务器用 `command` 和 `args` 启动。下面的日期时间服务器示例需要本机已有 `uvx`。 [MCP Time 服务器文档](https://github.com/modelcontextprotocol/servers/tree/main/src/time)

```json
{
  "mcpServers": {
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"]
    }
  }
}
```

已运行的 Streamable HTTP 服务器用 `url` 连接，例如在 `mcpServers` 中添加 `"my_server": {"url": "http://127.0.0.1:3001/mcp"}`。把地址换成实际的 MCP 端点。HTTP 可用 `headers` 提供请求头，本地进程可用 `env` 设置环境变量、用 `cwd` 指定工作目录。相对工作目录以仓库根目录为准。

Agent 在启动时发现 MCP 工具。模型可以按需调用这些工具并取得文本或结构化结果。用 `{"mcpServers": {}}` 可以启动没有外部服务器的会话。

### 论文知识库

在线搜索返回的论文按带版本的 arXiv ID 保存在仓库根目录的 `.agent-data/papers.sqlite3`。重开对话后，Agent 可用 `search_paper_library` 对这些论文的标题和摘要做向量检索。

`search_paper_passages` 接收英文查询和可选的带版本 arXiv ID。指定 ID 时首次下载 PDF 并索引正文，后续查询复用片段。省略 ID 时检索所有已索引的正文，最多返回 3 段，附页码和版本链接。

首次做向量检索时，FastEmbed 会下载 BGE 英文嵌入模型到 `.agent-data/embedding-models/`。模型在本地 CPU 上运行。中文问题由对话模型改写成英文检索请求。

### 记忆

Agent 会按需要通过工具保存、查找、修改和淘汰记忆。可以在一轮说“以后找论文优先考虑顶会”，退出后重新运行，再问“我之前有什么论文筛选偏好？”重要性达到 `0.8` 的记录会升为长期记忆。模型可以主动淘汰两种记忆，长期记忆跳过轮次淘汰。

短期记忆从写入或修改后开始计数，再完成 10 个用户轮次时自动淘汰。查询记忆、单轮内反复调用工具、退出后的闲置时间和调用失败都不会额外增加轮次。当前对话也只向模型传入最近 10 个已完成轮次的消息。在 `.env` 中可用 `MEMORY_SHORT_TERM_TURNS=20` 同时改变这两个上限。

启动时会在仓库根目录创建 `.agent-data/memory.sqlite3`。记忆和轮次计数在重启后继续使用，SQLite 由 Python 标准库提供。
