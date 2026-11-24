# 论文知识检索 Agent

用 Python 实现的 Agent，在线检索论文并根据原文证据回答问题。

## 项目目录

```text
.
├── docs/             # 项目约定
├── llm/              # 模型请求和流式响应
├── memory/           # 记忆存储
├── scripts/          # Git 维护脚本
├── tests/            # 测试
├── tools/            # Agent 工具与论文知识库
├── .gitignore        # 本地配置和数据忽略规则
├── AGENTS.md         # 协作规则
├── README.md         # 使用说明
├── main.py           # 命令行入口
└── requirements.txt # Python 依赖
```

## 输入与输出

输入“ReAct 在 HotpotQA 和 FEVER 上分别用了几个 few-shot 示例？请引用原文页码。”

示例答复为 HotpotQA 用了 6 个，FEVER 用了 3 个。 [ReAct v3，第 4 页](https://arxiv.org/pdf/2210.03629v3#page=4)

## 检索结果的核验

打开 arXiv 版本链接核对论文信息，打开带 `#page=` 的 PDF 链接核对原文。

Thank you to arXiv for use of its open access interoperability.

## 环境准备

项目使用名为 `agent` 的 Conda 环境。首次配置时运行以下命令。

```bash
conda create -n agent python=3.12 pip -y
conda activate agent
python -m pip install -r requirements.txt
```

## 主要脚本使用说明

`main.py` 启动论文助手。先在仓库根目录的 `.env` 填入所选平台的 API Key。

```dotenv
MODEL=deepseek-flash
API_KEY=你的_API_Key
MAX_OUTPUT_TOKENS=2048
MAX_TOOL_CALLS=20
```

- `MODEL` 可选 `deepseek-flash` 或 `glm-4.7-flash`。
- `MAX_OUTPUT_TOKENS` 是每次模型生成的输出上限，默认 `2048`。修改后重启。
- `MAX_TOOL_CALLS` 是每次提问的工具调用总上限，默认 `10`，`0` 停用工具。
- `MEMORY_SHORT_TERM_TURNS` 同时设置短期记忆淘汰轮数和最近问答数，默认 `10`。

```bash
python main.py
```

在「用户 ❯」后输入问题，输入 `/exit` 退出。普通回复实时显示，论文答复在复核后显示。

### 添加 MCP 服务器

在仓库根目录的 `mcp.json` 中添加服务器，替换示例地址后重启 Agent。

```json
{
  "mcpServers": {
    "my_server": {"url": "http://127.0.0.1:3000/mcp"}
  }
}
```

HTTP 服务器可加 `headers`。本地服务器用 `command`、`args`，可加 `env`、`cwd`。启动后可问「当前有哪些工具？」检查注册结果。

### 本地数据

数据位于仓库根目录的 `.agent-data/`。

```text
.agent-data/
├── memory.sqlite3    # Agent 记忆
├── papers.sqlite3    # 论文目录和已索引片段
└── embedding-models/ # 首次向量检索时下载的嵌入模型
```
