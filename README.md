# 论文知识检索 Agent

这个项目用于学习如何用 Python 手写 Agent 的对话和工具调用流程，并按用户问题在线查找论文证据。

## 项目目录

```text
.
├── docs/          # Git 工作约定、命令规则和 README 编写约定
├── scripts/       # 仓库 Git 维护脚本及其隔离测试
├── tests/         # 记忆与对话入口的测试
├── tools/         # Agent 的工具注册表、论文检索和记忆工具
├── .gitignore     # 本地笔记、环境变量文件、记忆库及缓存的忽略规则
├── AGENTS.md      # 项目协作与教学方式
├── README.md      # 项目用途与目录
├── chat.py        # 命令行多轮对话入口
├── memory.py      # SQLite 记忆存储与按轮次淘汰
└── requirements.txt # Python 依赖
```

## 输入与输出

在对话中输入“我把 ReAct 的论文标题记成了 Synergjzing Reasoning and Acting，请帮我找到原论文并给出版本链接。”终端会显示模型生成的查询式，回答会给出找到的论文，例如 [ReAct 的 arXiv 版本](https://arxiv.org/abs/2210.03629v3)。

## 检索结果的核验

打开回答中的 arXiv 版本链接，核对标题、作者和摘要。首次提交日期以 arXiv 记录为准。

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

每轮最多执行 10 次工具调用。输入 `/exit` 结束对话。

### 记忆

Agent 会按需要通过工具保存、查找、修改和淘汰记忆。可以在一轮说“以后找论文优先考虑顶会”，退出后重新运行，再问“我之前有什么论文筛选偏好？”重要性达到 `0.8` 的记录会升为长期记忆。模型可以主动淘汰两种记忆，长期记忆跳过轮次淘汰。

短期记忆从写入或修改后开始计数，再完成 10 个用户轮次时自动淘汰。查询记忆、单轮内反复调用工具、退出后的闲置时间和调用失败都不会额外增加轮次。当前对话也只向模型传入最近 10 个已完成轮次的消息。在 `.env` 中可用 `MEMORY_SHORT_TERM_TURNS=20` 同时改变这两个上限。

启动时会在仓库根目录创建 `.agent-data/memory.sqlite3`。记忆和轮次计数在重启后继续使用，SQLite 由 Python 标准库提供。
