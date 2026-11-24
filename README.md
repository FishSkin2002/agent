# 论文知识检索 Agent

这个项目用于学习如何用 Python 手写 Agent 的对话和工具调用流程，并按用户问题在线查找论文证据。

## 项目目录

```text
.
├── docs/          # Git 工作约定、命令规则和 README 编写约定
├── tools/         # 项目专用 Git 写入工具及其隔离测试
├── .gitignore     # 本地笔记、环境变量文件及缓存的忽略规则
├── AGENTS.md      # 项目协作与教学方式
├── README.md      # 项目用途与目录
├── chat.py        # 命令行多轮对话入口
├── paper_retrieval.py # 多查询检索与结果合并
├── paper_search.py # 在线论文检索入口
└── requirements.txt # Python 依赖
```

## 输入与输出

要找 ReAct 的原论文，可以输入 arXiv 查询式 `ti:"Synergizing Reasoning and Acting"`。输出为 JSON 数组，最多包含 5 篇论文。这个查询返回 `paper_id` 为 `2210.03629v3` 的论文，以及标题、作者、首次提交日期、摘要、摘要页链接和可选的发表信息。

## 检索结果的核验

通过结果中的 `url` 打开对应版本的 arXiv 摘要页，核对标题、作者和摘要。`submitted_at` 是 arXiv 首次提交日期。`journal_ref` 是 arXiv 记录提供的发表信息，缺失时返回 `null`。

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

### 在线论文检索

`paper_search.py` 接收英文关键词或 arXiv 查询式，并输出论文元数据。

```bash
python paper_search.py 'ti:"Synergizing Reasoning and Acting"'
```

### 命令行对话

`chat.py` 通过 GLM 或 DeepSeek API 进行命令行多轮对话。模型可用 `search_papers` 提交一条主查询和最多两条补充查询。程序分别查询 arXiv，合并去重后返回最多 5 篇论文，供模型根据摘要和链接回答。

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

可以问“我把 ReAct 的论文标题记成了 Synergjzing Reasoning and Acting，请帮我找到原论文并给出版本链接。”终端会显示模型给出的查询式。每轮最多执行 3 次工具调用。输入 `/exit` 结束对话。
