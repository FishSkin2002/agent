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
└── chat.py        # 命令行多轮对话入口
```

## 环境准备

项目使用名为 `agent` 的 Conda 环境。首次配置时运行以下命令。

```bash
conda create -n agent python=3.12 -y
conda activate agent
```

日常开发用 `conda activate agent` 进入环境。

## 主要脚本使用说明

`chat.py` 通过 GLM 或 DeepSeek API 进行命令行多轮对话。

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

可以先问“什么是 BM25？”，再追问“它为什么要考虑文档长度？”。输入 `/exit` 结束对话。
