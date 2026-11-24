# 论文知识检索 Agent

这个项目从用 Python 手写检索流程起步，学习如何构建一个能按问题查找论文证据的 Agent。

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

在仓库根目录的 `.env` 中按 `变量名=密钥` 格式填写一项，另一项留空。

- `GLM_API_KEY` 使用智谱开放平台通用 API Key，模型为 `glm-4.7-flash`。
- `DEEPSEEK_API_KEY` 使用 DeepSeek API Key，模型为 `deepseek-flash`。

```bash
python chat.py
```

可以先问“什么是 BM25？”，再追问“它为什么要考虑文档长度？”。输入 `/exit` 结束对话。
