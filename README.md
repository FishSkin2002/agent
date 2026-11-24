# 论文知识检索 Agent

这个项目从用 Python 手写检索流程起步，学习如何构建一个能按问题查找论文证据的 Agent。

## 项目目录

```text
.
├── docs/          # Git 工作约定、命令规则和 README 编写约定
├── tools/         # 项目专用 Git 写入工具及其隔离测试
├── .gitignore     # 本地笔记、环境变量文件及缓存的忽略规则
├── AGENTS.md      # 项目协作与教学方式
└── README.md      # 项目用途与目录
```

## 环境准备

项目使用名为 `agent` 的 Conda 环境。首次配置时运行以下命令。

```bash
conda create -n agent python=3.12 -y
conda activate agent
```

日常开发用 `conda activate agent` 进入环境。
