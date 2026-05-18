# agent_dev

一个面向中国大厂 `Agent Dev / Agent Eval` 岗位的 `Task Forge v2` 对外版原型。

它的目标不是直接评测模型，而是自动把一段真实 Python 工作流代码转成“适合做 Agent benchmark 的任务草案”，也就是一个小型的 benchmark authoring pipeline。

换句话说，这个仓库展示的是：

- 你如何从真实代码里挖掘可评测任务
- 你如何判断一个 source file 值不值得做 benchmark seed
- 你如何用 planner / critic / curator 这条链把任务边界收紧

## 这个仓库在面试里体现什么能力

- 从真实 workflow 提炼 benchmark boundary 的能力
- 区分 `recoverable facts` 和 `user-only policy` 的能力
- 识别 `leakage / scope creep / harness overreach` 的能力
- 设计 task-authoring pipeline，而不只是堆 prompt 的能力

## 它会做什么

给它一个 Python 源文件后，它会：

1. 挖掘源码里的 workflow、风险点、路径线索、可恢复事实
2. 从本地 `knowledge/` 里检索 benchmark authoring 规则
3. 判断这个 source 是否是一个强 benchmark seed
4. 生成任务草案
5. 可选地跑四个 critic
6. 做共识收敛
7. 输出一个 curated benchmark task
8. 把 artifact 写到 `artifacts/`

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env
```

然后填：

- `TASK_FORGE_API_KEY`
- 可选 `TASK_FORGE_BASE_URL`
- 可选 `TASK_FORGE_CHAT_MODEL`
- 可选 `TASK_FORGE_SOURCE_ROOT`

跑测试：

```bash
make test
```

跑单文件：

```bash
make run
```

## 两种输入方式

这个对外版只依赖两类外部输入：

- `.env` 里的 API key / model 配置
- source code 的位置

source code 可以通过两种方式提供：

1. 在 `.env` 里设置 `TASK_FORGE_SOURCE_ROOT`
2. 运行时用 `--source` 或 `--source-root` 显式指定

如果 `TASK_FORGE_SOURCE_ROOT` 不填，默认会扫 [examples/source](</C:/Users/agares/OneDrive/0 求职/面试/agent_dev/examples/source>)。

## 常用命令

- `make test`
  - 跑本地单测，不需要 key
- `make run`
  - 跑一个示例 source 的完整流程
- `make run-no-critics`
  - 跑无 critic 的 ablation baseline
- `make batch`
  - 批量扫描一个 source root

直接命令行也可以：

```bash
python -m task_forge_v2.run_pipeline --source examples/source/example_pair_promotion.py
python -m task_forge_v2.run_batch --source-root path/to/your/source_tree --limit 4
```

## 环境变量说明

最小 OpenAI 配置例子：

```env
TASK_FORGE_API_KEY=...
TASK_FORGE_BASE_URL=https://api.openai.com/v1
TASK_FORGE_CHAT_MODEL=gpt-4.1-mini
TASK_FORGE_SOURCE_ROOT=C:\path\to\your\python\sources
```

如果你接别的 OpenAI-compatible 服务，只需要改：

- `TASK_FORGE_API_KEY`
- `TASK_FORGE_BASE_URL`
- `TASK_FORGE_CHAT_MODEL`

## 仓库结构

- `task_forge_v2/`
  - 核心 pipeline 代码
- `knowledge/`
  - 本地检索知识库
- `examples/source/`
  - 示例源文件
- `tests/`
  - 单测

## 产物输出

- 运行 artifact 写到 `artifacts/`
- retrieval cache 写到 `.cache/retrieval_index/`

每次 run 典型会生成：

- `run_bundle.json`
- `source_summary.json`
- `retrieval_hits.json`
- `source_triage.json`
- `draft_task.json`
- `critic_reviews.json`
- `consensus_review.json`
- `curated_task.json`
- `anchor_assessment.json`
- `summary.md`

## 面试时可以怎么讲

- 这个仓库回答的是“怎么造 benchmark”，不是“怎么跑 benchmark”
- 它的关键不是某个模型答得多好，而是 benchmark authoring 的边界控制是否稳定
- 真正重要的设计点是：
  - 先 triage source，再决定是否值得进入昂贵链路
  - 把 critic 的职责拆开，不让一个 prompt 同时做生成、挑错、裁决
  - 最后用 curator 强行把任务压回单一 boundary 和单一 planning unit
