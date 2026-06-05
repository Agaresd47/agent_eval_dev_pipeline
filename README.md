# Python 脏脚本规范化重构


给它一份 Python 源代码,它把脚本里糊在一起的多个职责拆成多个可独立验证的小任务,输出结构化的方案 JSON。终稿是否通过由规则函数裁定,LLM 不参与判定。

输出的方案 JSON 要对下游 agent 自洽:只读 spec 不回看源码也能把任务做下去。审查员同时看 spec 和源码,但只把源码当 reference,优先审 spec 本身,源码只在 spec 不清楚时作补充判断。

## 仓库内容

- 从真实代码里抽出可独立验证的小任务,而不是从无到有编题
- 判断哪段源代码值得做规范化重构,把昂贵的审查链路只花在值得的样本上
- 把任务边界用 draft → 4 critic → 共识合成 → curator 这条链收紧,避免任务边界漂移
- 4 个 critic 各盯一类典型偏移:**边界漂移**(多决策捆绑)、**源码泄漏**(题面预设答案)、**范围漂移**(一次处理多对象)、**过度干涉**(替 agent 决定步骤)
- 用同源 A/B 对照(开 / 关审查员)量化审查员步骤的实际作用

## 输入 / 输出

**输入**:一份 Python 源文件,或一棵 source 树。

**输出**:`artifacts/` 下一组 JSON,完整记录源码摘要、检索命中、triage 判定、任务草案、4 个 critic 反馈、共识修订、终稿、最终判定与失败标签。

每轮 run 的关键 artifact:

- `run_bundle.json`——总装,聚合下面所有 JSON
- `source_summary.json`——源码摘要 + 风险点
- `retrieval_hits.json`——记忆检索命中
- `source_triage.json`——是否值得做规范化重构
- `critic_reviews.json`——4 个 critic 各自的 blocking issues 与 suggestions
- `consensus_review.json`——4 个 critic 反馈合成的可执行修订方案
- `curated_task.json`——审查员收敛后的终稿(规则层判定前最重要的中间产物)
- `anchor_assessment.json`——规则函数判定 + 失败标签

## 快速开始

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
cp .env.example .env   # Windows: copy .env.example .env
```

填 `.env`:

- `TASK_FORGE_API_KEY`(必填)
- `TASK_FORGE_BASE_URL`(可选,OpenAI 兼容端点)
- `TASK_FORGE_CHAT_MODEL`(可选)
- `TASK_FORGE_SOURCE_ROOT`(可选,源码扫描根目录;不填走 `examples/source`)

常用命令:

- `make run` 跑一个示例
- `make test` 跑单测
- `make run-no-critics` 关掉审查员,跑同源 A/B 对照的另一半
- `make batch` 批量扫源码树

也可以直接命令行:

```bash
python -m task_forge_v2.run_pipeline --source examples/source/example_pair_promotion.py
python -m task_forge_v2.run_batch --source-root path/to/source_tree --limit 4
```

## 设计要点

1. **先 triage 再花钱**:不值得规范化重构的源直接走 6 节点快速通道归档(1 次 LLM 调用、约 27 秒),只把昂贵审查链路花在值得的脚本上
2. **拆开 critic 职责**:生成、挑错、裁决分到不同 prompt,4 个 critic 并行各盯一类偏移
3. **curator 强行收敛**:把任务压回单一边界、单一规划单元,避免一个任务捆多个决策
4. **规则函数判定,LLM 不参与**:`postprocess` 先整形方案、补默认值、去答案泄漏;`assess_task` 按硬性条件输出判定与失败标签。同一份方案重跑结果一致,生成与验收解耦

## 仓库结构

- `task_forge_v2/`——核心 pipeline 代码
- `knowledge/`——本地检索知识库
- `examples/source/`——两个示例脏脚本,仅用于 smoke test 跑通整条 graph
- `tests/`——单测
- `scripts/`——离线工具脚本,目前只放 `run_dryrun_plan_check.py`(自洽性 dry-run,不在主链路)
- `artifacts/`——运行产物(运行后生成)
- `report_data/`——dry-run 输出 + 一份 `sample_dryrun_plan_check.*` 样本
- `.cache/retrieval_index/`——检索缓存(运行后生成)

## 自洽性 dry-run(不在主链路)

主链路是 LLM 当作者、规则当裁判,只到 `assess_task` 出分桶。这一步换角度:拉 4 个跨厂商模型当下游执行者按 spec 给出执行计划,看 spec 在跨家族 runner 上是否站得住。

- 测试者:`Kimi K2.5`、`GLM 4.7 Flash`、`DeepSeek V4 Flash`、`MiMo V2.5 Pro`,各自读 `curated_task.json` + 可见源码,产出执行计划 JSON
- 裁判:`gpt-5.4-mini`,按 boundary / grounding / execution / policy 四档打分,聚合成 `pass / partial / fail`
- `MiMo` 与起草模型同家族,单列作参考,不计入跨厂商汇总

跑法(测试者和裁判走主管线同一个 OpenAI 兼容端点):

```bash
# ARTIFACTS 填 artifacts/ 下任意 run id,可填多个
make dryrun-check ARTIFACTS="<artifact_id>"
```

产出落在 `report_data/<run-prefix>_dryrun_plan_check.{json,csv,md}`,`report_data/sample_dryrun_plan_check.md` 是一份历史 run 的样本输出。

## 测试覆盖

`tests/` 只覆盖确定性层:mining 的 AST 抽取、postprocess 的 shape 清洗、triage 规则、validators、assessment 分桶、telemetry。LLM 节点(generator / 4 critic / consensus / curator)的输出由下游确定性层接住,所以不 mock,靠 `make run` 走真实 API 做集成验证。

`scripts/run_dryrun_plan_check.py` 是一次性自洽性检查,产物归 `report_data/`,不进单测。
