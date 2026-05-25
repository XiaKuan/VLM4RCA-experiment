# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

VLM4RCA：基于 VLM + LLM 的根因分析系统，使用 OpenRCA 数据集。

## 常用命令

```bash
# 安装依赖
uv sync

# 运行测试
uv run pytest tests/ -v

# 运行单个测试
uv run pytest tests/test_xxx.py::test_func -v

# 代码检查
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## 数据说明

OpenRCA Bank 数据集位于 `data/OpenRCA/Bank/`：

- `query.csv`：任务描述和评分标准
- `record.csv`：根因标注（组件、原因、时间戳）
- `cases/`：136 个故障案例，每个包含 `case_meta.json`、`metrics.csv`、`logs.csv`、`traces.csv`
- `telemetry/`：原始遥测数据，按日期组织

### 关键数据字段

- `case_meta.json`：`instruction`（任务描述）、`matched_faults`（根因）、`evidence_components`（相关组件）
- `metrics.csv`：宽表格式，列为 `组件__指标名`，行为时间戳
- 任务类型：task_1（时间定位）、task_2（组件定位）、task_3（原因定位）、task_4（综合定位）

## 开发约定

- Python >= 3.12，使用 uv 管理依赖
- 配置通过 YAML 文件管理，敏感信息用环境变量
- VLM/LLM 调用使用 OpenAI API 兼容接口

## Worktree 开发规范

修复问题或开发新功能时，**必须**使用 git worktree 从 `main` 创建隔离分支，禁止直接在 `main` 上修改。

```bash
# 创建 worktree（基于 main）
git worktree add .claude/worktrees/<feature-name> -b <feature-name> main

# 开发完成后合并回 main
git checkout main && git merge <feature-name>

# 清理 worktree
git worktree remove .claude/worktrees/<feature-name>
```

命名规范：`fix/<issue-desc>` 或 `feat/<feature-desc>`，例如 `fix/s2-scoring`、`feat/eval-subset`。

## 架构原则（来自前项目经验）

### 数据层

- **Dataset Adapter 模式**：所有数据集特定逻辑集中在 adapter 中，换数据集只需新增 adapter
- **Config immutable**：配置加载后 freeze，防止变异 bug

### LLM/VLM 交互

- **Structured output 优先**：用 JSON Schema 约束输出，减少自然语言规则的脆弱性
- **Quarantine 模式**：LLM 输出不做 strict validation，解析失败的字段标记为 null 而非报错
- **Provider 抽象层**：隔离不同 provider（DashScope/OpenAI）的差异，不泄漏到业务层
- **反锚定规则**：明确告诉 LLM 数值大小 ≠ 根因概率，anomaly_score 需 rank-based 归一化后传入

### Prompt 管理

- **Rule 编号系统**：每条 prompt rule 有编号，关联一个失败 case
- **Minimal prompt 起步**：从正向指令开始，每次发现失败模式先尝试 JSON Schema，再加 prompt 规则
- **Prompt 改动必须有 regression 测试**：防止规则被意外删除

### 评分层

- **分数 rank-based 归一化**：不传原始数值给 LLM
- **min_sample_size guard**：样本不够给 0 分
- **两级数据过滤**：组件过滤 → 非空率过滤

## 测试要求

- **TDD**：先写测试再实现，尤其是边界条件
- **Protocol-based DI**：FakeClient 实现协议接口，不依赖真实 API
- **三场景模式**：正常输入 → 正常输出、缺失前置 → StageError、边界输入 → 不崩溃
- **Prompt 内容断言**：验证关键规则没有被意外删除
