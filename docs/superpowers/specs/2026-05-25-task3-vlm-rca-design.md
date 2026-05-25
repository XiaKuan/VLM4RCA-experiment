# VLM4RCA Task 3 Reason Localization Design

Date: 2026-05-25
Status: Draft for user review
Scope: OpenRCA task_3 reason localization

## 1. Goal

第一版系统聚焦 OpenRCA Bank 数据集中的 task_3 原因定位。目标不是先实现完整端到端 RCA，而是验证在候选组件集合已给定或可生成的前提下，VLM 从指标图像中抽取的视觉事实是否能提升根因原因判断。

主流程采用“VLM 事实抽取 + LLM 裁决”：

1. 读取 OpenRCA case。
2. 生成候选组件集合。
3. 用专家槽位和统计兜底选择少量代表指标。
4. 生成受控数量的指标图。
5. VLM 只从图像中抽取结构化视觉事实。
6. LLM 基于候选组件、视觉事实、结构化指标摘要、trace/log 事实上下文和标签空间输出原因排序。
7. 评估 OpenRCA 官方 Correct / Partial，并额外报告 top@1、top@3、top@5。

第一版明确不让 VLM 直接输出根因、置信度、统计显著性或因果判断。VLM 的职责是“看图并结构化描述可见事实”，RCA Judge 的职责是“在封闭候选和标签空间内推理并排序”。

## 2. Non-Goals

- 不实现 task_1 时间定位、task_2 组件定位、task_4 综合定位的完整闭环。
- 不把 trace 或调用链画成图交给 VLM；第一版 trace/log 只作为结构化事实上下文。
- 不让统计模块主导根因裁决。统计特征在该数据集上较弱，第一版统计主要用于指标筛选、text-only baseline 摘要和证据事实生成。
- 不把 oracle 候选结果解释为真实端到端 RCA 能力。oracle setting 只用于 diagnosis-only / upper-bound 分析。

## 3. Experimental Settings

### 3.1 VLM Main Pipeline

VLM 主流程包含图像输入：

1. Candidate Provider 生成候选组件。
2. Metric Selector 选择代表指标。
3. Figure Builder 生成最多 4 张图。
4. VLM Observation Extractor 输出结构化视觉事实。
5. RCA Judge 输出 ranked predictions。

### 3.2 Text-Only Baseline

Text-only baseline 不使用图像和 VLM。它复用同一 Candidate Provider、Metric Selector、Trace/Log Context Builder、RCA Judge、标签空间、prompt 模板和解码参数。唯一差异是 Judge 输入中没有 VLM observations。

### 3.3 Deterministic Text-Summary Baseline

为避免把 VLM 增益误解为“更好的文本摘要”，第一版增加一个 deterministic text-summary baseline。它从同一批 selected metrics 中用程序生成与 VLM observations 共享核心枚举字段的事实摘要，并映射为同构 `ObservationEvidence`。该 baseline 用于比较“程序化事实抽取”与“VLM 视觉事实抽取”。

程序化字段判定使用固定规则：

- `direction`: 比较 fault/post 窗口与 baseline 窗口的中位数或 p95。默认 robust threshold 为 `3.0 * 1.4826 * MAD_baseline`，低方差序列使用 `epsilon` guard。超过阈值时标记为 `increase` 或 `decrease`，否则为 `none`。
- `pattern_type`: 根据阈值越界点的形态标记为 `spike`、`step_increase`、`step_decrease`、`gradual_increase`、`gradual_decrease`、`oscillation`、`flatline`、`missing_data` 或 `no_visible_change`。
- `time_relation_to_fault`: 第一次明显变化相对 fault time 的位置，标记为 `before_fault`、`at_fault`、`after_fault`、`spans_fault` 或 `unclear`。
- `duration_pattern`: 根据连续越界样本数标记为 `transient`、`sustained`、`intermittent`、`periodic`、`single_point` 或 `unknown`。

这些规则不用于最终裁决，只用于生成与 VLM facts 可比较的结构化事实。

### 3.4 Oracle and Heuristic Candidate Settings

系统支持两种候选来源：

- `oracle`: 使用数据集提供的 evidence/root-cause 相关组件构造候选集。该 setting 标记为 diagnosis-only / upper-bound，用于隔离原因裁决能力。
- `heuristic`: 使用关键预警指标和统计规则筛选候选组件，用于模拟真实 SRE 流程。

`oracle` 候选生成属于实验 harness 的特权步骤。它可以读取 `EvaluationLabelBundle` 来构造 `CandidateSet`，但构造完成后，下游 Metric Selector、Figure Builder、VLM、Trace/Log Context Builder 和 RCA Judge 只能看到候选组件列表及 `is_oracle` 标记，不能看到 ground truth、reason label 或 evidence annotations。

评估报告中 oracle 和 heuristic 必须分表。heuristic 结果额外报告 candidate recall 上限、候选集大小和候选为空比例。若真因不在 heuristic 候选集中，该 case 的 Judge 不可能 Correct。

### 3.5 Reason Label Taxonomy and Aliases

Judge 的 `reason_label` 来自版本化的封闭 taxonomy。第一版使用一个生成后可人工审阅的 label catalog，来源优先级如下：

1. OpenRCA task_3 的全局评分说明或任务定义中出现的合法原因标签。
2. 训练/开发 split 中 `record.csv` 和 `case_meta.json` 根因记录的 unique reason/cause 字段，用于补全实际出现但官方说明未枚举的标签。
3. 手工维护的通用 SRE category 映射，例如 latency、cpu、memory、disk、network、error、traffic。

Taxonomy 构建是数据集级预处理步骤，只产出“可能的标签集合”和 alias 映射，不产出 case 到标签的映射。运行单个 case 时，RCA Judge 只能看到 taxonomy 全表和候选组件，不能看到该 case 的 ground truth label。

每个 reason label 使用稳定 canonical id，例如 lowercase snake_case。Alias mapping 来自三类来源：

- 官方任务定义或训练/开发 split 中原始 reason/cause 字符串的规范化结果。
- task_3 评分说明中的同义表达。
- 人工维护的 alias YAML。

默认报告使用 non-transductive taxonomy/alias：不得从评测 case 的答案字段、matched faults 或 accepted answer 中派生 label 或 alias。如果必须从全量 `record.csv` / `case_meta.json` 构建 taxonomy 来覆盖数据集闭集标签，该 run 必须标记为 `transductive_label_catalog=true`，并与默认结果分表报告。

如果 OpenRCA 当前实验划分没有明确 train/dev split，默认 non-transductive catalog 只能使用官方任务定义和人工维护 alias YAML；任何从全量答案文件抽取 label/alias 的结果都归入 transductive run。

Alias 变更必须版本化，并有 evaluator golden regression test 覆盖。Sidecar 记录 taxonomy version、alias version、label mapping hash 和 `transductive_label_catalog` 标记。

### 3.6 Heuristic Candidate Seed Algorithm

Heuristic Candidate Provider 第一版使用固定的可复现 seed 规则，不在实现阶段自由发挥：

1. 从 metrics 宽表列名中解析 `(component, metric_name)`。默认按 `组件__指标名` 分割，解析失败的列进入 unknown component/metric namespace 并记录 warning。
2. 复用专家槽位规则将每个 metric 归入 latency、traffic、error、cpu、memory、disk、network 或 unknown。
3. 对每个 metric 计算窗口差异分数：baseline 与 fault/post 窗口的 robust median shift、p95 shift 和持续越界比例，取最大值作为 `metric_anomaly_score`。robust shift 使用 `abs(stat_fault_or_post - stat_baseline) / (1.4826 * MAD_baseline + epsilon)`；持续越界比例使用 baseline median ± `3.0 * 1.4826 * MAD_baseline`。分数只用于候选排序和选图，不作为 RCA 裁决证据。
4. 每个 component 的类别分数取该类别 top 1-2 metric 分数的最大值。
5. Component score 取类别分数的加权最大值，默认权重为 latency/error 1.2、cpu/memory/disk/network 1.0、traffic 0.8、unknown 0.6。
6. 输出 top `candidate_top_k` 个组件，默认 `candidate_top_k=5`。若所有 component score 低于 `low_signal_threshold=1.0`，仍输出 top-k 但记录 `low_signal_candidates=true`。Unknown metric namespace 可进入 sidecar/debug，但 unknown component 不得进入 `CandidateSet`；无可解析真实组件时输出空候选。
7. Tie-break 固定为 component score 降序、最高优先级类别、canonical component 名字典序。

这些规则是第一版 heuristic baseline，不声明为最优候选算法。后续可以通过消融替换为随机候选、仅专家槽位候选或弱统计候选。

## 4. Architecture

### 4.1 Dataset Adapter

读取 OpenRCA `query.csv`、`record.csv`、`case_meta.json`、`metrics.csv`、`logs.csv`、`traces.csv`。数据集字段和路径细节只留在 adapter 内部。

Adapter 输出两个分离对象：

- `RuntimeCaseBundle`: pipeline 可见，不含 ground truth。
- `EvaluationLabelBundle`: 只给 Evaluator 使用，包含官方答案和别名映射。

这种隔离防止 Metric Selector、Figure Builder、VLM 和 RCA Judge 误读标签。唯一例外是 diagnosis-only 的 Oracle Candidate Provider，它作为实验 harness 在 runtime 前使用标签构造候选集；它输出后不得把标签字段透传给下游。

### 4.2 Candidate Provider

负责生成 `CandidateSet`。支持 `oracle` 和 `heuristic` 模式。输出必须记录候选来源、排序、去重、别名归一化和 oracle 标记。Heuristic 模式必须实现 §3.6 的 seed algorithm，并把 score、category contribution 和 low-signal 标记写入 sidecar。

### 4.3 Metric Selector

使用专家槽位 + 统计兜底选择指标。核心规则：

1. 每类指标定义 2-4 个专家槽位。
2. 实际列名用关键词/正则匹配槽位。
3. 同一槽位匹配多个指标时，按异常分数选择 top 1-2。
4. 某类无专家槽位命中时，从 unknown 或该类剩余指标中按统计异常补充。
5. 每个候选组件每类最多 1-3 个代表指标。
6. 图像预算控制在 VLM 稳定处理范围内。

专家槽位优先级基于通用 SRE 排障经验：尾延迟优先于平均值，比例型错误优先于原始计数，资源利用率优先于低层吞吐计数。该优先级是版本化配置；变更必须带 metric selector regression test。

专家槽位优先级：

- latency: p95/p99 latency > avg latency > duration
- traffic: request rate / throughput > request count
- error: error rate > 5xx rate > exception count
- cpu: cpu usage/utilization > throttling > load
- memory: working set/RSS > usage percent > OOM/restart
- disk: io wait/latency > utilization > read/write bytes
- network: retransmit/drop > latency > rx/tx bytes

### 4.4 Figure Builder

最多生成 4 张图：

- 1 张候选组件横向对比图。
- 最多 3 张 top 组件细节图。

图像数量和 panel 数量分开控制。第一版默认 `max_images=4`、`max_panels_per_image=6`、`max_total_panels=18`。Metric Selector 可以选出多于 18 个候选指标，但 Figure Builder 只绘制 Figure Planner 选中的指标：

1. 覆盖最多 top 3 candidate components。
2. 每个组件优先保留 latency/error 槽位中的最高优先级指标。
3. 剩余 panel 给该组件最高异常分数的资源类指标。
4. Tie-break 使用类别优先级、异常分数、`metric_id` 字典序。
5. 未入图指标仍保存在 `selected_metric_set` 和 `metric_summary` 中，可供 text-only baseline 使用。

第一版默认一个 panel 只画一个代表指标。每个 panel 必须有稳定 `panel_id`，并能追溯到唯一 `metric_id`。如果后续允许多曲线 panel，必须引入 `series_id`、`legend_label`、`color`，且 VLM 只能引用 manifest 中存在的 id。

图中需要清晰标记组件名、指标名、时间窗口和故障时间参考线。时间轴默认使用相对时间，`x=0` 表示 fault time，刻度用 “minutes before/after fault” 或等价相对单位。绝对时间戳只保存在 manifest/sidecar 中，不作为图上主要读数。图像设计的目标是帮助 VLM 读取曲线结构，而不是在图中塞入自然语言解释。

### 4.5 VLM Observation Extractor

VLM 只接收图像、`FigureManifest` 中允许引用的 `panel_id/series_id`、时间窗口说明和严格 JSON Schema。VLM 不接收 ground truth、OpenRCA 标签空间、trace/log 或候选外信息。

输出是结构化视觉事实，不是根因判断。schema 使用 `additionalProperties=false`，并禁止因果词和推理字段。

### 4.6 Trace and Log Context Builder

Trace/log 不画图，只生成结构化事实上下文。字段必须是事实型，例如调用边、窗口、调用数、延迟变化、错误变化、日志事件类别和计数变化。

禁止进入 Judge 输入的内容包括 case 名、注入原因、官方标签名、ground truth 字段，以及 `root`、`suspect`、`affected`、`culprit` 等解释性词。

### 4.7 RCA Judge

LLM 是最终裁决者，但只能在给定候选组件和封闭 reason label taxonomy 内输出。Judge 不看图像，只看结构化输入。

所有 setting 共享以下输入：

- `CandidateSet`
- complete `MetricSummary`
- `TraceContext` / `LogContext`
- label mapping
- reason label taxonomy

Observation block 按 setting 切换：

| Setting | MetricSummary | ObservationEvidence | Purpose |
| --- | --- | --- | --- |
| VLM main | yes | `source_type=vlm`, mapped from `VlmObservation` | Test whether visual facts add value over the shared structured metric summary. |
| Text-only baseline | yes | no | Measure the shared non-visual input without any observation block. |
| Deterministic text-summary baseline | yes | `source_type=deterministic_text`, derived from `MetricSummary` | Control for the benefit of an additional structured fact block without visual perception. |

Deterministic `ObservationEvidence` is a derived view of `MetricSummary`, not independent evidence. It is intentionally included to match the VLM main pipeline's extra observation channel while keeping the raw non-visual inputs identical. Judge 输出排序列表，每个预测必须引用已有证据 id。`rationale` 只能解释这些引用，不允许引入输入中没有的新事实。

### 4.8 Evaluator / Reporter

Evaluator 读取 `EvaluationLabelBundle` 和 predictions，计算：

- OpenRCA 官方 Correct / Partial。
- `top@1/3/5 reason`。
- 可选诊断扩展 `diagnostic_top@1/3/5`。

Reporter 写主结果和 sidecar。主结果保持评估友好，sidecar 保存复现和分析所需的轻量中间产物。

## 5. Data Contracts

### 5.1 RuntimeCaseBundle

Pipeline 的唯一 case 输入，不含 ground truth。字段：

- `case_id`
- `task_type`
- `sanitized_instruction`
- `time_window_spec`
- `metrics_frame`
- `logs_frame`
- `traces_frame`
- source references

原始 instruction 只进入审计 sidecar，不直接给 VLM 或 Judge。

`sanitized_instruction` 是可选的短任务描述，只能保留任务类型和非答案性约束，例如 “reason localization for one OpenRCA case”。Sanitizer 必须删除或替换 case 名、故障注入原因、matched faults、evidence components、accepted aliases、具体组件答案、具体 reason label，以及任何从 `case_meta.json` 标签字段复制出的文本。若 sanitizer 无法确认安全，runtime 使用空 instruction。

### 5.2 EvaluationLabelBundle

默认只给 Evaluator 使用。诊断性 oracle 候选生成可以在实验 harness 中读取它来构造 `CandidateSet`，但不能把标签字段透传到 runtime pipeline。字段：

- OpenRCA task_3 official answer
- `official_required_elements`: first-version task_3 reason-localization uses `["reason_label"]`
- `canonical_component`
- `canonical_reason_label`
- accepted aliases

任何非评估模块不能读取该对象。

### 5.3 CandidateSet

Candidate Provider 输出。每个 `CandidateComponent` 包含：

- `candidate_id`
- `canonical_component`
- `raw_component`
- `source`
- `rank`
- `score` optional
- `selection_reason`
- `is_oracle`

`candidate_id` 是 runtime pipeline 内的稳定组件引用键，格式建议为 `cand:{case_id}:{rank}:{canonical_component}`。下游指标、图像、observation、prediction 都通过 `candidate_id` 引用候选组件，不再另设第二套组件引用字段。

### 5.4 ReasonLabelTaxonomy

封闭原因标签空间。字段：

- `taxonomy_version`
- `reason_labels`: canonical reason label list
- `sre_category_mapping`: reason label 到通用 SRE category 的映射
- `alias_map`: raw label / synonym 到 canonical reason label 的映射
- `source_refs`: query/task definition/config/training split 来源引用
- `transductive_label_catalog`
- `mapping_hash`

RCA Judge 只能输出 `reason_labels` 中存在的 canonical label。

### 5.5 TimeWindowSpec

统一时间窗口定义：

- `baseline_window`
- `fault_window`
- `post_window`
- unit
- timezone
- sampling interval
- inclusive/exclusive boundary rules

图像、metric summary 和 VLM `time_relation_to_fault` 都引用同一个窗口定义。

### 5.6 SelectedMetricSet

每个 selected metric 包含：

- `metric_id`
- `candidate_id`
- `raw_column`
- `sre_category`
- `slot`
- `display_name`
- `selection_status`
- `selection_reason`
- `missing_rate`
- `sampling_interval`
- `unit` optional
- `metric_namespace` optional

异常分数、丢弃指标和 tie-break 细节只进入 debug sidecar。

### 5.7 FigureManifest

字段：

- `image_id`
- `image_path`
- `image_hash`
- `figure_type`
- `panels`

每个 panel 包含：

- `panel_id`
- `candidate_id`
- `metric_id`
- `sre_category`
- `time_window`
- layout reference

### 5.8 VlmObservation

严格事实 schema。字段：

- `observation_id`
- `image_id`
- `panel_id`
- `series_id` optional
- `pattern_type`
- `direction`
- `time_relation_to_fault`
- `duration_pattern`
- `readability_status`

`observation_id` 格式为 `vlmobs:{case_id}:{image_id}:{panel_id}:{index}`。同一 panel 可以有多个 observation，但每个 observation 必须引用 manifest 中存在的 `panel_id`。

字段使用封闭枚举：

- `pattern_type`: `stable`, `spike`, `drop`, `step_increase`, `step_decrease`, `gradual_increase`, `gradual_decrease`, `oscillation`, `level_shift`, `missing_data`, `flatline`, `no_visible_change`, `unknown`
- `direction`: `increase`, `decrease`, `mixed`, `none`, `unknown`
- `time_relation_to_fault`: `before_fault`, `at_fault`, `after_fault`, `spans_fault`, `unrelated_to_fault_window`, `unclear`
- `duration_pattern`: `transient`, `sustained`, `intermittent`, `periodic`, `single_point`, `unknown`
- `readability_status`: `readable`, `partially_readable`, `unreadable`, `axis_unclear`, `legend_unclear`, `overplotted`, `low_resolution`

禁止字段或语义包括 `confidence`、`root_cause`、`reason`、`statistical_support`、`culprit`、`likely`、`because`、`caused_by`、`primary`。解析层发现禁用字段或因果表达时重试或拒收。

### 5.9 MetricSummary

程序生成的结构化事实摘要，用于 text-only baseline 和公平对照。字段：

- `metric_summary_id`
- `metric_id`
- `candidate_id`
- `baseline_window_summary`
- `fault_window_summary`
- `post_window_summary`
- `pattern_type`
- `direction`
- `time_relation_to_fault`
- `duration_pattern`
- `summary_method`
- `summary_warnings`

`metric_summary_id` 格式为 `ms:{case_id}:{metric_id}`。`pattern_type`、`direction`、`time_relation_to_fault`、`duration_pattern` 与 `VlmObservation` 使用同一套枚举。

### 5.10 ObservationEvidence

RCA Judge 的 evidence block 使用统一的 `ObservationEvidence`，避免因 prompt 字段差异引入 baseline 混杂。VLM 主流程和 deterministic baseline 都先映射成这个结构：

- `observation_evidence_id`
- `source_type`: `vlm` or `deterministic_text`
- `candidate_id`
- `metric_id`
- `pattern_type`
- `direction`
- `time_relation_to_fault`
- `duration_pattern`
- `source_ref`: VLM 使用 `observation_id`，deterministic 使用 `metric_summary_id`
- `readability_status`: VLM 填写枚举值，deterministic 固定为 `not_applicable`

`ObservationEvidence.readability_status` 的取值为 `VlmObservation.readability_status` 枚举加 `not_applicable`。Judge prompt 面向 `ObservationEvidence`，因此主流程与 deterministic baseline 的字段名和枚举一致。原始 `VlmObservation` 和 `MetricSummary` 仍写入 sidecar 供审计。

### 5.11 TraceContext and LogContext

`TraceContext` 使用事实字段：

- `trace_edge_id`
- source
- target
- window
- call count
- latency change
- error change

`LogContext` 使用事实字段：

- `log_event_id`
- component
- window
- event category
- count change
- short sanitized summary

### 5.12 RcaPrediction

最终排序项：

- `rank`
- `candidate_id`
- `canonical_component`
- `sre_category`
- `reason_label`
- `metric_summary_ids`
- `observation_evidence_ids`
- `trace_edge_ids`
- `log_event_ids`
- `rationale`

`candidate_id` 必须来自 `CandidateSet`，`reason_label` 必须来自封闭 taxonomy。

### 5.13 Sidecar

默认 `analysis` 级别保存：

- `run_metadata`: commit, config hash, dataset version, model id, schema version, temperature, seed or seed support note
- `candidate_set`
- `reason_label_taxonomy`
- `selected_metric_set`
- `figure_manifest`
- `vlm_observations`
- `metric_summary`
- `observation_evidence`
- `trace_context`
- `log_context`
- `label_mapping`
- `ranked_predictions`
- parse status/warnings
- omitted counts

`debug` 级别额外保存完整 prompt、raw request/response、解析错误细节、完整统计分数和被丢弃指标。

## 6. Model Interaction

所有模型调用都走结构化输出。prompt 模板和 schema 必须版本化，例如：

- `vlm_observation_v1`
- `rca_judge_v1`
- `deterministic_text_summary_judge_v1`

sidecar 记录模板 id、schema version、model id、temperature、seed 或 provider 不支持 seed 的说明。

VLM prompt 只要求报告图像可见事实。RCA Judge prompt 要求在候选集合和标签空间内输出排序，并引用输入中的 evidence ids。主流程、text-only baseline 和 deterministic text-summary baseline 共用同一个 Judge prompt：

- VLM 主流程提供 `ObservationEvidence`，其中 `source_type=vlm`。
- Deterministic text-summary baseline 提供同构 `ObservationEvidence`，其中 `source_type=deterministic_text`。
- Text-only baseline 不提供 observation evidence，只提供 `MetricSummary`、trace/log context、候选和 taxonomy。

除 evidence block 内容外，Judge 的候选、标签空间、prompt 模板和解码参数保持一致。

模型参数固定为低随机性，例如 temperature 0 或接近 0。

## 7. Evaluation

OpenRCA task_3 的官方主评估只使用该 query 的 official required elements。第一版 reason-localization setting 将官方答案形状固定为：

```text
official_task3_item = reason_label
```

组件字段保留用于 diagnosis-only 分析和错误诊断，但不混入官方 task_3 主指标。诊断性扩展指标使用：

```text
diagnostic_item = (canonical_component, reason_label)
```

OpenRCA Correct / Partial 基于 `official_task3_item` 和同一套别名映射计算。额外 top-k 默认报告 `top@k reason`；同时可以分表报告 `diagnostic_top@k exact`，用于分析候选组件和原因组合是否同时正确。

Evaluator 使用 §3.5 的 versioned alias mapping 将 raw prediction、dataset labels 和 accepted aliases 归一化为 canonical reason label。Alias mapping 不在评估时动态学习；任何 alias 变更都必须生成新的 mapping hash 并重新跑 evaluator golden tests。

指标：

- `Correct`: 复现 OpenRCA 官方 full solve 规则。
- `Partial`: 复现 OpenRCA 官方至少一个 required element 正确的规则。
- `top@1 reason`, `top@3 reason`, `top@5 reason`: 前 k 个去重预测中存在 reason_label 匹配。
- `diagnostic_top@1 exact`, `diagnostic_top@3 exact`, `diagnostic_top@5 exact`: 前 k 个去重预测中存在 `(canonical_component, reason_label)` 完全匹配。
- `diagnostic_top@1 partial`, `diagnostic_top@3 partial`, `diagnostic_top@5 partial`: 前 k 个预测中至少一个 diagnostic required element 匹配。

处理规则：

- official top-k 先按 canonical reason label 去重，diagnostic top-k 先按 diagnostic item 去重，重复项保留最高 rank。
- 候选外组件丢弃。
- 非法 reason label 丢弃并记录 parse warning。
- 空预测全 miss。
- oracle 和 heuristic 分表报告，不合并成单一主结论。

## 8. Error Handling

### 8.1 Dataset Errors

缺少 `metrics.csv`、`case_meta.json` 或 task_3 标签时，case 标记为 `skipped_with_reason`。`fault_time` 越界、时间列不可解析、采样间隔异常时记录 warning。无法构造 `TimeWindowSpec` 时跳过 case。

### 8.2 Candidate Errors

候选为空时不调用后续模型，输出空预测并记录 `empty_candidate_set`。`oracle` run 必须记录 `diagnosis_only=true`。

### 8.3 Metric Selection Errors

指标缺失、NaN 过多、零方差、槽位未命中都记录 `selection_status`，不中断整个 case。候选组件没有任何可画指标时，保留候选但标记 `no_metric_selected`。

### 8.4 Figure Errors

图像预算最多 4 张，并同时受 `max_panels_per_image` 和 `max_total_panels` 限制。每个 panel 必须追溯到一个 metric。图像生成失败时标记该 image `failed`，VLM 不处理该图。如果所有图失败，主流程降级为空 VLM observations，但仍可运行 metric summary baseline。

### 8.5 VLM Parse Errors

非法 JSON、禁用字段、因果词、未知 `panel_id/series_id`、字段不在枚举内时，最多重试固定次数。仍失败则该图记为 `vlm_parse_failed`。

### 8.6 Trace/Log Errors

trace/log 缺失时输出空 context。上下文超长时按固定 cap 截断，并记录 omitted count。泄漏过滤需要覆盖大小写、snake_case、camelCase 和常见变体，例如 `rootCause`、`root_cause`、`suspected_component`、`affected_by`、`culprit`、`injected_fault`、`fault_type`。

### 8.7 Judge Parse Errors

Judge 必须输出排序列表。未知组件、候选外组件、未知 reason label、重复 reason label 或重复 diagnostic item 按固定规则处理：候选外项丢弃，重复项保留最高 rank，非法标签进入 parse warning 且不计命中。有效预测为空时，该 case top-k 全部 miss，官方指标按空答案处理。

## 9. Testing Plan

### 9.1 Contract Tests

- `RuntimeCaseBundle` 不含 `ground_truth`、`matched_faults`、`evidence_components`。
- `sanitized_instruction` 删除 case 名、注入原因、组件答案、reason label、accepted aliases 和标签字段文本；无法确认安全时为空。
- `EvaluationLabelBundle` 只能由 Evaluator 使用。
- `ReasonLabelTaxonomy` 的 label、alias 和 mapping hash 稳定；默认 non-transductive catalog 不从评测答案派生 alias，transductive catalog 必须显式打标。
- `VlmObservation` schema 禁止额外字段、禁用字段和因果词。
- `FigureManifest` 的 `panel_id` 能和 VLM observations 做 round-trip 校验。
- `CandidateSet` 能稳定处理别名、去重、oracle 标记和排序。
- `RcaPrediction` 的 `candidate_id` 必须来自候选集，`reason_label` 必须来自封闭标签空间，非空 evidence id 必须存在。

### 9.2 Module Behavior Tests

- 指标槽位匹配符合专家优先级。
- 槽位无命中、NaN 多、零方差、多指标 tie-break 有确定结果。
- 每个 case 最多 4 张图，默认一个 panel 一个指标。
- Heuristic Candidate Provider 在固定 fixture 上输出稳定 top-k、分数和 low-signal 标记；unknown component 不进入 CandidateSet。
- Trace/log 泄漏过滤移除标签名、case 名、解释性词及常见变体。
- VLM 非法 JSON、未知 panel、禁用字段、因果词触发重试或拒收。
- Judge 未知组件、非法标签、重复预测、空输出按固定规则处理。
- Judge rationale 抽样检查不能包含未被 evidence ids 引用的组件名、指标名或新事实。
- analysis sidecar 包含复现索引，debug sidecar 才包含 raw prompt/response。

### 9.3 Experiment-Level Tests

- Oracle diagnosis-only setting 报告 Correct、Partial、top@1/3/5 reason、diagnostic_top@1/3/5、candidate set size。
- Heuristic end-to-end setting 报告 candidate recall 上限和候选为空比例。
- 主流程与 baseline 的候选、完整 MetricSummary、trace、Judge、标签空间和解码参数完全一致；VLM main 与 deterministic baseline 额外提供同构 ObservationEvidence，text-only baseline 不提供 observation block。
- Deterministic text-summary baseline 与 VLM observations 使用同构事实字段。

### 9.4 End-to-End Integration Tests

- 使用一个完整 fixture case 和 fake VLM/LLM client 跑通 VLM 主流程，验证主结果、sidecar 和 evaluator 输出。
- 使用同一个 fixture 跑通 text-only baseline 和 deterministic text-summary baseline，验证三者共享 candidate set、selected metrics、trace/log context、taxonomy 和 Judge 配置。
- 验证 sidecar 中 `candidate_id`、`metric_id`、`panel_id`、`observation_evidence_id`、`trace_edge_id`、`log_event_id`、`ranked_predictions` 的引用完整性。
- 使用 evaluator golden case 验证 Correct、Partial、top@1 reason、top@3 reason、top@5 reason、diagnostic_top@k、非法预测、重复预测和空预测。

## 10. Risks and Mitigations

- VLM 增益可能来自文本重写而非视觉理解。Mitigation: 增加 deterministic text-summary baseline。
- Oracle 候选可能高估能力。Mitigation: oracle 标记为 diagnosis-only / upper-bound，并与 heuristic 分表报告。
- Trace/log 可能泄漏标签。Mitigation: 使用 sanitized facts、禁用解释性字段并测试过滤。
- Taxonomy/alias 从评测答案派生会造成 transductive evaluation。Mitigation: 默认使用 non-transductive catalog；全量派生只作为单独 transductive run 分表报告。
- VLM 可能隐式推理。Mitigation: 严格 schema、禁用词过滤、抽样审计 observations。
- Deterministic text-summary baseline 可能实现复杂且带入程序规则偏差。Mitigation: 明确字段判定规则，记录 summary_method，并用同构 `ObservationEvidence` 控制 prompt 差异。
- 4 张图预算可能覆盖不足，且不同 VLM 模型的多图和分辨率能力不同。Mitigation: 声明为第一版成本约束，把 image/panel budget 配置化，并可做 1/2/4 图预算敏感性分析。
- 统计选图可能成为隐藏混杂变量。Mitigation: 报告 metric selector 覆盖率，并可增加随机/弱统计/仅专家槽位消融。

## 11. Acceptance Criteria

该设计的第一版实现完成后，应满足：

1. 能在 OpenRCA task_3 cases 上生成 VLM 主流程、text-only baseline 和 deterministic text-summary baseline 的结果。
2. 主流程中 VLM 输出只含结构化视觉事实，不含根因、置信度或因果判断。
3. Pipeline 运行对象与评估标签对象隔离，非 Evaluator 模块无法读取 ground truth。
4. 每个 case 的图像预算默认不超过 4 张，panel 和 observation 可追溯。
5. sidecar analysis 级别足以复现候选、指标、图像、VLM facts、Judge predictions 和评估结果。
6. 报告包含 Correct、Partial、top@1/3/5 reason、diagnostic_top@1/3/5，并区分 oracle 与 heuristic setting。
7. 契约测试和关键模块测试覆盖标签泄漏、sanitized instruction、schema 禁用字段、图像预算、trace/log 过滤、评估 top-k golden cases。
8. Reason label taxonomy、alias mapping 和 heuristic candidate seed algorithm 均版本化，并能通过 fixture/golden tests 复现。
