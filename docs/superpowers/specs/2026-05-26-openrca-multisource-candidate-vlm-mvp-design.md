# OpenRCA Multi-Source Candidate Recall and VLM Evidence MVP Design

Date: 2026-05-26
Status: Draft for user review
Scope: Independent Phase 0 MVP for OpenRCA candidate retrieval, rendering-budget audit, and VLM evidence sanity check

## 1. Relation to the Existing Task 3 Design

This document defines a new independent MVP. It does not replace
`docs/superpowers/specs/2026-05-25-task3-vlm-rca-design.md`.

The existing Task 3 design remains the later reason-localization design. This Phase 0 MVP focuses on the pre-ranking stage of VLM4RCA: whether the system can retrieve and visualize plausible root-cause-related candidates under a fixed budget before any causal ranking is performed.

## 2. Terminology

- `OpenRCA benchmark`: the original benchmark described in the OpenRCA paper and task materials. It contains multi-source telemetry such as metrics, logs, and traces.
- `OpenRCA-Bank subset`: the local fixed case bank used for Phase 0 validation. The first implementation should design adapter interfaces so that other OpenRCA systems can be added later, but Phase 0 acceptance only requires this local subset.
- `Phase 0`: the recall-first MVP defined in this document.
- `Full variant`: the `Metric + Trace + Log + Topology` candidate-source variant.
- `Pre-ranking`: all steps before LLM causal ranking, including candidate retrieval, candidate merging, budgeted rendering, and optional VLM evidence extraction.

## 3. Goal

Phase 0 tests one focused research question:

> Under a fixed candidate budget and bounded rendering budget, does multi-source candidate retrieval bring true root-cause-related components/services into the downstream VLM/LLM stage more reliably than a metric-only statistical gate?

The MVP verifies three hypotheses:

1. `H1 candidate recall`: compare `Metric-only`, `Metric + Trace`, `Metric + Trace + Log`, and `Metric + Trace + Log + Topology` on `Recall@K`. The focus is whether multi-source retrieval improves the chance that the true root-cause-related component/service enters the candidate pool, especially in soft latency and dependency-related cases. The result is not assumed to be monotonic at every K.
2. `H2 budget control`: show that multi-source retrieval does not imply drawing all telemetry. All candidate-source variants share the same candidate budget, so `Recall@8` is comparable across variants. Rendering-budget auditing is performed on the Full variant only. If other variants are rendered for qualitative comparison, they must use the same cap and be reported separately.
3. `H3-lite VLM evidence sanity`: on a small selected sample, verify that diagnostic panels can produce non-redundant candidate-level visual evidence. In Phase 0, the VLM is a diagnostic panel reader and visual evidence extractor, not a root-cause judge.

Phase 0 does not claim end-to-end RCA success.

## 4. Non-Goals

- Do not report final Top-1 or Top-3 RCA accuracy as a Phase 0 success criterion.
- Do not implement or evaluate a formal LLM causal ranker.
- Do not ask the VLM to identify root cause, culprit, final diagnosis, or candidate ranking.
- Do not run the full OpenRCA benchmark in the first MVP.
- Do not make fault-type coverage a primary metric. It is deferred to Phase 1.
- Do not replace the existing Task 3 reason-localization design.

## 5. Experiment Scope

### 5.1 Case Scope

Phase 0 uses two stages:

| Stage | Total Cases | Metric-Obvious | Soft Latency / Network / Dependency | Mixed / Ambiguous |
| --- | ---: | ---: | ---: | ---: |
| Pilot | 15 | 5 | 5 | 5 |
| Full MVP | 30 | 10 | 10 | 10 |

Case selection criteria:

- Clear ground-truth component/service or a mappable resource/pod/container target.
- `inject_time` or another reliable fault-time anchor.
- Metrics are available. Trace/log absence is allowed but must be recorded as modality availability.
- The soft subset includes cases such as latency degradation, network delay, dependency slow response, trace-heavy failures, or log-heavy failures.
- A fixed case-selection manifest is saved so the sample does not drift across runs.

Ground truth is used only for case grouping and evaluation. It must not enter candidate scoring, candidate selection, visualization planning, source-specific retrieval, VLM input, or any runtime prompt.

### 5.2 Modality Availability

Every case records:

```yaml
modality_availability:
  metrics_available: true
  traces_available: true
  logs_available: false
  topology_available: true
  topology_source: trace_derived
```

By default, topology is trace-derived. Therefore, `topology_available=true` requires `traces_available=true` unless a static topology file is provided by the adapter. Valid `topology_source` values are `trace_derived`, `static_topology`, and `unavailable`.

Reports separate two views:

- `Primary analysis`: evaluate the incremental value of a modality only on cases where that modality is available.
- `Secondary analysis`: report overall recall across all cases, with missing modalities marked as unavailable rather than treated as method failures.

## 6. Experiment Variants

The main experiment uses incremental ablation, not a full source-combination grid:

| Variant | Candidate Sources | Purpose |
| --- | --- | --- |
| `M` | Metric-only | Baseline statistical gate |
| `M+T` | Metric + Trace | Measure trace contribution for latency and dependency cases |
| `M+T+L` | Metric + Trace + Log | Measure log contribution for timeout, retry, and error cases |
| `M+T+L+Topo` | Metric + Trace + Log + Topology | Measure one-hop topology expansion contribution |

The main report includes `Recall@3`, `Recall@5`, and `Recall@8`. `Recall@8` is the most important K because it matches the final candidate budget and the Full-variant rendering budget.

For every case, the report records:

```yaml
case_id: case_001
case_group: soft_latency
ground_truth_component: gateway
metric_hit_at_8: false
metric_trace_hit_at_8: true
metric_trace_log_hit_at_8: true
full_hit_at_8: true
first_hit_source: Trace
new_hit_source: Trace
```

Definitions:

- `first_hit_source`: the first source in the incremental ablation chain that retrieves the ground-truth component/service. Values: `Metric`, `Trace`, `Log`, `Topology`, `NotHit`.
- `new_hit_source`: relative to the metric-only baseline, the first added source that recovers a ground truth that metric-only missed. Values: `Trace`, `Log`, `Topology`, `None`, `NotHit`.

If metric-only already hits, `first_hit_source=Metric` and `new_hit_source=None`.

## 7. Incident Windows

All sources compare the same baseline and incident windows anchored at `inject_time`:

```yaml
incident_window:
  pre_window: 10min
  post_window: 20min
  baseline_window: 30min before pre_window
```

This corresponds to:

```text
baseline window: T0 - 40min to T0 - 10min
incident window: T0 - 10min to T0 + 20min
```

The implementation must record boundary rules, timezone handling, sampling interval assumptions, and missing-data warnings in sidecar metadata.

## 8. Candidate Data Contract

The canonical candidate object is:

```yaml
candidate_key: service:checkout-service
variant_candidate_id: cand:{case_id}:{variant}:{rank}:{target_type}:{canonical_target}
target_type: service | edge | resource
canonical_target: checkout-service
raw_target: checkout_service
introduced_by: trace
metric_only_present: false
present_in_variants:
  - M+T
  - M+T+L
  - M+T+L+Topo
sources:
  - trace
source_scores:
  trace: 4.2
evidence_summary:
  - "span duration p95 increased"
rank: 1
selected_for_rendering: true
```

`candidate_key` is stable across variants and is used for cross-variant merge, `present_in_variants`, `first_hit_source`, and `new_hit_source` analysis. `variant_candidate_id` is variant-scoped and rank-scoped; it is used for reports, rendering manifests, and VLM evidence references.

`introduced_by` means the earliest source in the incremental ablation chain that first introduces this canonical candidate. `sources` lists all evidence sources that support the candidate after merging.

`evidence_summary` is used for cheap candidate analysis and non-VLM reports. It must not be embedded inside image manifests given to the VLM.

## 9. Candidate Sources

### 9.1 Metric Candidates

Metric retrieval groups metrics by canonical component/service and computes cheap anomaly features:

- robust z-score
- relative change
- p95 shift
- optional sustained-change ratio

Metrics are categorized into latency, error, traffic, CPU, memory, disk, network, and unknown. Each component keeps top evidence by category. The metric source contributes service/resource candidates.

### 9.2 Trace Candidates

Trace retrieval computes two cheap features:

- service span duration shift
- edge latency shift for `caller -> callee`

Trace edges are saved as shadow edge candidates. An edge may also introduce its caller and callee as projected service candidates. The projected services must compete under the same top-8 final candidate budget. The edge itself does not directly count as a component-level hit in the main metric.

### 9.3 Log Candidates

Log retrieval uses keyword and simple event-category deltas. No log embedding is required in Phase 0.

Default keywords:

```text
timeout
retry
failed
error
exception
unavailable
deadline exceeded
connection refused
connection reset
slow
latency
backoff
circuit breaker
```

For each service, compute:

```text
keyword_rate_delta = incident_keyword_count_per_min - baseline_keyword_count_per_min
```

If the effective baseline and incident durations are identical and complete, this is equivalent to a count delta scaled by a constant. If effective window duration differs due to missing data, rate-normalized deltas are required.

### 9.4 Topology Expansion

By default, topology is built from trace-derived caller/callee edges. If an adapter provides a static topology file, the same expansion rules apply and `topology_source=static_topology`. It expands existing service candidates by one hop:

- upstream neighbors
- downstream neighbors

Default constraints:

```yaml
topology_expansion:
  expand_hops: 1
  max_neighbors_per_candidate: 2
  max_total_topology_candidates: 5
```

Topology candidates must record why they were introduced, such as `upstream_of_trace_hit`, `downstream_of_metric_hit`, or `neighbor_of_log_hit`.

## 10. Candidate Merging and Ranking

Default soft quotas:

```yaml
source_quota:
  metric: 3
  trace_service: 2
  trace_edge_projected_service: 2
  log: 2
  topology: 2
max_final_candidates: 8
```

Quotas are soft quotas. They preserve source diversity before global top-8 truncation. If a source has fewer candidates than its quota, the unused protected quota is not reserved; remaining slots are filled from all enabled sources by the same global ranking rule.

Phase 0 uses one fixed deterministic ranking algorithm for all case groups:

1. Build ranked candidate lists for each enabled source.
2. Normalize scores within each source.
3. Mark candidates as protected according to the enabled source soft quotas.
4. Merge candidates with the same canonical service. A merged candidate is protected if any source-specific candidate merged into it was protected.
5. Merge duplicate edge candidates separately into the shadow edge pool.
6. Aggregate merged source scores into one normalized candidate score.
7. Partition merged service/resource candidates into protected and non-protected groups.
8. Sort protected candidates and non-protected candidates separately by:
   - number of supporting sources, descending
   - aggregated normalized score, descending
   - fixed source priority, `metric > trace > log > topology`
   - canonical target name, ascending
9. Construct the final list by taking sorted protected candidates first, then filling remaining slots with sorted non-protected candidates.
10. Truncate to `max_final_candidates=8`. If protected candidates alone exceed eight after merging, keep the top eight protected candidates by the same deterministic sort and omit non-protected candidates.

The fixed priority avoids group-specific tuning. Source priority is a tie-break, not the primary score. In a single-source variant such as `M`, the enabled source may fill the full top-8 budget.

## 11. Ground Truth Mapping and Matching

The primary metric is service/component-level recall:

```text
component_hit_at_k = ground-truth canonical component appears in top-k service/resource candidates
```

Ground truth mapping records:

```yaml
gt_mapping:
  raw_ground_truth: pod/payment-v2-xxx
  mapped_component: payment-service
  mapping_type: pod_to_service
  mapping_confidence: exact | heuristic | unknown
```

Canonicalization supports case folding, underscore/hyphen normalization, and version/pod suffix cleanup where justified. Ambiguous mappings are marked as `mapping_confidence=unknown` and reviewed separately.

If a case has multiple ground-truth components, the main table uses any-hit, while sidecar records all-hit details:

```yaml
hit_detail:
  any_hit: true
  all_hit: false
  hit_targets:
    - gateway
  missed_targets:
    - user-service
```

Shadow edge recall is reported separately:

```text
edge_hit_at_k = ground-truth or manually annotated relevant dependency edge appears in top-k edge shadow candidates
```

Edge recall is diagnostic only and is not a Phase 0 success criterion.

## 12. Rendering Budget

Default Phase 0 budget:

```yaml
rendering_budget:
  max_rendered_candidates: 8
  max_total_images: 8
  max_curves_per_image: 6
  max_vlm_calls_per_case: 2
  max_images_per_vlm_call: 4
```

Upper-bound configuration for later review or Phase 1 sensitivity analysis:

```yaml
rendering_budget_upper_bound:
  max_rendered_candidates: 10
  max_total_images: 12
  max_curves_per_image: 6
  max_vlm_calls_per_case: 3
```

Phase 0 main experiments use only the default budget. Under this budget, each rendered candidate gets at most one primary image. Secondary images are not used in the main experiment and are allowed only in upper-bound sensitivity analysis.

All variants share the same candidate budget, so `Recall@8` is comparable across variants. Rendering-budget auditing is defined only on the Full variant. Metric-only or other variant charts may be generated during the pilot for qualitative comparison, but they must use the same rendering cap, be reported separately, and not be mixed into the main rendering-efficiency statistics.

## 13. Diagnostic Visualization Templates

### 13.1 Service Diagnostic Timeline

Used for service/resource candidates. The image contains at most six curves:

- service latency p95
- error rate
- request count or throughput
- CPU or memory if available
- log keyword count
- related span duration

### 13.2 Edge Latency Decomposition Panel

Used for trace edge shadow candidates or edge-projected service candidates. The image contains at most six curves:

- caller span duration
- callee span duration
- edge latency
- caller service latency
- callee service latency
- request count

### 13.3 Local Topology Panel

Used for topology-introduced candidates or selected soft-case audits. This is not a curve chart, so it has separate complexity limits:

```yaml
topology_panel_constraints:
  max_nodes: 5
  max_edges: 6
  max_upstream_neighbors: 2
  max_downstream_neighbors: 2
```

The panel shows the candidate service, up to two upstream neighbors, and up to two downstream neighbors. Node color represents metric/trace/log normalized scores. Edge width represents edge latency shift.

### 13.4 Shared Image Requirements

Every image includes:

- `case_id`
- `candidate_key`
- `variant_candidate_id`
- canonical target
- image type
- inject-time vertical line where applicable
- baseline window shading where applicable
- incident window shading where applicable
- legend for curves or graph encodings
- manifest traceability through `source_refs`

`source_refs` are traceability handles only:

```yaml
source_refs:
  - type: metric
    id: metric:gateway:latency_p95
  - type: trace_edge
    id: trace_edge:gateway->user-service
  - type: log_event
    id: log_event:gateway:timeout_count
```

They must not include ground-truth labels, candidate recall status, `first_hit_source`, candidate score explanations, or explanatory evidence summaries.

A panel may contain fewer than six curves when modalities are unavailable. The system must not backfill missing slots with unrelated or low-value signals solely to reach the curve limit.

### 13.5 Template Selection

For each final candidate, choose exactly one primary image:

1. If the candidate is introduced by topology and a valid local subgraph is available, use `local_topology_panel`.
2. Else if the candidate is projected from a trace edge and edge-level telemetry is available, use `edge_latency_decomposition_panel`.
3. Else use `service_diagnostic_timeline`.
4. If the chosen panel is not renderable, mark `render_status=no_renderable_signal` and do not substitute out-of-pool objects.

If the image count exceeds eight, truncate by final candidate rank.

## 14. VLM Evidence Sanity Check

The VLM sanity check runs on 10-15 selected cases, preferably:

- metric-only miss but Full variant hit
- soft latency or dependency-related cases
- cases where Trace, Log, or Topology is the `first_hit_source`

The VLM sanity subset is an enriched diagnostic subset, not an unbiased estimate of VLM performance over the full MVP set. Optionally include 3-5 randomly sampled pilot cases as a sanity-control subset and report them separately.

The VLM reads images and non-explanatory manifests only.

VLM input:

```yaml
case_id: case_001
time_window_spec:
  t0: inject_time
  baseline_window: T0-40min to T0-10min
  incident_window: T0-10min to T0+20min
images:
  - image_id: img_001
    candidate_key: service:gateway
    variant_candidate_id: cand:case_001:full:1:service:gateway
    canonical_target: gateway
    target_type: service
    image_type: service_diagnostic_timeline
    source_refs:
      - type: metric
        id: metric:gateway:latency_p95
allowed_variant_candidate_ids:
  - cand:case_001:full:1:service:gateway
allowed_image_ids:
  - img_001
```

The VLM does not receive:

- ground truth component/service
- case group
- first-hit source
- candidate recall result
- official reason label
- explanatory candidate evidence summary
- root-cause, culprit, injected-fault, or final-diagnosis fields

### 14.1 VLM Output Schema

```yaml
candidate_visual_evidence:
  - evidence_id: ve:case_001:img_001:0
    candidate_key: service:gateway
    variant_candidate_id: cand:case_001:full:1:service:gateway
    image_id: img_001
    image_type: service_diagnostic_timeline
    visual_facts:
      - fact_type: temporal_shift
        observation: "Latency rises after T0."
        support_level: strong
        time_relation_to_t0: after_t0
    supports_candidate_pattern: true
    contradicts_candidate_pattern: uncertain
    missing_evidence:
      - "No downstream trace panel is shown."
    readability_status: readable
```

Allowed `fact_type` values:

- `temporal_shift`
- `propagation_pattern`
- `parent_child_mismatch`
- `log_metric_alignment`
- `weak_signal`
- `evidence_gap`
- `readability_issue`

Allowed `time_relation_to_t0` values:

- `before_t0`
- `around_t0`
- `after_t0`
- `spans_t0`
- `unclear`

`T0` is the case `inject_time` or equivalent fault-time anchor.

Allowed `readability_status` values:

- `readable`
- `partially_readable`
- `unreadable`

`supports_candidate_pattern=true` means the image supports the visual pattern associated with the candidate, such as latency degradation, mismatch, alignment, or propagation. It does not mean the candidate is the root cause.

Forbidden fields and semantics:

- `root_cause`
- `culprit`
- `is_the_cause`
- `final diagnosis`
- cross-candidate final ranking
- confidence as RCA probability

## 15. Evaluation Metrics

### 15.1 Candidate Recall

```yaml
candidate_recall:
  component_recall_at_3
  component_recall_at_5
  component_recall_at_8
  soft_subset_component_recall_at_8
  first_hit_source_distribution
  new_hit_source_distribution
```

### 15.2 Shadow Edge Metrics

```yaml
shadow_edge:
  edge_recall_at_3
  edge_recall_at_5
  edge_recall_at_8
```

These are diagnostic metrics only.

### 15.3 Rendering Efficiency

```yaml
render_efficiency:
  avg_rendered_candidates_per_case
  avg_images_per_case
  max_images_per_case
  avg_curves_per_image
  max_curves_per_image
  avg_vlm_calls_per_case
  max_topology_nodes_per_panel
  max_topology_edges_per_panel
  no_renderable_signal_rate
```

### 15.4 VLM Sanity Metrics

```yaml
vlm_sanity:
  schema_valid_rate
  forbidden_field_rate
  root_cause_leakage_rate
  evidence_extraction_success_rate
  readable_image_rate
  candidate_evidence_coverage
  non_redundant_visual_fact_rate
  temporal_shift_coverage
  propagation_pattern_coverage
  parent_child_mismatch_coverage
  log_metric_alignment_coverage
  evidence_gap_coverage
```

A VLM fact is non-redundant if it describes a visual relation not explicitly present in the cheap candidate evidence text, such as temporal lag, cross-signal alignment, propagation direction, parent-child mismatch, or readability limitations.

## 16. Required Report Tables

### 16.1 Candidate Recall Incremental Ablation

```markdown
| Variant | Recall@3 | Recall@5 | Recall@8 | Soft Recall@8 | Avg Candidates |
|---|---:|---:|---:|---:|---:|
| M | `<computed>` | `<computed>` | `<computed>` | `<computed>` | `<computed>` |
| M+T | `<computed>` | `<computed>` | `<computed>` | `<computed>` | `<computed>` |
| M+T+L | `<computed>` | `<computed>` | `<computed>` | `<computed>` | `<computed>` |
| M+T+L+Topo | `<computed>` | `<computed>` | `<computed>` | `<computed>` | `<computed>` |
```

### 16.2 First Hit Source

```markdown
| Case | Group | GT Component | M@8 | M+T@8 | M+T+L@8 | Full@8 | First Hit Source | New Hit Source |
|---|---|---|---:|---:|---:|---:|---|---|
```

### 16.3 Rendering Efficiency

```markdown
| Metric | Value |
|---|---:|
| Avg Images / Case | `<computed>` |
| Max Images / Case | `<computed>` |
| Avg Curves / Image | `<computed>` |
| Max Curves / Image | `<computed>` |
| Avg VLM Calls / Case | `<computed>` |
| Max Topology Nodes / Panel | `<computed>` |
| Max Topology Edges / Panel | `<computed>` |
| No Renderable Signal Rate | `<computed>` |
```

### 16.4 VLM Sanity Check

```markdown
| Evidence Type | Coverage | Non-redundant Rate | Common Failure |
|---|---:|---:|---|
| Temporal shift | `<computed>` | `<computed>` | `<observed failure mode>` |
| Propagation pattern | `<computed>` | `<computed>` | `<observed failure mode>` |
| Parent-child mismatch | `<computed>` | `<computed>` | `<observed failure mode>` |
| Log-metric alignment | `<computed>` | `<computed>` | `<observed failure mode>` |
| Evidence gap | `<computed>` | `<computed>` | `<observed failure mode>` |
```

## 17. Delivery, Budget, and Research Criteria

Delivery and research support are judged separately. A complete run can be an engineering success even if the research hypothesis is not supported.

### 17.1 Minimum Delivery

- Candidate recall ablation runs on the 15-case pilot.
- Full variant rendering-budget audit runs with at most eight images per case.
- Report includes `first_hit_source` and `new_hit_source`.
- VLM sanity check can produce schema-valid outputs on selected pilot cases.

### 17.2 Full MVP Delivery

- Candidate recall ablation runs on the 30-case MVP set.
- All main tables are generated.
- Modality availability and missing-source cases are reported.
- Rendering budget audit reports image, curve, VLM-call, and topology-panel complexity metrics.
- VLM sanity check runs on 10-15 selected cases.
- VLM output contains no root-cause decision fields.

### 17.3 Budget Success Criteria

- Average images per case is at most 8.
- Maximum images per case is at most 8.
- Average curves per image is at most 6.
- Maximum VLM calls per case is at most 2.
- Topology panels stay within `max_nodes=5` and `max_edges=6`.

### 17.4 VLM Boundary Success Criteria

- VLM sanity check produces schema-valid outputs for selected cases after retry handling.
- `forbidden_field_rate` is zero after parser validation and retry handling.
- `root_cause_leakage_rate` is zero after parser validation and retry handling.

### 17.5 Research-Positive Criteria

- `Full Recall@8 - Metric Recall@8 >= 10 percentage points`.
- `Soft subset Full Recall@8 - Metric Recall@8 >= 15 percentage points`.
- VLM sanity check shows non-redundant visual evidence in selected cases.

## 18. Failure Diagnosis

If candidate recall does not improve:

- inspect trace parser coverage
- inspect service canonicalization
- inspect log keyword coverage
- inspect topology expansion budget
- inspect modality availability

If recall improves but rendering fails:

- inspect source references
- inspect panel template coverage
- inspect no-renderable-signal cases
- inspect missing modality patterns

If images render but VLM facts are redundant:

- compare VLM facts against cheap candidate evidence text
- inspect whether templates expose temporal lag, propagation, or mismatch
- reduce overplotted curves or topology density

If VLM outputs root-cause decisions:

- tighten schema
- remove explanatory text from manifests
- add parser rejection for forbidden fields and root-cause language
- retry with a stricter prompt

## 19. Sidecar Requirements

Each run writes analysis-level sidecar data:

- run metadata: commit, config hash, dataset subset id, model ids, prompt/schema versions
- case-selection manifest id
- modality availability
- ground-truth mapping and mapping confidence
- candidate pools for all variants with stable `candidate_key` and variant-scoped `variant_candidate_id`
- final top-8 candidates
- `first_hit_source` and `new_hit_source`
- shadow edge candidates and metrics
- render manifests with `source_refs`
- render status and complexity metrics
- VLM sanity outputs, parse status, forbidden-field audit
- report table inputs

Debug sidecars may include raw prompts/responses, but ground truth remains excluded from runtime prompt inputs.

## 20. Reference

- OpenRCA OpenReview page: https://openreview.net/forum?id=M4qNIzQYpd
