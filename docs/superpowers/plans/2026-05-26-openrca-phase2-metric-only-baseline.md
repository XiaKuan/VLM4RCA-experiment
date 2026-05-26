# OpenRCA Phase 2 Metric-Only Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first comparable OpenRCA baseline by producing deterministic Metric-only (`M`) top-8 candidates for the fixed 15-case pilot and reporting component Recall@3/5/8.

**Architecture:** Extend the Phase 1 OpenRCA data/evaluation skeleton with a small candidate package dedicated to candidate contracts, metric feature extraction, metric scoring, and Metric-only candidate building. Keep the Phase 2 pipeline single-source by construction: it reads Phase 1 sidecars, reads only `metrics.csv`, writes stable candidate JSON, evaluates only the `M` variant, and produces the first Markdown/JSON baseline report.

**Tech Stack:** Python 3.12, Pydantic v2, pandas, numpy, PyYAML, pytest, uv, pathlib/json standard libraries.

---

## Scope Boundary

Phase 2 is accepted when it can:

- Run on the fixed pilot 15 cases from `configs/openrca_pilot15.yaml`.
- Build Metric-only candidate lists from `metrics.csv` only.
- Score metric features with robust z-score, relative change, and p95 shift.
- Emit at most 8 `M` candidates per case.
- Give every candidate a stable `candidate_key` and deterministic `variant_candidate_id`.
- Compute component Recall@3, Recall@5, and Recall@8 for the `M` variant.
- Write the first baseline report table with one `M` row.

Phase 2 must not:

- Read traces or logs for candidate generation.
- Build topology candidates.
- Render images.
- Call a VLM, LLM, OpenAI-compatible endpoint, or model client.
- Use ground truth during metric feature scoring, candidate selection, or candidate ranking.

## Prerequisites

This plan builds on the Phase 1 plan at `docs/superpowers/plans/2026-05-26-openrca-phase1-data-eval-skeleton.md`.

Before starting implementation, the worktree must contain the Phase 1 files listed below:

- `configs/openrca_pilot15.yaml`
- `src/vlm4rca/openrca/adapter.py`
- `src/vlm4rca/openrca/canonicalization.py`
- `src/vlm4rca/openrca/manifest.py`
- `src/vlm4rca/openrca/models.py`
- `src/vlm4rca/openrca/phase1.py`
- `src/vlm4rca/evaluation/recall.py`

## File Structure

Create these files:

- `src/vlm4rca/candidates/__init__.py` - exports candidate contracts and Metric-only builder entry points.
- `src/vlm4rca/candidates/models.py` - Pydantic candidate data contract, metric evidence contract, and build-result contract.
- `src/vlm4rca/candidates/metric_features.py` - metric column parsing, metric category heuristics, window slicing, robust z-score, relative change, p95 shift, and per-metric scoring.
- `src/vlm4rca/candidates/metric_builder.py` - groups scored metric evidence by canonical component, ranks components, and emits top-8 `M` candidates.
- `src/vlm4rca/openrca/phase2_metric.py` - Phase 2 CLI and orchestration for the fixed pilot or any compatible manifest.
- `tests/candidates/test_candidate_models.py`
- `tests/candidates/test_metric_features.py`
- `tests/candidates/test_metric_builder.py`
- `tests/evaluation/test_metric_recall.py`
- `tests/openrca/test_phase2_metric.py`

Modify these files:

- `src/vlm4rca/evaluation/recall.py` - add candidate-aware helpers while preserving the Phase 1 empty-run behavior.
- `src/vlm4rca/evaluation/__init__.py` - export the new helper functions.

Runtime outputs go under `outputs/openrca_phase2_metric_pilot15/` and are not committed.

---

### Task 0: Execution Worktree And Phase 1 Prerequisite Check

**Files:**
- No repository files changed in this task.

- [ ] **Step 1: Create an isolated worktree**

Run:

```bash
mkdir -p .claude/worktrees/feat
git worktree add .claude/worktrees/feat/openrca-phase2-metric-baseline -b feat/openrca-phase2-metric-baseline main
cd .claude/worktrees/feat/openrca-phase2-metric-baseline
```

Expected: command succeeds and `pwd` ends with `.claude/worktrees/feat/openrca-phase2-metric-baseline`.

- [ ] **Step 2: Install dependencies**

Run:

```bash
uv sync
```

Expected: dependencies are installed without errors.

- [ ] **Step 3: Verify Phase 1 files exist**

Run:

```bash
test -f configs/openrca_pilot15.yaml
test -f src/vlm4rca/openrca/adapter.py
test -f src/vlm4rca/openrca/canonicalization.py
test -f src/vlm4rca/openrca/manifest.py
test -f src/vlm4rca/openrca/models.py
test -f src/vlm4rca/openrca/phase1.py
test -f src/vlm4rca/evaluation/recall.py
```

Expected: all commands exit with status 0.

- [ ] **Step 4: Verify the Phase 1 test suite**

Run:

```bash
uv run pytest tests/openrca tests/evaluation -v
```

Expected: Phase 1 tests pass. Record any pre-existing failure before editing files.

---

### Task 1: Candidate Data Contracts

**Files:**
- Create: `src/vlm4rca/candidates/__init__.py`
- Create: `src/vlm4rca/candidates/models.py`
- Create: `tests/candidates/test_candidate_models.py`

- [ ] **Step 1: Write the failing candidate contract tests**

Create `tests/candidates/test_candidate_models.py`:

```python
import pytest

from vlm4rca.candidates.models import (
    MetricCandidateBuildResult,
    MetricFeatureEvidence,
    RcaCandidate,
)


def test_metric_candidate_uses_stable_keys_and_variant_scoped_id() -> None:
    candidate = RcaCandidate(
        case_id="case_001",
        variant="M",
        candidate_key="service:tomcat01",
        variant_candidate_id="cand:case_001:M:1:service:tomcat01",
        target_type="service",
        canonical_target="tomcat01",
        raw_target="Tomcat01",
        introduced_by="metric",
        metric_only_present=True,
        present_in_variants=["M"],
        sources=["metric"],
        source_scores={"metric": 8.25},
        evidence_summary=[
            "cpu: Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil robust_z=4.00 relative_change=1.50 p95_shift=1.40"
        ],
        rank=1,
        selected_for_rendering=True,
    )

    assert candidate.candidate_key == "service:tomcat01"
    assert candidate.variant_candidate_id == "cand:case_001:M:1:service:tomcat01"
    assert candidate.present_in_variants == ["M"]
    assert candidate.sources == ["metric"]


def test_metric_candidate_rejects_variant_id_that_does_not_match_rank() -> None:
    with pytest.raises(ValueError, match="variant_candidate_id must be"):
        RcaCandidate(
            case_id="case_001",
            variant="M",
            candidate_key="service:tomcat01",
            variant_candidate_id="cand:case_001:M:2:service:tomcat01",
            target_type="service",
            canonical_target="tomcat01",
            raw_target="Tomcat01",
            introduced_by="metric",
            metric_only_present=True,
            present_in_variants=["M"],
            sources=["metric"],
            source_scores={"metric": 8.25},
            evidence_summary=["cpu evidence"],
            rank=1,
            selected_for_rendering=True,
        )


def test_metric_feature_evidence_serializes_feature_scores() -> None:
    evidence = MetricFeatureEvidence(
        metric_column="Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil",
        raw_target="Tomcat01",
        canonical_target="tomcat01",
        metric_category="cpu",
        robust_z_score=4.0,
        relative_change=1.5,
        p95_shift=1.4,
        score=6.9,
        baseline_points=6,
        incident_points=4,
    )

    payload = evidence.model_dump(mode="json")

    assert payload["metric_category"] == "cpu"
    assert payload["score"] == 6.9
    assert payload["baseline_points"] == 6
    assert payload["incident_points"] == 4


def test_metric_build_result_limits_m_variant_to_eight_candidates() -> None:
    candidates = [
        RcaCandidate(
            case_id="case_001",
            variant="M",
            candidate_key=f"service:svc-{rank}",
            variant_candidate_id=f"cand:case_001:M:{rank}:service:svc-{rank}",
            target_type="service",
            canonical_target=f"svc-{rank}",
            raw_target=f"svc-{rank}",
            introduced_by="metric",
            metric_only_present=True,
            present_in_variants=["M"],
            sources=["metric"],
            source_scores={"metric": float(20 - rank)},
            evidence_summary=["metric evidence"],
            rank=rank,
            selected_for_rendering=True,
        )
        for rank in range(1, 9)
    ]

    result = MetricCandidateBuildResult(case_id="case_001", variant="M", candidates=candidates)

    assert result.variant == "M"
    assert len(result.candidates) == 8


def test_metric_build_result_rejects_more_than_eight_candidates() -> None:
    candidates = [
        RcaCandidate(
            case_id="case_001",
            variant="M",
            candidate_key=f"service:svc-{rank}",
            variant_candidate_id=f"cand:case_001:M:{rank}:service:svc-{rank}",
            target_type="service",
            canonical_target=f"svc-{rank}",
            raw_target=f"svc-{rank}",
            introduced_by="metric",
            metric_only_present=True,
            present_in_variants=["M"],
            sources=["metric"],
            source_scores={"metric": float(20 - rank)},
            evidence_summary=["metric evidence"],
            rank=rank,
            selected_for_rendering=True,
        )
        for rank in range(1, 10)
    ]

    with pytest.raises(ValueError, match="Metric-only candidates are capped at 8"):
        MetricCandidateBuildResult(case_id="case_001", variant="M", candidates=candidates)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/candidates/test_candidate_models.py -v
```

Expected: FAIL because `vlm4rca.candidates.models` does not exist.

- [ ] **Step 3: Implement candidate contracts**

Create `src/vlm4rca/candidates/models.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

VariantName = Literal["M", "M+T", "M+T+L", "M+T+L+Topo"]
CandidateSource = Literal["metric", "trace", "log", "topology"]
TargetType = Literal["service", "edge", "resource"]
MetricCategory = Literal[
    "latency",
    "error",
    "traffic",
    "cpu",
    "memory",
    "disk",
    "network",
    "unknown",
]


class MetricFeatureEvidence(BaseModel):
    metric_column: str
    raw_target: str
    canonical_target: str
    metric_category: MetricCategory
    robust_z_score: float
    relative_change: float
    p95_shift: float
    score: float
    baseline_points: int
    incident_points: int


class RcaCandidate(BaseModel):
    case_id: str
    variant: VariantName
    candidate_key: str
    variant_candidate_id: str
    target_type: TargetType
    canonical_target: str
    raw_target: str
    introduced_by: CandidateSource
    metric_only_present: bool
    present_in_variants: list[VariantName]
    sources: list[CandidateSource]
    source_scores: dict[CandidateSource, float]
    evidence_summary: list[str] = Field(default_factory=list)
    rank: int = Field(ge=1)
    selected_for_rendering: bool

    @field_validator("candidate_key")
    @classmethod
    def candidate_key_has_target_type_prefix(cls, value: str) -> str:
        if not value.startswith(("service:", "edge:", "resource:")):
            raise ValueError("candidate_key must start with service:, edge:, or resource:")
        return value

    @model_validator(mode="after")
    def ids_match_candidate_identity(self) -> RcaCandidate:
        expected_key = f"{self.target_type}:{self.canonical_target}"
        if self.candidate_key != expected_key:
            raise ValueError(f"candidate_key must be {expected_key}")

        expected_variant_id = (
            f"cand:{self.case_id}:{self.variant}:{self.rank}:{self.target_type}:{self.canonical_target}"
        )
        if self.variant_candidate_id != expected_variant_id:
            raise ValueError(f"variant_candidate_id must be {expected_variant_id}")

        if self.variant == "M" and self.present_in_variants != ["M"]:
            raise ValueError("Metric-only candidates must have present_in_variants=['M']")
        if self.variant == "M" and self.sources != ["metric"]:
            raise ValueError("Metric-only candidates must have sources=['metric']")
        return self


class MetricCandidateBuildResult(BaseModel):
    case_id: str
    variant: Literal["M"]
    candidates: list[RcaCandidate]
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_metric_only_budget(self) -> MetricCandidateBuildResult:
        if len(self.candidates) > 8:
            raise ValueError("Metric-only candidates are capped at 8")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(self.candidates) + 1)):
            raise ValueError("Metric-only candidate ranks must be consecutive from 1")
        for candidate in self.candidates:
            if candidate.case_id != self.case_id:
                raise ValueError("All candidates must use the build result case_id")
            if candidate.variant != "M":
                raise ValueError("MetricCandidateBuildResult only accepts variant M")
        return self
```

Create `src/vlm4rca/candidates/__init__.py`:

```python
from vlm4rca.candidates.models import (
    CandidateSource,
    MetricCandidateBuildResult,
    MetricCategory,
    MetricFeatureEvidence,
    RcaCandidate,
    TargetType,
    VariantName,
)

__all__ = [
    "CandidateSource",
    "MetricCandidateBuildResult",
    "MetricCategory",
    "MetricFeatureEvidence",
    "RcaCandidate",
    "TargetType",
    "VariantName",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/candidates/test_candidate_models.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/candidates/__init__.py src/vlm4rca/candidates/models.py tests/candidates/test_candidate_models.py
git commit -m "feat: add RCA candidate data contracts"
```

Expected: commit succeeds.

---

### Task 2: Metric Feature Parsing And Scoring

**Files:**
- Create: `src/vlm4rca/candidates/metric_features.py`
- Create: `tests/candidates/test_metric_features.py`

- [ ] **Step 1: Write the failing metric feature tests**

Create `tests/candidates/test_metric_features.py`:

```python
import pandas as pd

from vlm4rca.candidates.metric_features import (
    MetricColumn,
    categorize_metric,
    parse_metric_column,
    score_metric_series,
    score_metrics_dataframe,
)
from vlm4rca.openrca.models import IncidentWindows


def _windows() -> IncidentWindows:
    return IncidentWindows(
        inject_time=1_000,
        baseline_start=600,
        baseline_end=900,
        incident_start=900,
        incident_end=1_200,
        pre_window_seconds=100,
        post_window_seconds=200,
        baseline_window_seconds=300,
    )


def test_parse_metric_column_extracts_component_and_category() -> None:
    parsed = parse_metric_column("Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil")

    assert parsed == MetricColumn(
        metric_column="Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil",
        raw_target="Tomcat01",
        canonical_target="tomcat01",
        metric_name="container__OSLinux-CPU_CPU_CPUCpuUtil",
        metric_category="cpu",
    )


def test_categorize_metric_uses_openrca_metric_name_heuristics() -> None:
    assert categorize_metric("JVM_Memory_HeapMemoryUsage") == "memory"
    assert categorize_metric("OSLinux_LOCALDISK_LOCALDISK-sda_DSKBps") == "disk"
    assert categorize_metric("NETWORK_ens160_NETOutErr") == "error"
    assert categorize_metric("NETWORK_ens160_NETKBTotalPerSec") == "network"
    assert categorize_metric("request_count") == "traffic"
    assert categorize_metric("span_duration_p95") == "latency"
    assert categorize_metric("unrecognized_signal") == "unknown"


def test_score_metric_series_computes_required_features() -> None:
    baseline = pd.Series([10.0, 10.0, 11.0, 9.0, 10.0, 10.0])
    incident = pd.Series([20.0, 21.0, 19.0, 20.0])

    evidence = score_metric_series(
        MetricColumn(
            metric_column="Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil",
            raw_target="Tomcat01",
            canonical_target="tomcat01",
            metric_name="container__OSLinux-CPU_CPU_CPUCpuUtil",
            metric_category="cpu",
        ),
        baseline,
        incident,
    )

    assert evidence.robust_z_score > 6.0
    assert round(evidence.relative_change, 2) == 1.0
    assert round(evidence.p95_shift, 2) == 0.9
    assert evidence.score > evidence.robust_z_score
    assert evidence.baseline_points == 6
    assert evidence.incident_points == 4


def test_score_metric_series_returns_zero_score_for_constant_no_change() -> None:
    baseline = pd.Series([0.0, 0.0, 0.0])
    incident = pd.Series([0.0, 0.0, 0.0])

    evidence = score_metric_series(
        MetricColumn(
            metric_column="Redis01__container__JVM-Memory_HeapMemoryUsage",
            raw_target="Redis01",
            canonical_target="redis01",
            metric_name="container__JVM-Memory_HeapMemoryUsage",
            metric_category="memory",
        ),
        baseline,
        incident,
    )

    assert evidence.robust_z_score == 0.0
    assert evidence.relative_change == 0.0
    assert evidence.p95_shift == 0.0
    assert evidence.score == 0.0


def test_score_metrics_dataframe_filters_to_baseline_and_incident_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1_000, 1_100],
            "Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 20, 21, 20],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [5, 5, 5, 5, 5, 5],
        }
    )

    evidence, warnings = score_metrics_dataframe(frame, _windows())

    assert warnings == []
    assert [item.canonical_target for item in evidence] == ["tomcat01", "redis01"]
    assert evidence[0].metric_category == "cpu"
    assert evidence[0].baseline_points == 3
    assert evidence[0].incident_points == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/candidates/test_metric_features.py -v
```

Expected: FAIL because `vlm4rca.candidates.metric_features` does not exist.

- [ ] **Step 3: Implement metric parsing and feature scoring**

Create `src/vlm4rca/candidates/metric_features.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vlm4rca.candidates.models import MetricCategory, MetricFeatureEvidence
from vlm4rca.openrca.canonicalization import canonicalize_component
from vlm4rca.openrca.models import IncidentWindows

EPSILON = 1e-9
MIN_POINTS_PER_WINDOW = 2


@dataclass(frozen=True)
class MetricColumn:
    metric_column: str
    raw_target: str
    canonical_target: str
    metric_name: str
    metric_category: MetricCategory


def categorize_metric(metric_name: str) -> MetricCategory:
    lowered = metric_name.lower()
    if any(token in lowered for token in ("latency", "duration", "response_time", "responsetime")):
        return "latency"
    if any(token in lowered for token in ("error", "err", "failed", "failure", "exception")):
        return "error"
    if any(token in lowered for token in ("request", "throughput", "qps", "tps", "count")):
        return "traffic"
    if "cpu" in lowered:
        return "cpu"
    if any(token in lowered for token in ("memory", "mem", "heap", "swap")):
        return "memory"
    if any(token in lowered for token in ("disk", "dsk", "filesystem", "fsavailable", "fsused")):
        return "disk"
    if any(token in lowered for token in ("network", "net", "tcp", "packet", "bandwidth")):
        return "network"
    return "unknown"


def parse_metric_column(column_name: str) -> MetricColumn | None:
    if column_name == "timestamp":
        return None

    parts = column_name.split("__")
    if len(parts) >= 2:
        raw_target = parts[0]
        metric_name = "__".join(parts[1:])
    else:
        raw_target = "unknown"
        metric_name = column_name

    canonical_target = canonicalize_component(raw_target)
    if not canonical_target:
        return None

    return MetricColumn(
        metric_column=column_name,
        raw_target=raw_target,
        canonical_target=canonical_target,
        metric_name=metric_name,
        metric_category=categorize_metric(metric_name),
    )


def _finite_values(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _scale_for_baseline(values: np.ndarray) -> float:
    if values.size == 0:
        return 1.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > EPSILON:
        return 1.4826 * mad
    std = float(np.std(values))
    if std > EPSILON:
        return std
    return 1.0


def _relative_delta(incident_value: float, baseline_value: float) -> float:
    denominator = abs(baseline_value)
    if denominator <= EPSILON:
        denominator = 1.0
    return (incident_value - baseline_value) / denominator


def score_metric_series(
    metric: MetricColumn,
    baseline: pd.Series,
    incident: pd.Series,
) -> MetricFeatureEvidence:
    baseline_values = _finite_values(baseline)
    incident_values = _finite_values(incident)

    if baseline_values.size == 0 or incident_values.size == 0:
        robust_z_score = 0.0
        relative_change = 0.0
        p95_shift = 0.0
    else:
        baseline_median = float(np.median(baseline_values))
        incident_median = float(np.median(incident_values))
        robust_z_score = (incident_median - baseline_median) / _scale_for_baseline(baseline_values)

        baseline_mean = float(np.mean(baseline_values))
        incident_mean = float(np.mean(incident_values))
        relative_change = _relative_delta(incident_mean, baseline_mean)

        baseline_p95 = float(np.percentile(baseline_values, 95))
        incident_p95 = float(np.percentile(incident_values, 95))
        p95_shift = _relative_delta(incident_p95, baseline_p95)

    score = abs(robust_z_score) + abs(relative_change) + abs(p95_shift)
    return MetricFeatureEvidence(
        metric_column=metric.metric_column,
        raw_target=metric.raw_target,
        canonical_target=metric.canonical_target,
        metric_category=metric.metric_category,
        robust_z_score=round(float(robust_z_score), 6),
        relative_change=round(float(relative_change), 6),
        p95_shift=round(float(p95_shift), 6),
        score=round(float(score), 6),
        baseline_points=int(baseline_values.size),
        incident_points=int(incident_values.size),
    )


def score_metrics_dataframe(
    metrics: pd.DataFrame,
    windows: IncidentWindows,
) -> tuple[list[MetricFeatureEvidence], list[str]]:
    if "timestamp" not in metrics.columns:
        raise ValueError("metrics.csv must contain a timestamp column")

    frame = metrics.copy()
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
    baseline_mask = (frame["timestamp"] >= windows.baseline_start) & (
        frame["timestamp"] < windows.baseline_end
    )
    incident_mask = (frame["timestamp"] >= windows.incident_start) & (
        frame["timestamp"] < windows.incident_end
    )

    evidence: list[MetricFeatureEvidence] = []
    warnings: list[str] = []
    for column_name in frame.columns:
        parsed = parse_metric_column(column_name)
        if parsed is None:
            continue

        baseline = frame.loc[baseline_mask, column_name]
        incident = frame.loc[incident_mask, column_name]
        scored = score_metric_series(parsed, baseline, incident)
        if (
            scored.baseline_points < MIN_POINTS_PER_WINDOW
            or scored.incident_points < MIN_POINTS_PER_WINDOW
        ):
            warnings.append(
                f"{column_name} skipped: baseline_points={scored.baseline_points} incident_points={scored.incident_points}"
            )
            continue
        evidence.append(scored)

    return evidence, warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/candidates/test_metric_features.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/candidates/metric_features.py tests/candidates/test_metric_features.py
git commit -m "feat: score OpenRCA metric anomaly features"
```

Expected: commit succeeds.

---

### Task 3: Metric Candidate Builder

**Files:**
- Create: `src/vlm4rca/candidates/metric_builder.py`
- Create: `tests/candidates/test_metric_builder.py`
- Modify: `src/vlm4rca/candidates/__init__.py`

- [ ] **Step 1: Write the failing metric builder tests**

Create `tests/candidates/test_metric_builder.py`:

```python
from pathlib import Path

import pandas as pd

from vlm4rca.candidates.metric_builder import (
    build_metric_candidates_from_dataframe,
    build_metric_candidates_for_case,
)
from vlm4rca.openrca.models import IncidentWindows


def _windows() -> IncidentWindows:
    return IncidentWindows(
        inject_time=1_000,
        baseline_start=600,
        baseline_end=900,
        incident_start=900,
        incident_end=1_200,
        pre_window_seconds=100,
        post_window_seconds=200,
        baseline_window_seconds=300,
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1_000, 1_100],
            "Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 40, 41, 42],
            "Tomcat01__container__JVM-Memory_HeapMemoryUsage": [50, 50, 50, 75, 76, 75],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [20, 20, 20, 20, 20, 20],
            "Mysql02__container__OSLinux_LOCALDISK_LOCALDISK-sda_DSKBps": [5, 5, 5, 30, 31, 32],
        }
    )


def test_build_metric_candidates_groups_evidence_by_component() -> None:
    result = build_metric_candidates_from_dataframe("case_001", _frame(), _windows(), max_candidates=8)

    assert result.case_id == "case_001"
    assert result.variant == "M"
    assert [candidate.canonical_target for candidate in result.candidates] == [
        "tomcat01",
        "mysql02",
        "redis01",
    ]
    assert result.candidates[0].candidate_key == "service:tomcat01"
    assert result.candidates[0].variant_candidate_id == "cand:case_001:M:1:service:tomcat01"
    assert result.candidates[0].source_scores["metric"] > result.candidates[1].source_scores["metric"]
    assert result.candidates[0].introduced_by == "metric"
    assert result.candidates[0].metric_only_present is True
    assert result.candidates[0].present_in_variants == ["M"]
    assert result.candidates[0].sources == ["metric"]
    assert result.candidates[0].selected_for_rendering is True
    assert len(result.candidates[0].evidence_summary) == 2


def test_build_metric_candidates_caps_output_to_top_eight() -> None:
    frame = pd.DataFrame({"timestamp": [600, 700, 800, 900, 1_000, 1_100]})
    for index in range(10):
        frame[f"Svc{index:02d}__container__OSLinux-CPU_CPU_CPUCpuUtil"] = [
            10,
            10,
            10,
            20 + index,
            21 + index,
            22 + index,
        ]

    result = build_metric_candidates_from_dataframe("case_001", frame, _windows(), max_candidates=8)

    assert len(result.candidates) == 8
    assert [candidate.rank for candidate in result.candidates] == list(range(1, 9))
    assert all(candidate.variant_candidate_id.startswith("cand:case_001:M:") for candidate in result.candidates)


def test_build_metric_candidates_for_case_reads_metrics_csv(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _frame().to_csv(metrics_path, index=False)

    result = build_metric_candidates_for_case("case_001", metrics_path, _windows())

    assert result.candidates[0].canonical_target == "tomcat01"
    assert result.warnings == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/candidates/test_metric_builder.py -v
```

Expected: FAIL because `vlm4rca.candidates.metric_builder` does not exist.

- [ ] **Step 3: Implement the Metric-only candidate builder**

Create `src/vlm4rca/candidates/metric_builder.py`:

```python
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from vlm4rca.candidates.metric_features import score_metrics_dataframe
from vlm4rca.candidates.models import (
    MetricCandidateBuildResult,
    MetricCategory,
    MetricFeatureEvidence,
    RcaCandidate,
)
from vlm4rca.openrca.models import IncidentWindows


def _evidence_sort_key(evidence: MetricFeatureEvidence) -> tuple[float, str]:
    return (-evidence.score, evidence.metric_column)


def _select_top_evidence_by_category(
    evidence_items: list[MetricFeatureEvidence],
) -> list[MetricFeatureEvidence]:
    best_by_category: dict[MetricCategory, MetricFeatureEvidence] = {}
    for evidence in sorted(evidence_items, key=_evidence_sort_key):
        current = best_by_category.get(evidence.metric_category)
        if current is None or evidence.score > current.score:
            best_by_category[evidence.metric_category] = evidence
    return sorted(best_by_category.values(), key=_evidence_sort_key)[:3]


def _candidate_score(evidence_items: list[MetricFeatureEvidence]) -> float:
    top_evidence = _select_top_evidence_by_category(evidence_items)
    return round(sum(item.score for item in top_evidence), 6)


def _evidence_summary(evidence_items: list[MetricFeatureEvidence]) -> list[str]:
    summaries: list[str] = []
    for evidence in _select_top_evidence_by_category(evidence_items):
        summaries.append(
            f"{evidence.metric_category}: {evidence.metric_column} "
            f"robust_z={evidence.robust_z_score:.2f} "
            f"relative_change={evidence.relative_change:.2f} "
            f"p95_shift={evidence.p95_shift:.2f}"
        )
    return summaries


def build_metric_candidates_from_dataframe(
    case_id: str,
    metrics: pd.DataFrame,
    windows: IncidentWindows,
    max_candidates: int = 8,
) -> MetricCandidateBuildResult:
    scored_metrics, warnings = score_metrics_dataframe(metrics, windows)
    grouped: dict[str, list[MetricFeatureEvidence]] = defaultdict(list)
    raw_targets: dict[str, str] = {}

    for evidence in scored_metrics:
        grouped[evidence.canonical_target].append(evidence)
        raw_targets.setdefault(evidence.canonical_target, evidence.raw_target)

    ranked_components = sorted(
        grouped.items(),
        key=lambda item: (
            -_candidate_score(item[1]),
            -len(item[1]),
            item[0],
        ),
    )

    candidates: list[RcaCandidate] = []
    for rank, (canonical_target, evidence_items) in enumerate(
        ranked_components[:max_candidates],
        start=1,
    ):
        candidate_key = f"service:{canonical_target}"
        candidates.append(
            RcaCandidate(
                case_id=case_id,
                variant="M",
                candidate_key=candidate_key,
                variant_candidate_id=f"cand:{case_id}:M:{rank}:service:{canonical_target}",
                target_type="service",
                canonical_target=canonical_target,
                raw_target=raw_targets[canonical_target],
                introduced_by="metric",
                metric_only_present=True,
                present_in_variants=["M"],
                sources=["metric"],
                source_scores={"metric": _candidate_score(evidence_items)},
                evidence_summary=_evidence_summary(evidence_items),
                rank=rank,
                selected_for_rendering=True,
            )
        )

    return MetricCandidateBuildResult(
        case_id=case_id,
        variant="M",
        candidates=candidates,
        warnings=warnings,
    )


def build_metric_candidates_for_case(
    case_id: str,
    metrics_path: Path,
    windows: IncidentWindows,
    max_candidates: int = 8,
) -> MetricCandidateBuildResult:
    metrics = pd.read_csv(metrics_path)
    return build_metric_candidates_from_dataframe(
        case_id=case_id,
        metrics=metrics,
        windows=windows,
        max_candidates=max_candidates,
    )
```

- [ ] **Step 4: Export the Metric-only builder**

Modify `src/vlm4rca/candidates/__init__.py`:

```python
from vlm4rca.candidates.metric_builder import (
    build_metric_candidates_for_case,
    build_metric_candidates_from_dataframe,
)
from vlm4rca.candidates.models import (
    CandidateSource,
    MetricCandidateBuildResult,
    MetricCategory,
    MetricFeatureEvidence,
    RcaCandidate,
    TargetType,
    VariantName,
)

__all__ = [
    "CandidateSource",
    "MetricCandidateBuildResult",
    "MetricCategory",
    "MetricFeatureEvidence",
    "RcaCandidate",
    "TargetType",
    "VariantName",
    "build_metric_candidates_for_case",
    "build_metric_candidates_from_dataframe",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/candidates/test_metric_builder.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/vlm4rca/candidates/__init__.py src/vlm4rca/candidates/metric_builder.py tests/candidates/test_metric_builder.py
git commit -m "feat: build metric-only top candidates"
```

Expected: commit succeeds.

---

### Task 4: Candidate-Aware Component Recall@K

**Files:**
- Modify: `src/vlm4rca/evaluation/recall.py`
- Modify: `src/vlm4rca/evaluation/__init__.py`
- Create: `tests/evaluation/test_metric_recall.py`

- [ ] **Step 1: Write failing candidate-aware recall tests**

Create `tests/evaluation/test_metric_recall.py`:

```python
from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.evaluation.recall import (
    candidates_by_case_to_targets,
    evaluate_component_recall_at_k,
    summarize_variant_recall,
)
from vlm4rca.openrca.models import (
    GroundTruthMapping,
    IncidentWindows,
    ModalityAvailability,
    Phase1CaseSidecar,
)


def _sidecar(case_id: str, case_group: str, mapped_component: str) -> Phase1CaseSidecar:
    return Phase1CaseSidecar(
        case_id=case_id,
        case_group=case_group,
        task_index="task_3",
        inject_time=1000,
        case_path=f"/tmp/{case_id}",
        raw_ground_truth=[mapped_component],
        incident_window=IncidentWindows(
            inject_time=1000,
            baseline_start=600,
            baseline_end=900,
            incident_start=900,
            incident_end=1200,
            pre_window_seconds=100,
            post_window_seconds=200,
            baseline_window_seconds=300,
        ),
        modality_availability=ModalityAvailability(
            metrics_available=True,
            traces_available=True,
            logs_available=True,
            topology_available=True,
            topology_source="trace_derived",
        ),
        gt_mapping=[
            GroundTruthMapping(
                raw_ground_truth=mapped_component,
                mapped_component=mapped_component,
                mapping_type="component_exact",
                mapping_confidence="exact",
                source="test",
            )
        ],
    )


def _candidate(case_id: str, rank: int, canonical_target: str, score: float) -> RcaCandidate:
    return RcaCandidate(
        case_id=case_id,
        variant="M",
        candidate_key=f"service:{canonical_target}",
        variant_candidate_id=f"cand:{case_id}:M:{rank}:service:{canonical_target}",
        target_type="service",
        canonical_target=canonical_target,
        raw_target=canonical_target,
        introduced_by="metric",
        metric_only_present=True,
        present_in_variants=["M"],
        sources=["metric"],
        source_scores={"metric": score},
        evidence_summary=["metric evidence"],
        rank=rank,
        selected_for_rendering=True,
    )


def test_candidates_by_case_to_targets_orders_by_rank() -> None:
    mapping = candidates_by_case_to_targets(
        {
            "case_1": [
                _candidate("case_1", 2, "redis01", 1.0),
                _candidate("case_1", 1, "tomcat01", 2.0),
            ]
        }
    )

    assert mapping == {"case_1": ["tomcat01", "redis01"]}


def test_evaluate_component_recall_at_k_accepts_candidate_targets() -> None:
    sidecars = [
        _sidecar("case_1", "metric_obvious", "tomcat01"),
        _sidecar("case_2", "soft_latency", "mysql02"),
    ]
    candidates = {
        "case_1": [_candidate("case_1", 1, "tomcat01", 10.0)],
        "case_2": [_candidate("case_2", 1, "redis01", 9.0), _candidate("case_2", 2, "mysql02", 8.0)],
    }

    summary = evaluate_component_recall_at_k(
        sidecars,
        candidates_by_case=candidates_by_case_to_targets(candidates),
        ks=[1, 3, 5, 8],
    )

    assert summary.component_recall_at_k == {"1": 0.5, "3": 1.0, "5": 1.0, "8": 1.0}
    assert summary.case_results[0].hit_at_k["1"] is True
    assert summary.case_results[1].hit_at_k["1"] is False
    assert summary.case_results[1].hit_at_k["3"] is True


def test_summarize_variant_recall_includes_soft_recall_and_average_candidates() -> None:
    sidecars = [
        _sidecar("case_1", "metric_obvious", "tomcat01"),
        _sidecar("case_2", "soft_latency", "mysql02"),
    ]
    candidates = {
        "case_1": [_candidate("case_1", 1, "tomcat01", 10.0)],
        "case_2": [_candidate("case_2", 1, "redis01", 9.0)],
    }

    row = summarize_variant_recall("M", sidecars, candidates, ks=[3, 5, 8])

    assert row == {
        "variant": "M",
        "recall_at_3": 0.5,
        "recall_at_5": 0.5,
        "recall_at_8": 0.5,
        "soft_recall_at_8": 0.0,
        "avg_candidates": 1.0,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/evaluation/test_metric_recall.py -v
```

Expected: FAIL because candidate-aware helper functions do not exist.

- [ ] **Step 3: Add candidate-aware helpers while preserving Phase 1 API**

Modify `src/vlm4rca/evaluation/recall.py` so the complete file is:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from vlm4rca.openrca.canonicalization import canonicalize_component
from vlm4rca.openrca.models import Phase1CaseSidecar, RecallAtKSummary, RecallCaseResult


def evaluate_component_recall_at_k(
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_case: Mapping[str, Sequence[str]],
    ks: Sequence[int] = (3, 5, 8),
) -> RecallAtKSummary:
    ordered_ks = sorted(int(k) for k in ks)
    case_results: list[RecallCaseResult] = []
    hit_counts = {str(k): 0 for k in ordered_ks}

    for sidecar in sidecars:
        targets = [mapping.mapped_component for mapping in sidecar.gt_mapping]
        candidates = [canonicalize_component(candidate) for candidate in candidates_by_case.get(sidecar.case_id, [])]
        hit_at_k: dict[str, bool] = {}
        hit_targets_for_case: set[str] = set()

        for k in ordered_ks:
            top_k = set(candidates[:k])
            hits = [target for target in targets if target in top_k]
            hit = bool(hits)
            hit_at_k[str(k)] = hit
            if hit:
                hit_counts[str(k)] += 1
                hit_targets_for_case.update(hits)

        missed_targets = [target for target in targets if target not in hit_targets_for_case]
        case_results.append(
            RecallCaseResult(
                case_id=sidecar.case_id,
                hit_at_k=hit_at_k,
                hit_targets=sorted(hit_targets_for_case),
                missed_targets=missed_targets,
            )
        )

    denominator = len(sidecars)
    recall = {
        str(k): (hit_counts[str(k)] / denominator if denominator else 0.0)
        for k in ordered_ks
    }
    return RecallAtKSummary(
        n_cases=denominator,
        ks=ordered_ks,
        component_recall_at_k=recall,
        case_results=case_results,
    )


def candidates_by_case_to_targets(
    candidates_by_case: Mapping[str, Sequence[Any]],
) -> dict[str, list[str]]:
    targets_by_case: dict[str, list[str]] = {}
    for case_id, candidates in candidates_by_case.items():
        ordered = sorted(candidates, key=lambda candidate: int(candidate.rank))
        targets_by_case[case_id] = [str(candidate.canonical_target) for candidate in ordered]
    return targets_by_case


def _soft_sidecars(sidecars: Sequence[Phase1CaseSidecar]) -> list[Phase1CaseSidecar]:
    return [sidecar for sidecar in sidecars if sidecar.case_group == "soft_latency"]


def summarize_variant_recall(
    variant: str,
    sidecars: Sequence[Phase1CaseSidecar],
    candidates_by_case: Mapping[str, Sequence[Any]],
    ks: Sequence[int] = (3, 5, 8),
) -> dict[str, float | str]:
    targets_by_case = candidates_by_case_to_targets(candidates_by_case)
    summary = evaluate_component_recall_at_k(sidecars, targets_by_case, ks=ks)
    soft_summary = evaluate_component_recall_at_k(_soft_sidecars(sidecars), targets_by_case, ks=[8])
    denominator = len(sidecars)
    total_candidates = sum(len(candidates_by_case.get(sidecar.case_id, [])) for sidecar in sidecars)
    avg_candidates = total_candidates / denominator if denominator else 0.0

    return {
        "variant": variant,
        "recall_at_3": summary.component_recall_at_k.get("3", 0.0),
        "recall_at_5": summary.component_recall_at_k.get("5", 0.0),
        "recall_at_8": summary.component_recall_at_k.get("8", 0.0),
        "soft_recall_at_8": soft_summary.component_recall_at_k.get("8", 0.0),
        "avg_candidates": round(avg_candidates, 6),
    }
```

- [ ] **Step 4: Export candidate-aware helpers**

Modify `src/vlm4rca/evaluation/__init__.py`:

```python
from vlm4rca.evaluation.recall import (
    candidates_by_case_to_targets,
    evaluate_component_recall_at_k,
    summarize_variant_recall,
)

__all__ = [
    "candidates_by_case_to_targets",
    "evaluate_component_recall_at_k",
    "summarize_variant_recall",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/evaluation/test_recall.py tests/evaluation/test_metric_recall.py -v
```

Expected: Phase 1 recall tests and Phase 2 candidate-aware recall tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add src/vlm4rca/evaluation/__init__.py src/vlm4rca/evaluation/recall.py tests/evaluation/test_metric_recall.py
git commit -m "feat: evaluate metric candidate recall"
```

Expected: commit succeeds.

---

### Task 5: Phase 2 Metric Pipeline And Baseline Report

**Files:**
- Create: `src/vlm4rca/openrca/phase2_metric.py`
- Create: `tests/openrca/test_phase2_metric.py`

- [ ] **Step 1: Write failing Phase 2 pipeline tests**

Create `tests/openrca/test_phase2_metric.py`:

```python
import json
from pathlib import Path

import pandas as pd

from vlm4rca.openrca.phase2_metric import (
    build_metric_baseline_report_markdown,
    main,
    run_metric_baseline,
)


def _write_case(root: Path, case_id: str, component: str, hit_component: str) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1000,
                "context_start": 0,
                "context_end": 2000,
                "matched_faults": [{"component": component, "reason": "high CPU usage", "timestamp": 1000}],
                "system_components": [component, hit_component, "Redis01"],
                "evidence_components": [component, hit_component, "Redis01"],
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1000, 1100],
            f"{hit_component}__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 40, 41, 42],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [20, 20, 20, 20, 20, 20],
        }
    )
    frame.to_csv(case_dir / "metrics.csv", index=False)
    (case_dir / "logs.csv").write_text("time,timestamp,service,log_name,message\n1,1,a,app,ok\n", encoding="utf-8")
    (case_dir / "traces.csv").write_text(
        "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n1,1,a,t,s,p,2\n",
        encoding="utf-8",
    )


def _write_manifest(path: Path, data_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "dataset: openrca_bank",
                f"data_root: {data_root.as_posix()}",
                "cases:",
                "  - case_id: case_1",
                "    case_group: metric_obvious",
                "    selection_reason: fixture one",
                "  - case_id: case_2",
                "    case_group: soft_latency",
                "    selection_reason: fixture two",
            ]
        ),
        encoding="utf-8",
    )


def test_metric_baseline_report_markdown_contains_first_m_row() -> None:
    markdown = build_metric_baseline_report_markdown(
        {
            "variant": "M",
            "recall_at_3": 0.5,
            "recall_at_5": 0.5,
            "recall_at_8": 1.0,
            "soft_recall_at_8": 0.0,
            "avg_candidates": 4.25,
        }
    )

    assert "| Variant | Recall@3 | Recall@5 | Recall@8 | Soft Recall@8 | Avg Candidates |" in markdown
    assert "| M | 0.500 | 0.500 | 1.000 | 0.000 | 4.25 |" in markdown


def test_run_metric_baseline_writes_candidates_recall_and_report(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", component="Tomcat01", hit_component="Tomcat01")
    _write_case(data_root, "case_2", component="Mysql02", hit_component="Redis01")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "outputs"

    result = run_metric_baseline(manifest_path, data_root, output_dir)

    assert result["report_row"]["variant"] == "M"
    assert result["recall_summary"]["component_recall_at_k"]["3"] == 0.5
    assert (output_dir / "candidates" / "M" / "case_1.json").exists()
    assert (output_dir / "candidates" / "M" / "case_2.json").exists()
    assert (output_dir / "recall_M.json").exists()
    assert (output_dir / "baseline_report.md").exists()

    candidate_payload = json.loads((output_dir / "candidates" / "M" / "case_1.json").read_text(encoding="utf-8"))
    assert len(candidate_payload["candidates"]) <= 8
    assert candidate_payload["candidates"][0]["candidate_key"] == "service:tomcat01"
    assert candidate_payload["candidates"][0]["variant_candidate_id"] == "cand:case_1:M:1:service:tomcat01"


def test_phase2_metric_cli_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", component="Tomcat01", hit_component="Tomcat01")
    _write_case(data_root, "case_2", component="Mysql02", hit_component="Redis01")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "phase2"

    exit_code = main(
        [
            "--manifest",
            str(manifest_path),
            "--data-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    summary = json.loads((output_dir / "recall_M.json").read_text(encoding="utf-8"))
    assert summary["component_recall_at_k"]["8"] == 0.5
    assert "Metric-only Baseline Report" in (output_dir / "baseline_report.md").read_text(encoding="utf-8")
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_phase2_metric.py -v
```

Expected: FAIL because `vlm4rca.openrca.phase2_metric` does not exist.

- [ ] **Step 3: Implement the Phase 2 metric pipeline and report writer**

Create `src/vlm4rca/openrca/phase2_metric.py`:

```python
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from vlm4rca.candidates.metric_builder import build_metric_candidates_for_case
from vlm4rca.candidates.models import MetricCandidateBuildResult, RcaCandidate
from vlm4rca.evaluation.recall import (
    candidates_by_case_to_targets,
    evaluate_component_recall_at_k,
    summarize_variant_recall,
)
from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.models import Phase1CaseSidecar
from vlm4rca.openrca.phase1 import build_phase1_sidecars


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _candidate_payload(result: MetricCandidateBuildResult) -> dict[str, Any]:
    return result.model_dump(mode="json")


def build_metric_baseline_report_markdown(report_row: dict[str, float | str]) -> str:
    return "\n".join(
        [
            "# Metric-only Baseline Report",
            "",
            "| Variant | Recall@3 | Recall@5 | Recall@8 | Soft Recall@8 | Avg Candidates |",
            "|---|---:|---:|---:|---:|---:|",
            (
                f"| {report_row['variant']} | "
                f"{float(report_row['recall_at_3']):.3f} | "
                f"{float(report_row['recall_at_5']):.3f} | "
                f"{float(report_row['recall_at_8']):.3f} | "
                f"{float(report_row['soft_recall_at_8']):.3f} | "
                f"{float(report_row['avg_candidates']):.2f} |"
            ),
            "",
        ]
    )


def _build_metric_results(
    adapter: OpenRCABankAdapter,
    sidecars: Sequence[Phase1CaseSidecar],
    max_candidates: int,
) -> dict[str, MetricCandidateBuildResult]:
    results: dict[str, MetricCandidateBuildResult] = {}
    for sidecar in sidecars:
        metrics_path = adapter.telemetry_path(sidecar.case_id, "metrics")
        results[sidecar.case_id] = build_metric_candidates_for_case(
            case_id=sidecar.case_id,
            metrics_path=metrics_path,
            windows=sidecar.incident_window,
            max_candidates=max_candidates,
        )
    return results


def run_metric_baseline(
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    max_candidates: int = 8,
) -> dict[str, Any]:
    manifest = load_case_manifest(manifest_path)
    adapter = OpenRCABankAdapter(data_root)
    sidecars = build_phase1_sidecars(adapter, manifest)
    metric_results = _build_metric_results(adapter, sidecars, max_candidates=max_candidates)
    candidates_by_case: dict[str, list[RcaCandidate]] = {
        case_id: result.candidates for case_id, result in metric_results.items()
    }

    targets_by_case = candidates_by_case_to_targets(candidates_by_case)
    recall_summary = evaluate_component_recall_at_k(sidecars, targets_by_case, ks=[3, 5, 8])
    report_row = summarize_variant_recall("M", sidecars, candidates_by_case, ks=[3, 5, 8])
    markdown = build_metric_baseline_report_markdown(report_row)

    for case_id, result in metric_results.items():
        _write_json(output_dir / "candidates" / "M" / f"{case_id}.json", _candidate_payload(result))

    _write_json(output_dir / "recall_M.json", recall_summary.model_dump(mode="json"))
    _write_json(output_dir / "report_M.json", report_row)
    _write_json(
        output_dir / "candidate_rankings_M.json",
        {case_id: targets for case_id, targets in sorted(targets_by_case.items())},
    )
    (output_dir / "baseline_report.md").write_text(markdown, encoding="utf-8")

    return {
        "metric_results": {
            case_id: result.model_dump(mode="json")
            for case_id, result in sorted(metric_results.items())
        },
        "recall_summary": recall_summary.model_dump(mode="json"),
        "report_row": report_row,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run OpenRCA Phase 2 Metric-only baseline")
    parser.add_argument("--manifest", type=Path, default=Path("configs/openrca_pilot15.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data/OpenRCA/Bank"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openrca_phase2_metric_pilot15"))
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args(argv)

    run_metric_baseline(
        manifest_path=args.manifest,
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_phase2_metric.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/openrca/phase2_metric.py tests/openrca/test_phase2_metric.py
git commit -m "feat: add metric-only baseline pipeline"
```

Expected: commit succeeds.

---

### Task 6: Real Pilot 15 Acceptance

**Files:**
- Modify: `tests/openrca/test_phase2_metric.py`

- [ ] **Step 1: Add a real pilot acceptance test**

Append this test to `tests/openrca/test_phase2_metric.py`:

```python
def test_real_pilot15_metric_baseline_acceptance(tmp_path: Path) -> None:
    manifest_path = Path("configs/openrca_pilot15.yaml")
    data_root = Path("data/OpenRCA/Bank")
    if not manifest_path.exists() or not data_root.exists():
        return

    result = run_metric_baseline(
        manifest_path=manifest_path,
        data_root=data_root,
        output_dir=tmp_path / "phase2_metric",
    )

    metric_results = result["metric_results"]
    assert len(metric_results) == 15
    for case_id, payload in metric_results.items():
        candidates = payload["candidates"]
        assert len(candidates) <= 8
        for candidate in candidates:
            assert candidate["candidate_key"] == f"{candidate['target_type']}:{candidate['canonical_target']}"
            assert candidate["variant_candidate_id"] == (
                f"cand:{case_id}:M:{candidate['rank']}:{candidate['target_type']}:{candidate['canonical_target']}"
            )
            assert candidate["present_in_variants"] == ["M"]
            assert candidate["sources"] == ["metric"]

    report_row = result["report_row"]
    assert report_row["variant"] == "M"
    assert 0.0 <= report_row["recall_at_3"] <= 1.0
    assert 0.0 <= report_row["recall_at_5"] <= 1.0
    assert 0.0 <= report_row["recall_at_8"] <= 1.0
    assert report_row["avg_candidates"] <= 8.0
```

- [ ] **Step 2: Run the acceptance test**

Run:

```bash
uv run pytest tests/openrca/test_phase2_metric.py::test_real_pilot15_metric_baseline_acceptance -v
```

Expected: PASS when `data/OpenRCA/Bank` and `configs/openrca_pilot15.yaml` are available. If either path is unavailable in the execution environment, the test returns without assertions.

- [ ] **Step 3: Run the real pilot pipeline**

Run:

```bash
uv run python -m vlm4rca.openrca.phase2_metric --manifest configs/openrca_pilot15.yaml --data-root data/OpenRCA/Bank --output-dir outputs/openrca_phase2_metric_pilot15
```

Expected: command exits with status 0 and writes:

```text
outputs/openrca_phase2_metric_pilot15/baseline_report.md
outputs/openrca_phase2_metric_pilot15/candidate_rankings_M.json
outputs/openrca_phase2_metric_pilot15/recall_M.json
outputs/openrca_phase2_metric_pilot15/report_M.json
outputs/openrca_phase2_metric_pilot15/candidates/M/task3_001.json
```

- [ ] **Step 4: Verify pilot output invariants**

Run:

```bash
uv run python -c "import json, pathlib; root=pathlib.Path('outputs/openrca_phase2_metric_pilot15'); files=sorted((root/'candidates'/'M').glob('*.json')); assert len(files)==15, len(files); payloads=[json.loads(path.read_text()) for path in files]; assert all(len(p['candidates'])<=8 for p in payloads); assert all(c['candidate_key']==f\"{c['target_type']}:{c['canonical_target']}\" for p in payloads for c in p['candidates']); assert all(c['variant_candidate_id']==f\"cand:{p['case_id']}:M:{c['rank']}:{c['target_type']}:{c['canonical_target']}\" for p in payloads for c in p['candidates']); print(len(files)); print(max(len(p['candidates']) for p in payloads)); print((root/'baseline_report.md').read_text())"
```

Expected: output prints `15`, a maximum candidate count no greater than `8`, and the Markdown table with the `M` row.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/openrca/test_phase2_metric.py
git commit -m "test: cover metric baseline pilot acceptance"
```

Expected: commit succeeds.

---

### Task 7: Final Verification

**Files:**
- No new source files in this task.

- [ ] **Step 1: Run the full test suite**

Run:

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Run lint**

Run:

```bash
uv run ruff check src/ tests/
```

Expected: no lint errors.

- [ ] **Step 3: Run formatter check**

Run:

```bash
uv run ruff format --check src/ tests/
```

Expected: all files are already formatted.

- [ ] **Step 4: Inspect Phase 2 output summary**

Run:

```bash
uv run python -c "import json, pathlib; root=pathlib.Path('outputs/openrca_phase2_metric_pilot15'); print(json.dumps(json.loads((root/'report_M.json').read_text()), indent=2, sort_keys=True)); print((root/'baseline_report.md').read_text())"
```

Expected: output includes numeric `recall_at_3`, `recall_at_5`, `recall_at_8`, `soft_recall_at_8`, `avg_candidates`, and a Markdown table with the `M` variant row.

- [ ] **Step 5: Confirm no forbidden Phase 2 outputs were produced**

Run:

```bash
find outputs/openrca_phase2_metric_pilot15 -type f | grep -E '\.png$|\.jpg$|\.jpeg$|\.svg$|\.html$' || true
```

Expected: no image or HTML paths are printed.

- [ ] **Step 6: Commit final verification notes if tracked docs changed**

Run:

```bash
git status --short
```

Expected: only intended Phase 2 source and test files are modified or created. Runtime files under `outputs/` may exist locally and should not be committed.

---

## Self-Review

Spec coverage:

- Metric candidate builder: Task 3 builds deterministic Metric-only candidates from `metrics.csv`.
- Metric feature scoring with robust z-score, relative change, p95 shift: Task 2 implements and tests all three.
- `M` variant top-8 candidate output: Task 1 enforces the contract, Task 3 caps builder output, Task 6 verifies real pilot output.
- Component Recall@3/5/8: Task 4 evaluates recall at the required K values, Task 5 writes `recall_M.json`.
- First baseline report table: Task 5 writes `baseline_report.md` with the required `M` row.
- Pilot 15 acceptance: Task 6 runs the real manifest and verifies 15 case result files.
- Stable `candidate_key` and `variant_candidate_id`: Task 1 validates both, Task 6 verifies them on the real pilot output.

Placeholder scan:

- The plan contains concrete file paths, test bodies, implementation bodies, commands, and expected outputs for every task.
- No placeholder implementation markers remain.

Type consistency:

- Candidate types are defined once in `src/vlm4rca/candidates/models.py` and imported by builder, evaluator tests, and pipeline code.
- `MetricCandidateBuildResult.candidates` contains `RcaCandidate` objects everywhere.
- Recall helpers continue to accept Phase 1 string rankings through `evaluate_component_recall_at_k`, while Phase 2 converts candidate objects with `candidates_by_case_to_targets`.
