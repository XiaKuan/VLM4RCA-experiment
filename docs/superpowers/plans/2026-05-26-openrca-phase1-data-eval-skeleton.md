# OpenRCA Phase 1 Data And Evaluation Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 data and evaluation foundation for the OpenRCA-Bank pilot: adapter interface, fixed 15-case manifest, incident windows, canonical component mapping, modality availability records, ground-truth mapping sidecars, and an empty Recall@K evaluator.

**Architecture:** Add a small `vlm4rca.openrca` package with focused modules for manifest loading, dataset access, windows, canonicalization, modality detection, ground-truth mapping, and Phase 1 output writing. Keep candidate retrieval, plotting, VLM calls, and model clients out of this phase by construction. Add a separate `vlm4rca.evaluation.recall` module that can run with empty candidate rankings and return deterministic zero-recall summaries.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, pytest, uv, pathlib/json/csv standard libraries.

---

## Scope Boundary

Phase 1 is accepted when it can:

- Read the fixed pilot 15 cases from `data/OpenRCA/Bank/cases`.
- Output one `modality_availability` record per case.
- Generate a ground-truth mapping sidecar with `raw_ground_truth -> mapped_component`.
- Compute baseline and incident windows from `inject_time`.
- Run an empty Recall@K evaluator with no candidate lists.

Phase 1 must not:

- Generate candidate retrieval outputs.
- Render plots, images, or topology diagrams.
- Call VLM, LLM, OpenAI-compatible APIs, or any model client.
- Use ground truth outside sidecar and evaluator tests.

## File Structure

Create these files:

- `configs/openrca_pilot15.yaml` - fixed Phase 1 pilot case manifest with 15 OpenRCA-Bank case IDs.
- `src/vlm4rca/openrca/__init__.py` - exports the public Phase 1 data helpers.
- `src/vlm4rca/openrca/models.py` - Pydantic models shared by manifest, sidecars, windows, and availability.
- `src/vlm4rca/openrca/manifest.py` - YAML manifest loader and duplicate validation.
- `src/vlm4rca/openrca/adapter.py` - `OpenRCAAdapter` protocol and `OpenRCABankAdapter`.
- `src/vlm4rca/openrca/windows.py` - incident and baseline window extraction from `inject_time`.
- `src/vlm4rca/openrca/canonicalization.py` - service/component canonicalization and mapping helpers.
- `src/vlm4rca/openrca/ground_truth.py` - raw ground-truth extraction and sidecar mapping.
- `src/vlm4rca/openrca/modality.py` - metrics/logs/traces/topology availability detection.
- `src/vlm4rca/openrca/phase1.py` - Phase 1 orchestration and CLI module.
- `src/vlm4rca/evaluation/__init__.py` - evaluation package marker.
- `src/vlm4rca/evaluation/recall.py` - empty-run-capable Recall@K evaluator.

Create these tests:

- `tests/openrca/test_manifest.py`
- `tests/openrca/test_adapter.py`
- `tests/openrca/test_windows.py`
- `tests/openrca/test_canonicalization.py`
- `tests/openrca/test_ground_truth.py`
- `tests/openrca/test_modality.py`
- `tests/evaluation/test_recall.py`
- `tests/openrca/test_phase1.py`

Generated runtime outputs go under `outputs/openrca_phase1_pilot15/` and are not committed.

---

### Task 0: Execution Worktree

**Files:**
- No repository files changed in this task.

- [ ] **Step 1: Create an isolated worktree**

Run:

```bash
mkdir -p .claude/worktrees/feat
git worktree add .claude/worktrees/feat/openrca-phase1-data-eval -b feat/openrca-phase1-data-eval main
cd .claude/worktrees/feat/openrca-phase1-data-eval
```

Expected: command succeeds and `pwd` ends with `.claude/worktrees/feat/openrca-phase1-data-eval`.

- [ ] **Step 2: Install dependencies**

Run:

```bash
uv sync
```

Expected: dependencies are installed without errors.

- [ ] **Step 3: Verify the starting test suite**

Run:

```bash
uv run pytest tests/ -v
```

Expected: current tests pass or the only output is the existing empty-test-suite status. Record any pre-existing failure before editing files.

---

### Task 1: Shared Models And Pilot Manifest

**Files:**
- Create: `src/vlm4rca/openrca/__init__.py`
- Create: `src/vlm4rca/openrca/models.py`
- Create: `src/vlm4rca/openrca/manifest.py`
- Create: `configs/openrca_pilot15.yaml`
- Create: `tests/openrca/test_manifest.py`

- [ ] **Step 1: Write the failing manifest tests**

Create `tests/openrca/test_manifest.py`:

```python
from pathlib import Path

import pytest

from vlm4rca.openrca.manifest import load_case_manifest


def test_loads_pilot_manifest_with_fixed_15_cases() -> None:
    manifest = load_case_manifest(Path("configs/openrca_pilot15.yaml"))

    assert manifest.dataset == "openrca_bank"
    assert manifest.data_root == "data/OpenRCA/Bank"
    assert len(manifest.cases) == 15
    assert [case.case_id for case in manifest.cases[:3]] == [
        "task3_008",
        "task3_009",
        "task3_010",
    ]
    assert {case.case_group for case in manifest.cases} == {
        "metric_obvious",
        "soft_latency",
        "mixed_ambiguous",
    }


def test_manifest_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "dataset: openrca_bank",
                "data_root: data/OpenRCA/Bank",
                "cases:",
                "  - case_id: task3_001",
                "    case_group: soft_latency",
                "    selection_reason: first",
                "  - case_id: task3_001",
                "    case_group: soft_latency",
                "    selection_reason: duplicate",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate case_id values"):
        load_case_manifest(manifest_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_manifest.py -v
```

Expected: FAIL because `vlm4rca.openrca.manifest` does not exist.

- [ ] **Step 3: Add the shared models**

Create `src/vlm4rca/openrca/models.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

CaseGroup = Literal["metric_obvious", "soft_latency", "mixed_ambiguous"]
TopologySource = Literal["trace_derived", "static_topology", "unavailable"]
MappingConfidence = Literal["exact", "heuristic", "unknown"]
MappingType = Literal[
    "component_exact",
    "normalized_component",
    "pod_to_service",
    "suffix_stripped",
    "unknown",
]


class ManifestCase(BaseModel):
    case_id: str
    case_group: CaseGroup
    selection_reason: str


class CaseManifest(BaseModel):
    dataset: str
    data_root: str
    cases: list[ManifestCase]


class IncidentWindows(BaseModel):
    inject_time: int
    baseline_start: int
    baseline_end: int
    incident_start: int
    incident_end: int
    pre_window_seconds: int
    post_window_seconds: int
    baseline_window_seconds: int
    boundary_rule: str = "[start, end)"
    timezone: str = "UTC"
    sampling_interval_assumption: str = "epoch seconds; telemetry timestamps are interpreted as UTC seconds"
    warnings: list[str] = Field(default_factory=list)


class ModalityAvailability(BaseModel):
    metrics_available: bool
    traces_available: bool
    logs_available: bool
    topology_available: bool
    topology_source: TopologySource
    warnings: list[str] = Field(default_factory=list)


class GroundTruthMapping(BaseModel):
    raw_ground_truth: str
    mapped_component: str
    mapping_type: MappingType
    mapping_confidence: MappingConfidence
    source: str


class Phase1CaseSidecar(BaseModel):
    case_id: str
    case_group: CaseGroup
    task_index: str | None = None
    inject_time: int
    case_path: str
    raw_ground_truth: list[str]
    incident_window: IncidentWindows
    modality_availability: ModalityAvailability
    gt_mapping: list[GroundTruthMapping]


class RecallCaseResult(BaseModel):
    case_id: str
    hit_at_k: dict[str, bool]
    hit_targets: list[str]
    missed_targets: list[str]


class RecallAtKSummary(BaseModel):
    n_cases: int
    ks: list[int]
    component_recall_at_k: dict[str, float]
    case_results: list[RecallCaseResult]
```

- [ ] **Step 4: Add the manifest loader**

Create `src/vlm4rca/openrca/manifest.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from vlm4rca.openrca.models import CaseManifest


def load_case_manifest(path: Path) -> CaseManifest:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a YAML mapping: {path}")

    manifest = CaseManifest.model_validate(payload)
    case_ids = [case.case_id for case in manifest.cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        joined = ", ".join(duplicates)
        raise ValueError(f"Duplicate case_id values in manifest: {joined}")
    return manifest


def manifest_to_jsonable(manifest: CaseManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json")
```

- [ ] **Step 5: Add package exports**

Create `src/vlm4rca/openrca/__init__.py`:

```python
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.models import (
    CaseManifest,
    GroundTruthMapping,
    IncidentWindows,
    ModalityAvailability,
    Phase1CaseSidecar,
)

__all__ = [
    "CaseManifest",
    "GroundTruthMapping",
    "IncidentWindows",
    "ModalityAvailability",
    "Phase1CaseSidecar",
    "load_case_manifest",
]
```

- [ ] **Step 6: Add the fixed pilot manifest**

Create `configs/openrca_pilot15.yaml`:

```yaml
dataset: openrca_bank
data_root: data/OpenRCA/Bank
description: "Phase 1 fixed pilot subset for data and evaluation skeleton validation."
cases:
  - case_id: task3_008
    case_group: metric_obvious
    selection_reason: "MG02 high JVM CPU load; metric-heavy component localization case."
  - case_id: task3_009
    case_group: metric_obvious
    selection_reason: "Tomcat02 high disk I/O read usage; metric-heavy component localization case."
  - case_id: task3_010
    case_group: metric_obvious
    selection_reason: "IG01 high CPU usage; metric-heavy component localization case."
  - case_id: task3_014
    case_group: metric_obvious
    selection_reason: "Tomcat02 high disk I/O read usage; repeated metric-heavy service case."
  - case_id: task3_015
    case_group: metric_obvious
    selection_reason: "Mysql02 high CPU usage; database component metric-heavy case."
  - case_id: task3_001
    case_group: soft_latency
    selection_reason: "Tomcat01 network latency; trace and latency-sensitive case."
  - case_id: task3_002
    case_group: soft_latency
    selection_reason: "Tomcat02 network latency; trace and latency-sensitive case."
  - case_id: task3_004
    case_group: soft_latency
    selection_reason: "MG02 network latency; gateway-like dependency latency case."
  - case_id: task3_005
    case_group: soft_latency
    selection_reason: "apache02 network latency; edge-facing latency case."
  - case_id: task3_013
    case_group: soft_latency
    selection_reason: "MG01 network latency; later-date dependency latency case."
  - case_id: task3_003_multi
    case_group: mixed_ambiguous
    selection_reason: "Two matched faults across MG01 and Tomcat04; validates multi-ground-truth sidecars."
  - case_id: task6_008_multi
    case_group: mixed_ambiguous
    selection_reason: "Two apache02 network reasons in one case; validates duplicate raw component handling."
  - case_id: task6_009_multi
    case_group: mixed_ambiguous
    selection_reason: "MG02 JVM OOM and Tomcat03 disk I/O; mixed metric and resource failure."
  - case_id: task7_006_multi
    case_group: mixed_ambiguous
    selection_reason: "IG02 disk I/O and apache02 packet loss; mixed component/reason case."
  - case_id: task7_011_multi
    case_group: mixed_ambiguous
    selection_reason: "IG02 and MG02 disk I/O faults; multi-component metric-heavy case."
```

- [ ] **Step 7: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_manifest.py -v
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

Run:

```bash
git add configs/openrca_pilot15.yaml src/vlm4rca/openrca/__init__.py src/vlm4rca/openrca/models.py src/vlm4rca/openrca/manifest.py tests/openrca/test_manifest.py
git commit -m "feat: add OpenRCA pilot manifest models"
```

Expected: commit succeeds.

---

### Task 2: OpenRCA Adapter Interface

**Files:**
- Create: `src/vlm4rca/openrca/adapter.py`
- Create: `tests/openrca/test_adapter.py`

- [ ] **Step 1: Write the failing adapter tests**

Create `tests/openrca/test_adapter.py`:

```python
import json
from pathlib import Path

from vlm4rca.openrca.adapter import OpenRCABankAdapter


def _write_case(root: Path, case_id: str, *, include_logs: bool = True) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1000,
                "matched_faults": [{"component": "Tomcat01", "reason": "network latency", "timestamp": 1000}],
                "system_components": ["Tomcat01"],
                "evidence_components": ["Tomcat01"],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "metrics.csv").write_text("timestamp,Tomcat01__cpu\n940,1\n", encoding="utf-8")
    (case_dir / "traces.csv").write_text(
        "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n940,940,Tomcat01,t,s,p,1\n",
        encoding="utf-8",
    )
    if include_logs:
        (case_dir / "logs.csv").write_text("time,timestamp,service,log_name,message\n940,940,Tomcat01,app,ok\n", encoding="utf-8")


def test_bank_adapter_lists_cases_and_loads_metadata(tmp_path: Path) -> None:
    _write_case(tmp_path, "task3_002")
    _write_case(tmp_path, "task3_001")

    adapter = OpenRCABankAdapter(tmp_path)

    assert adapter.list_case_ids() == ["task3_001", "task3_002"]
    assert adapter.case_path("task3_001") == tmp_path / "cases" / "task3_001"
    assert adapter.load_case_meta("task3_001")["matched_faults"][0]["component"] == "Tomcat01"


def test_bank_adapter_reports_existing_non_empty_telemetry(tmp_path: Path) -> None:
    _write_case(tmp_path, "task3_001", include_logs=False)
    (tmp_path / "cases" / "task3_001" / "logs.csv").write_text("", encoding="utf-8")

    adapter = OpenRCABankAdapter(tmp_path)

    assert adapter.telemetry_exists("task3_001", "metrics") is True
    assert adapter.telemetry_exists("task3_001", "traces") is True
    assert adapter.telemetry_exists("task3_001", "logs") is False
    assert adapter.telemetry_exists("task3_001", "topology") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_adapter.py -v
```

Expected: FAIL because `vlm4rca.openrca.adapter` does not exist.

- [ ] **Step 3: Implement the adapter protocol and Bank adapter**

Create `src/vlm4rca/openrca/adapter.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

TelemetryKind = Literal["metrics", "logs", "traces", "topology"]

_TELEMETRY_FILES: dict[TelemetryKind, str] = {
    "metrics": "metrics.csv",
    "logs": "logs.csv",
    "traces": "traces.csv",
    "topology": "topology.csv",
}


class OpenRCAAdapter(Protocol):
    def list_case_ids(self) -> list[str]:
        raise NotImplementedError

    def case_path(self, case_id: str) -> Path:
        raise NotImplementedError

    def load_case_meta(self, case_id: str) -> dict[str, Any]:
        raise NotImplementedError

    def telemetry_path(self, case_id: str, kind: TelemetryKind) -> Path:
        raise NotImplementedError

    def telemetry_exists(self, case_id: str, kind: TelemetryKind) -> bool:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenRCABankAdapter:
    data_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", Path(self.data_root))

    @property
    def cases_dir(self) -> Path:
        return self.data_root / "cases"

    def list_case_ids(self) -> list[str]:
        if not self.cases_dir.exists():
            raise FileNotFoundError(f"OpenRCA cases directory does not exist: {self.cases_dir}")
        return sorted(
            path.name
            for path in self.cases_dir.iterdir()
            if path.is_dir() and (path / "case_meta.json").exists()
        )

    def case_path(self, case_id: str) -> Path:
        path = self.cases_dir / case_id
        if not path.exists():
            raise FileNotFoundError(f"OpenRCA case directory does not exist: {path}")
        return path

    def load_case_meta(self, case_id: str) -> dict[str, Any]:
        meta_path = self.case_path(case_id) / "case_meta.json"
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def telemetry_path(self, case_id: str, kind: TelemetryKind) -> Path:
        return self.case_path(case_id) / _TELEMETRY_FILES[kind]

    def telemetry_exists(self, case_id: str, kind: TelemetryKind) -> bool:
        path = self.telemetry_path(case_id, kind)
        return path.exists() and path.stat().st_size > 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_adapter.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/openrca/adapter.py tests/openrca/test_adapter.py
git commit -m "feat: add OpenRCA bank adapter"
```

Expected: commit succeeds.

---

### Task 3: Incident And Baseline Window Extraction

**Files:**
- Create: `src/vlm4rca/openrca/windows.py`
- Create: `tests/openrca/test_windows.py`

- [ ] **Step 1: Write the failing window tests**

Create `tests/openrca/test_windows.py`:

```python
from vlm4rca.openrca.windows import WindowConfig, extract_incident_windows


def test_extracts_default_phase1_windows_from_inject_time() -> None:
    windows = extract_incident_windows(
        {
            "inject_time": 1_614_841_020,
            "context_start": 1_614_835_800,
            "context_end": 1_614_843_000,
        }
    )

    assert windows.baseline_start == 1_614_838_620
    assert windows.baseline_end == 1_614_840_420
    assert windows.incident_start == 1_614_840_420
    assert windows.incident_end == 1_614_842_220
    assert windows.boundary_rule == "[start, end)"
    assert windows.timezone == "UTC"
    assert windows.warnings == []


def test_records_context_coverage_warnings() -> None:
    windows = extract_incident_windows(
        {
            "inject_time": 1_000,
            "context_start": 100,
            "context_end": 1_100,
        },
        WindowConfig(pre_window_seconds=100, post_window_seconds=200, baseline_window_seconds=300),
    )

    assert windows.baseline_start == 600
    assert windows.baseline_end == 900
    assert windows.incident_start == 900
    assert windows.incident_end == 1_200
    assert windows.warnings == ["context_end 1100 is before incident_end 1200"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_windows.py -v
```

Expected: FAIL because `vlm4rca.openrca.windows` does not exist.

- [ ] **Step 3: Implement window extraction**

Create `src/vlm4rca/openrca/windows.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vlm4rca.openrca.models import IncidentWindows


@dataclass(frozen=True)
class WindowConfig:
    pre_window_seconds: int = 10 * 60
    post_window_seconds: int = 20 * 60
    baseline_window_seconds: int = 30 * 60
    timezone: str = "UTC"


def extract_incident_windows(
    case_meta: Mapping[str, Any],
    config: WindowConfig = WindowConfig(),
) -> IncidentWindows:
    if "inject_time" not in case_meta:
        raise KeyError("case_meta is missing inject_time")

    inject_time = int(case_meta["inject_time"])
    baseline_start = inject_time - config.pre_window_seconds - config.baseline_window_seconds
    baseline_end = inject_time - config.pre_window_seconds
    incident_start = baseline_end
    incident_end = inject_time + config.post_window_seconds

    warnings: list[str] = []
    if "context_start" in case_meta and int(case_meta["context_start"]) > baseline_start:
        warnings.append(
            f"context_start {int(case_meta['context_start'])} is after baseline_start {baseline_start}"
        )
    if "context_end" in case_meta and int(case_meta["context_end"]) < incident_end:
        warnings.append(f"context_end {int(case_meta['context_end'])} is before incident_end {incident_end}")

    return IncidentWindows(
        inject_time=inject_time,
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        incident_start=incident_start,
        incident_end=incident_end,
        pre_window_seconds=config.pre_window_seconds,
        post_window_seconds=config.post_window_seconds,
        baseline_window_seconds=config.baseline_window_seconds,
        timezone=config.timezone,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_windows.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/openrca/windows.py tests/openrca/test_windows.py
git commit -m "feat: add incident window extraction"
```

Expected: commit succeeds.

---

### Task 4: Component Canonicalization And Ground Truth Mapping

**Files:**
- Create: `src/vlm4rca/openrca/canonicalization.py`
- Create: `src/vlm4rca/openrca/ground_truth.py`
- Create: `tests/openrca/test_canonicalization.py`
- Create: `tests/openrca/test_ground_truth.py`

- [ ] **Step 1: Write failing canonicalization tests**

Create `tests/openrca/test_canonicalization.py`:

```python
from vlm4rca.openrca.canonicalization import canonicalize_component, map_component_name


def test_canonicalize_component_normalizes_case_underscore_and_prefix() -> None:
    assert canonicalize_component("pod/Checkout_Service") == "checkout-service"
    assert canonicalize_component(" service:Tomcat01 ") == "tomcat01"


def test_map_component_exact_match_against_known_components() -> None:
    mapping = map_component_name("Tomcat01", ["Tomcat01", "Mysql02"])

    assert mapping.raw_ground_truth == "Tomcat01"
    assert mapping.mapped_component == "tomcat01"
    assert mapping.mapping_type == "component_exact"
    assert mapping.mapping_confidence == "exact"


def test_map_component_strips_resource_suffix_when_known_component_matches() -> None:
    mapping = map_component_name(
        "pod/payment-service-5f7c9d7b7c-abcde",
        ["payment-service", "checkout-service"],
    )

    assert mapping.mapped_component == "payment-service"
    assert mapping.mapping_type == "suffix_stripped"
    assert mapping.mapping_confidence == "heuristic"
```

- [ ] **Step 2: Write failing ground-truth tests**

Create `tests/openrca/test_ground_truth.py`:

```python
from vlm4rca.openrca.ground_truth import build_ground_truth_mappings, extract_raw_ground_truth


def test_extracts_unique_raw_ground_truth_from_matched_faults() -> None:
    case_meta = {
        "matched_faults": [
            {"component": "apache02", "reason": "network packet loss", "timestamp": 1},
            {"component": "apache02", "reason": "network latency", "timestamp": 2},
            {"component": "MG01", "reason": "network latency", "timestamp": 3},
        ]
    }

    assert extract_raw_ground_truth(case_meta) == ["apache02", "MG01"]


def test_builds_ground_truth_mapping_sidecar_entries() -> None:
    case_meta = {
        "matched_faults": [{"component": "Tomcat01", "reason": "network latency", "timestamp": 1}],
        "system_components": ["Tomcat01", "Mysql02"],
        "evidence_components": ["Tomcat01", "Redis01"],
    }

    mappings = build_ground_truth_mappings(case_meta)

    assert len(mappings) == 1
    assert mappings[0].raw_ground_truth == "Tomcat01"
    assert mappings[0].mapped_component == "tomcat01"
    assert mappings[0].source == "case_meta.matched_faults.component"
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_canonicalization.py tests/openrca/test_ground_truth.py -v
```

Expected: FAIL because canonicalization and ground-truth modules do not exist.

- [ ] **Step 4: Implement component canonicalization**

Create `src/vlm4rca/openrca/canonicalization.py`:

```python
from __future__ import annotations

import re

from vlm4rca.openrca.models import GroundTruthMapping

_PREFIXES = ("pod/", "container/", "service/", "service:", "resource:")
_HASH_SEGMENT_RE = re.compile(r"^[a-f0-9]{5,}$")
_ORDINAL_RE = re.compile(r"^\d+$")


def canonicalize_component(raw: str) -> str:
    value = raw.strip()
    lowered = value.lower()
    for prefix in _PREFIXES:
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.replace("_", "-")
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.lower().strip("-")


def strip_resource_suffix(canonical_name: str) -> str:
    parts = canonical_name.split("-")
    if len(parts) >= 3 and _HASH_SEGMENT_RE.fullmatch(parts[-2]) and _HASH_SEGMENT_RE.fullmatch(parts[-1]):
        return "-".join(parts[:-2])
    if len(parts) >= 2 and _ORDINAL_RE.fullmatch(parts[-1]):
        return "-".join(parts[:-1])
    return canonical_name


def map_component_name(raw_ground_truth: str, known_components: list[str]) -> GroundTruthMapping:
    canonical_known = {canonicalize_component(component): component for component in known_components}
    canonical_raw = canonicalize_component(raw_ground_truth)

    if canonical_raw in canonical_known:
        return GroundTruthMapping(
            raw_ground_truth=raw_ground_truth,
            mapped_component=canonical_raw,
            mapping_type="component_exact",
            mapping_confidence="exact",
            source="case_meta.matched_faults.component",
        )

    stripped = strip_resource_suffix(canonical_raw)
    if stripped in canonical_known:
        return GroundTruthMapping(
            raw_ground_truth=raw_ground_truth,
            mapped_component=stripped,
            mapping_type="suffix_stripped",
            mapping_confidence="heuristic",
            source="case_meta.matched_faults.component",
        )

    return GroundTruthMapping(
        raw_ground_truth=raw_ground_truth,
        mapped_component=canonical_raw,
        mapping_type="unknown",
        mapping_confidence="unknown",
        source="case_meta.matched_faults.component",
    )
```

- [ ] **Step 5: Implement ground truth extraction**

Create `src/vlm4rca/openrca/ground_truth.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vlm4rca.openrca.canonicalization import map_component_name
from vlm4rca.openrca.models import GroundTruthMapping


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def extract_raw_ground_truth(case_meta: Mapping[str, Any]) -> list[str]:
    matched_faults = case_meta.get("matched_faults") or []
    components = [
        str(fault["component"])
        for fault in matched_faults
        if isinstance(fault, Mapping) and fault.get("component")
    ]
    if components:
        return _dedupe_preserving_order(components)

    fallback = [str(component) for component in case_meta.get("ground_truth_components") or []]
    return _dedupe_preserving_order(fallback)


def build_ground_truth_mappings(case_meta: Mapping[str, Any]) -> list[GroundTruthMapping]:
    known_components = [
        str(component)
        for component in [
            *(case_meta.get("system_components") or []),
            *(case_meta.get("evidence_components") or []),
        ]
    ]
    return [
        map_component_name(raw_ground_truth, known_components)
        for raw_ground_truth in extract_raw_ground_truth(case_meta)
    ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_canonicalization.py tests/openrca/test_ground_truth.py -v
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

Run:

```bash
git add src/vlm4rca/openrca/canonicalization.py src/vlm4rca/openrca/ground_truth.py tests/openrca/test_canonicalization.py tests/openrca/test_ground_truth.py
git commit -m "feat: map OpenRCA ground truth components"
```

Expected: commit succeeds.

---

### Task 5: Modality Availability

**Files:**
- Create: `src/vlm4rca/openrca/modality.py`
- Create: `tests/openrca/test_modality.py`

- [ ] **Step 1: Write failing modality tests**

Create `tests/openrca/test_modality.py`:

```python
from pathlib import Path

from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.modality import detect_modality_availability


def _write_case_files(case_dir: Path, *, traces: bool = True, topology: bool = False) -> None:
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text('{"case_id": "case_1", "inject_time": 1000}', encoding="utf-8")
    (case_dir / "metrics.csv").write_text("timestamp,cpu\n1,2\n", encoding="utf-8")
    (case_dir / "logs.csv").write_text("time,timestamp,service,log_name,message\n1,1,a,app,ok\n", encoding="utf-8")
    if traces:
        (case_dir / "traces.csv").write_text(
            "time,timestamp,service,trace_id,span_id,parent_span_id,duration\n1,1,a,t,s,p,2\n",
            encoding="utf-8",
        )
    if topology:
        (case_dir / "topology.csv").write_text("source,target\na,b\n", encoding="utf-8")


def test_detects_trace_derived_topology_when_traces_exist(tmp_path: Path) -> None:
    _write_case_files(tmp_path / "cases" / "case_1", traces=True)
    adapter = OpenRCABankAdapter(tmp_path)

    availability = detect_modality_availability(adapter, "case_1")

    assert availability.metrics_available is True
    assert availability.logs_available is True
    assert availability.traces_available is True
    assert availability.topology_available is True
    assert availability.topology_source == "trace_derived"
    assert availability.warnings == []


def test_static_topology_is_available_without_traces(tmp_path: Path) -> None:
    _write_case_files(tmp_path / "cases" / "case_1", traces=False, topology=True)
    adapter = OpenRCABankAdapter(tmp_path)

    availability = detect_modality_availability(adapter, "case_1")

    assert availability.traces_available is False
    assert availability.topology_available is True
    assert availability.topology_source == "static_topology"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_modality.py -v
```

Expected: FAIL because `vlm4rca.openrca.modality` does not exist.

- [ ] **Step 3: Implement modality detection**

Create `src/vlm4rca/openrca/modality.py`:

```python
from __future__ import annotations

from vlm4rca.openrca.adapter import OpenRCAAdapter
from vlm4rca.openrca.models import ModalityAvailability, TopologySource


def detect_modality_availability(adapter: OpenRCAAdapter, case_id: str) -> ModalityAvailability:
    metrics_available = adapter.telemetry_exists(case_id, "metrics")
    logs_available = adapter.telemetry_exists(case_id, "logs")
    traces_available = adapter.telemetry_exists(case_id, "traces")
    static_topology_available = adapter.telemetry_exists(case_id, "topology")

    if static_topology_available:
        topology_available = True
        topology_source: TopologySource = "static_topology"
    elif traces_available:
        topology_available = True
        topology_source = "trace_derived"
    else:
        topology_available = False
        topology_source = "unavailable"

    warnings: list[str] = []
    if not metrics_available:
        warnings.append("metrics.csv is missing or empty")
    if not logs_available:
        warnings.append("logs.csv is missing or empty")
    if not traces_available:
        warnings.append("traces.csv is missing or empty")

    return ModalityAvailability(
        metrics_available=metrics_available,
        traces_available=traces_available,
        logs_available=logs_available,
        topology_available=topology_available,
        topology_source=topology_source,
        warnings=warnings,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_modality.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/openrca/modality.py tests/openrca/test_modality.py
git commit -m "feat: record OpenRCA modality availability"
```

Expected: commit succeeds.

---

### Task 6: Empty Recall@K Evaluator

**Files:**
- Create: `src/vlm4rca/evaluation/__init__.py`
- Create: `src/vlm4rca/evaluation/recall.py`
- Create: `tests/evaluation/test_recall.py`

- [ ] **Step 1: Write failing Recall@K tests**

Create `tests/evaluation/test_recall.py`:

```python
from vlm4rca.evaluation.recall import evaluate_component_recall_at_k
from vlm4rca.openrca.models import (
    GroundTruthMapping,
    IncidentWindows,
    ModalityAvailability,
    Phase1CaseSidecar,
)


def _sidecar(case_id: str, mapped_component: str) -> Phase1CaseSidecar:
    return Phase1CaseSidecar(
        case_id=case_id,
        case_group="metric_obvious",
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


def test_empty_candidate_rankings_return_zero_recall() -> None:
    summary = evaluate_component_recall_at_k(
        [_sidecar("case_1", "tomcat01"), _sidecar("case_2", "mysql02")],
        candidates_by_case={},
        ks=[3, 5, 8],
    )

    assert summary.n_cases == 2
    assert summary.component_recall_at_k == {"3": 0.0, "5": 0.0, "8": 0.0}
    assert summary.case_results[0].hit_at_k == {"3": False, "5": False, "8": False}
    assert summary.case_results[0].missed_targets == ["tomcat01"]


def test_candidate_rankings_can_hit_at_larger_k() -> None:
    summary = evaluate_component_recall_at_k(
        [_sidecar("case_1", "tomcat01")],
        candidates_by_case={"case_1": ["redis01", "mysql02", "tomcat01"]},
        ks=[1, 3],
    )

    assert summary.component_recall_at_k == {"1": 0.0, "3": 1.0}
    assert summary.case_results[0].hit_targets == ["tomcat01"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/evaluation/test_recall.py -v
```

Expected: FAIL because `vlm4rca.evaluation.recall` does not exist.

- [ ] **Step 3: Implement the evaluator**

Create `src/vlm4rca/evaluation/__init__.py`:

```python
from vlm4rca.evaluation.recall import evaluate_component_recall_at_k

__all__ = ["evaluate_component_recall_at_k"]
```

Create `src/vlm4rca/evaluation/recall.py`:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/evaluation/test_recall.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/evaluation/__init__.py src/vlm4rca/evaluation/recall.py tests/evaluation/test_recall.py
git commit -m "feat: add empty recall evaluator"
```

Expected: commit succeeds.

---

### Task 7: Phase 1 Pipeline And CLI

**Files:**
- Create: `src/vlm4rca/openrca/phase1.py`
- Create: `tests/openrca/test_phase1.py`

- [ ] **Step 1: Write failing Phase 1 pipeline tests**

Create `tests/openrca/test_phase1.py`:

```python
import json
from pathlib import Path

from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.phase1 import build_phase1_sidecars, main, write_phase1_outputs


def _write_case(root: Path, case_id: str, component: str) -> None:
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True)
    (case_dir / "case_meta.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "task_index": "task_3",
                "inject_time": 1000,
                "context_start": 0,
                "context_end": 3000,
                "matched_faults": [{"component": component, "reason": "network latency", "timestamp": 1000}],
                "system_components": [component],
                "evidence_components": [component],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "metrics.csv").write_text("timestamp,cpu\n1,2\n", encoding="utf-8")
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


def test_build_phase1_sidecars_from_adapter_and_manifest(tmp_path: Path) -> None:
    _write_case(tmp_path, "case_1", "Tomcat01")
    _write_case(tmp_path, "case_2", "Mysql02")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path)

    manifest = load_case_manifest(manifest_path)
    sidecars = build_phase1_sidecars(OpenRCABankAdapter(tmp_path), manifest)

    assert [sidecar.case_id for sidecar in sidecars] == ["case_1", "case_2"]
    assert sidecars[0].modality_availability.topology_source == "trace_derived"
    assert sidecars[0].gt_mapping[0].raw_ground_truth == "Tomcat01"
    assert sidecars[0].gt_mapping[0].mapped_component == "tomcat01"


def test_write_phase1_outputs_without_candidate_or_image_files(tmp_path: Path) -> None:
    _write_case(tmp_path / "data", "case_1", "Tomcat01")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, tmp_path / "data")
    sidecars = build_phase1_sidecars(OpenRCABankAdapter(tmp_path / "data"), load_case_manifest(manifest_path))

    output_dir = tmp_path / "outputs"
    write_phase1_outputs(sidecars, load_case_manifest(manifest_path), output_dir)

    assert (output_dir / "case_sidecars" / "case_1.json").exists()
    assert (output_dir / "modality_availability.json").exists()
    assert (output_dir / "ground_truth_mapping.json").exists()
    assert (output_dir / "recall_empty_run.json").exists()
    written_paths = [path.name for path in output_dir.rglob("*") if path.is_file()]
    assert not any("candidate" in name for name in written_paths)
    assert not any(name.endswith((".png", ".jpg", ".jpeg", ".svg", ".html")) for name in written_paths)


def test_cli_main_writes_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    _write_case(data_root, "case_1", "Tomcat01")
    _write_case(data_root, "case_2", "Mysql02")
    manifest_path = tmp_path / "manifest.yaml"
    _write_manifest(manifest_path, data_root)
    output_dir = tmp_path / "phase1"

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
    summary = json.loads((output_dir / "recall_empty_run.json").read_text(encoding="utf-8"))
    assert summary["n_cases"] == 2
    assert summary["component_recall_at_k"] == {"3": 0.0, "5": 0.0, "8": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/openrca/test_phase1.py -v
```

Expected: FAIL because `vlm4rca.openrca.phase1` does not exist.

- [ ] **Step 3: Implement Phase 1 orchestration and CLI**

Create `src/vlm4rca/openrca/phase1.py`:

```python
from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from vlm4rca.evaluation.recall import evaluate_component_recall_at_k
from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.ground_truth import build_ground_truth_mappings, extract_raw_ground_truth
from vlm4rca.openrca.manifest import load_case_manifest, manifest_to_jsonable
from vlm4rca.openrca.modality import detect_modality_availability
from vlm4rca.openrca.models import CaseManifest, Phase1CaseSidecar
from vlm4rca.openrca.windows import extract_incident_windows


def build_phase1_sidecars(
    adapter: OpenRCABankAdapter,
    manifest: CaseManifest,
) -> list[Phase1CaseSidecar]:
    available_case_ids = set(adapter.list_case_ids())
    sidecars: list[Phase1CaseSidecar] = []

    for manifest_case in manifest.cases:
        if manifest_case.case_id not in available_case_ids:
            raise FileNotFoundError(f"Manifest case_id is not present in OpenRCA-Bank cases: {manifest_case.case_id}")

        meta = adapter.load_case_meta(manifest_case.case_id)
        sidecars.append(
            Phase1CaseSidecar(
                case_id=manifest_case.case_id,
                case_group=manifest_case.case_group,
                task_index=meta.get("task_index"),
                inject_time=int(meta["inject_time"]),
                case_path=str(adapter.case_path(manifest_case.case_id)),
                raw_ground_truth=extract_raw_ground_truth(meta),
                incident_window=extract_incident_windows(meta),
                modality_availability=detect_modality_availability(adapter, manifest_case.case_id),
                gt_mapping=build_ground_truth_mappings(meta),
            )
        )

    return sidecars


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_phase1_outputs(
    sidecars: Sequence[Phase1CaseSidecar],
    manifest: CaseManifest,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest_cases.json", manifest_to_jsonable(manifest))

    for sidecar in sidecars:
        _write_json(
            output_dir / "case_sidecars" / f"{sidecar.case_id}.json",
            sidecar.model_dump(mode="json"),
        )

    _write_json(
        output_dir / "modality_availability.json",
        [
            {
                "case_id": sidecar.case_id,
                "case_group": sidecar.case_group,
                "modality_availability": sidecar.modality_availability.model_dump(mode="json"),
            }
            for sidecar in sidecars
        ],
    )
    _write_json(
        output_dir / "ground_truth_mapping.json",
        [
            {
                "case_id": sidecar.case_id,
                "case_group": sidecar.case_group,
                "gt_mapping": [mapping.model_dump(mode="json") for mapping in sidecar.gt_mapping],
            }
            for sidecar in sidecars
        ],
    )
    empty_recall = evaluate_component_recall_at_k(sidecars, candidates_by_case={}, ks=[3, 5, 8])
    _write_json(output_dir / "recall_empty_run.json", empty_recall.model_dump(mode="json"))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenRCA Phase 1 data and evaluation sidecars.")
    parser.add_argument("--manifest", type=Path, default=Path("configs/openrca_pilot15.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openrca_phase1_pilot15"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest = load_case_manifest(args.manifest)
    data_root = args.data_root if args.data_root is not None else Path(manifest.data_root)
    adapter = OpenRCABankAdapter(data_root)
    sidecars = build_phase1_sidecars(adapter, manifest)
    write_phase1_outputs(sidecars, manifest, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/openrca/test_phase1.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:

```bash
git add src/vlm4rca/openrca/phase1.py tests/openrca/test_phase1.py
git commit -m "feat: add OpenRCA phase1 sidecar pipeline"
```

Expected: commit succeeds.

---

### Task 8: Real Pilot Acceptance Test

**Files:**
- Modify: `tests/openrca/test_phase1.py`

- [ ] **Step 1: Add the pytest import**

Modify the import block at the top of `tests/openrca/test_phase1.py`:

```python
import json
from pathlib import Path

import pytest

from vlm4rca.openrca.adapter import OpenRCABankAdapter
from vlm4rca.openrca.manifest import load_case_manifest
from vlm4rca.openrca.phase1 import build_phase1_sidecars, main, write_phase1_outputs
```

- [ ] **Step 2: Add a real-data acceptance test**

Append this test to `tests/openrca/test_phase1.py`:

```python
@pytest.mark.skipif(
    not Path("data/OpenRCA/Bank/cases").exists(),
    reason="OpenRCA-Bank data symlink is not available",
)
def test_real_pilot15_outputs_phase1_sidecars(tmp_path: Path) -> None:
    output_dir = tmp_path / "pilot15"

    exit_code = main(
        [
            "--manifest",
            "configs/openrca_pilot15.yaml",
            "--data-root",
            "data/OpenRCA/Bank",
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    case_sidecars = sorted((output_dir / "case_sidecars").glob("*.json"))
    assert len(case_sidecars) == 15

    modality = json.loads((output_dir / "modality_availability.json").read_text(encoding="utf-8"))
    gt_mapping = json.loads((output_dir / "ground_truth_mapping.json").read_text(encoding="utf-8"))
    recall = json.loads((output_dir / "recall_empty_run.json").read_text(encoding="utf-8"))

    assert len(modality) == 15
    assert all("modality_availability" in row for row in modality)
    assert len(gt_mapping) == 15
    assert all(row["gt_mapping"] for row in gt_mapping)
    assert recall["n_cases"] == 15
    assert recall["component_recall_at_k"] == {"3": 0.0, "5": 0.0, "8": 0.0}

    written_paths = [path.name for path in output_dir.rglob("*") if path.is_file()]
    assert not any("candidate" in name for name in written_paths)
    assert not any(name.endswith((".png", ".jpg", ".jpeg", ".svg", ".html")) for name in written_paths)
```

- [ ] **Step 3: Run the real-data acceptance test**

Run:

```bash
uv run pytest tests/openrca/test_phase1.py::test_real_pilot15_outputs_phase1_sidecars -v
```

Expected: PASS when `data/OpenRCA/Bank` is present. If the data symlink is absent, expected output is SKIPPED with reason `OpenRCA-Bank data symlink is not available`.

- [ ] **Step 4: Run the full focused test suite**

Run:

```bash
uv run pytest tests/openrca tests/evaluation -v
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add tests/openrca/test_phase1.py
git commit -m "test: cover OpenRCA phase1 pilot acceptance"
```

Expected: commit succeeds.

---

### Task 9: Manual Pilot Run And Quality Checks

**Files:**
- Runtime output only under `outputs/openrca_phase1_pilot15/`

- [ ] **Step 1: Run the Phase 1 CLI on pilot 15**

Run:

```bash
uv run python -m vlm4rca.openrca.phase1 --manifest configs/openrca_pilot15.yaml --data-root data/OpenRCA/Bank --output-dir outputs/openrca_phase1_pilot15
```

Expected: command exits with status 0 and creates:

```text
outputs/openrca_phase1_pilot15/manifest_cases.json
outputs/openrca_phase1_pilot15/modality_availability.json
outputs/openrca_phase1_pilot15/ground_truth_mapping.json
outputs/openrca_phase1_pilot15/recall_empty_run.json
outputs/openrca_phase1_pilot15/case_sidecars/task3_001.json
```

- [ ] **Step 2: Confirm the pilot output shape**

Run:

```bash
uv run python -c "import json, pathlib; p=pathlib.Path('outputs/openrca_phase1_pilot15'); print(len(list((p/'case_sidecars').glob('*.json')))); print(len(json.loads((p/'modality_availability.json').read_text()))); print(len(json.loads((p/'ground_truth_mapping.json').read_text()))); print(json.loads((p/'recall_empty_run.json').read_text())['component_recall_at_k'])"
```

Expected output:

```text
15
15
15
{'3': 0.0, '5': 0.0, '8': 0.0}
```

- [ ] **Step 3: Confirm no candidates, plots, or model artifacts were produced**

Run:

```bash
find outputs/openrca_phase1_pilot15 -type f | grep -E 'candidate|\.png$|\.jpg$|\.jpeg$|\.svg$|\.html$' || true
```

Expected: no output.

- [ ] **Step 4: Run lint and full tests**

Run:

```bash
uv run ruff check src/ tests/
uv run pytest tests/ -v
```

Expected: ruff passes and tests pass.

- [ ] **Step 5: Inspect final status**

Run:

```bash
git status --short
```

Expected: no output if lint and tests did not require formatting changes.

- [ ] **Step 6: Commit final formatting changes if status is non-empty**

Run this only when Step 5 prints tracked Phase 1 files:

```bash
git add src tests configs
git commit -m "chore: finalize OpenRCA phase1 skeleton"
```

Expected: commit succeeds only when Step 5 showed tracked Phase 1 formatting changes after lint/test fixes.

---

## Self-Review

Spec coverage:

- OpenRCA adapter interface: Task 2 creates `OpenRCAAdapter` and `OpenRCABankAdapter`.
- OpenRCA-Bank subset case manifest: Task 1 creates `configs/openrca_pilot15.yaml` with 15 fixed case IDs.
- Incident and baseline window extractor: Task 3 creates `extract_incident_windows`.
- Service/component canonicalization: Task 4 creates `canonicalize_component` and mapping helpers.
- Modality availability records: Task 5 creates `detect_modality_availability`; Task 7 writes `modality_availability.json`.
- Ground-truth mapping: Task 4 creates mappings; Task 7 writes `ground_truth_mapping.json`.
- Recall@K empty framework: Task 6 creates `evaluate_component_recall_at_k`; Task 7 writes `recall_empty_run.json`.
- Acceptance on pilot 15: Task 8 adds the real-data acceptance test; Task 9 runs the CLI manually.
- No candidates, plots, or model calls: Task 7 output writer never writes candidate or image files; Task 8 and Task 9 assert no candidate/image artifacts.

Type consistency:

- `Phase1CaseSidecar.gt_mapping` uses `list[GroundTruthMapping]` throughout Tasks 4, 6, and 7.
- `ModalityAvailability.topology_source` uses `trace_derived`, `static_topology`, and `unavailable` throughout Tasks 1 and 5.
- Recall keys are stringified K values (`"3"`, `"5"`, `"8"`) in models, tests, and JSON output.

Final verification commands:

```bash
uv run pytest tests/openrca tests/evaluation -v
uv run ruff check src/ tests/
uv run python -m vlm4rca.openrca.phase1 --manifest configs/openrca_pilot15.yaml --data-root data/OpenRCA/Bank --output-dir outputs/openrca_phase1_pilot15
find outputs/openrca_phase1_pilot15 -type f | grep -E 'candidate|\.png$|\.jpg$|\.jpeg$|\.svg$|\.html$' || true
```
