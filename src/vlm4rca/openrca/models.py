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


class ManifestCase(BaseModel, frozen=True):
    case_id: str
    case_group: CaseGroup
    selection_reason: str


class CaseManifest(BaseModel, frozen=True):
    dataset: str
    data_root: str
    cases: list[ManifestCase]


class IncidentWindows(BaseModel, frozen=True):
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
    sampling_interval_assumption: str = (
        "epoch seconds; telemetry timestamps are interpreted as UTC seconds"
    )
    warnings: list[str] = Field(default_factory=list)


class ModalityAvailability(BaseModel, frozen=True):
    metrics_available: bool
    traces_available: bool
    logs_available: bool
    topology_available: bool
    topology_source: TopologySource
    warnings: list[str] = Field(default_factory=list)


class GroundTruthMapping(BaseModel, frozen=True):
    raw_ground_truth: str
    mapped_component: str
    mapping_type: MappingType
    mapping_confidence: MappingConfidence
    source: str


class Phase1CaseSidecar(BaseModel, frozen=True):
    case_id: str
    case_group: CaseGroup
    task_index: str | None = None
    inject_time: int
    case_path: str
    raw_ground_truth: list[str]
    incident_window: IncidentWindows
    modality_availability: ModalityAvailability
    gt_mapping: list[GroundTruthMapping]


class RecallCaseResult(BaseModel, frozen=True):
    case_id: str
    hit_at_k: dict[str, bool]
    hit_targets: list[str]
    missed_targets: list[str]


class RecallAtKSummary(BaseModel, frozen=True):
    n_cases: int
    ks: list[int]
    component_recall_at_k: dict[str, float]
    case_results: list[RecallCaseResult]
