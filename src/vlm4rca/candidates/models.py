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


METRIC_CANDIDATE_BUDGET = 8


class MetricFeatureEvidence(BaseModel, frozen=True):
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


class RcaCandidate(BaseModel, frozen=True):
    case_id: str
    variant: VariantName
    candidate_key: str
    variant_candidate_id: str
    target_type: TargetType
    canonical_target: str
    raw_target: str
    introduced_by: CandidateSource
    metric_only_present: bool
    present_in_variants: tuple[VariantName, ...]
    sources: tuple[CandidateSource, ...]
    source_scores: dict[CandidateSource, float]
    evidence_summary: tuple[str, ...] = Field(default_factory=tuple)
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

        if self.variant not in self.present_in_variants:
            raise ValueError("candidate variant must appear in present_in_variants")
        if self.introduced_by not in self.sources:
            raise ValueError("introduced_by must appear in sources")
        if self.variant == "M" and self.sources != ("metric",):
            raise ValueError("Metric-only candidates must have sources=('metric',)")
        if self.variant == "M" and self.introduced_by != "metric":
            raise ValueError("Metric-only candidates must have introduced_by='metric'")
        return self


class MetricCandidateBuildResult(BaseModel, frozen=True):
    case_id: str
    variant: Literal["M"]
    candidates: tuple[RcaCandidate, ...]
    warnings: tuple[str, ...] = Field(default_factory=tuple)

    @model_validator(mode="after")
    def enforce_metric_only_budget(self) -> MetricCandidateBuildResult:
        if len(self.candidates) > METRIC_CANDIDATE_BUDGET:
            raise ValueError(f"Metric-only candidates are capped at {METRIC_CANDIDATE_BUDGET}")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(self.candidates) + 1)):
            raise ValueError("Metric-only candidate ranks must be consecutive from 1")
        for candidate in self.candidates:
            if candidate.case_id != self.case_id:
                raise ValueError("All candidates must use the build result case_id")
            if candidate.variant != "M":
                raise ValueError("MetricCandidateBuildResult only accepts variant M")
        return self


class VariantRecallSummary(BaseModel, frozen=True):
    variant: str
    recall_at_3: float
    recall_at_5: float
    recall_at_8: float
    soft_recall_at_8: float
    avg_candidates: float
