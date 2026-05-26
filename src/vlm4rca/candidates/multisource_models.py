from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vlm4rca.candidates.models import CandidateSource, RcaCandidate, TargetType, VariantName

MAX_FINAL_CANDIDATES = 8

SourceBucket = Literal[
    "metric",
    "trace_service",
    "trace_edge_projected_service",
    "log",
    "topology",
]


class SourceCandidate(BaseModel, frozen=True):
    case_id: str
    candidate_key: str
    target_type: TargetType
    canonical_target: str
    raw_target: str
    source: CandidateSource
    source_bucket: SourceBucket
    score: float = Field(ge=0.0)
    evidence_summary: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    projection_from_edge: str | None = None
    topology_reason: str | None = None
    normalized_score: float = Field(default=0.0, ge=0.0)
    protected: bool = False

    @model_validator(mode="after")
    def validate_candidate_identity(self) -> SourceCandidate:
        expected_key = f"{self.target_type}:{self.canonical_target}"
        if self.candidate_key != expected_key:
            raise ValueError(f"candidate_key must be {expected_key}")
        if self.source_bucket == "metric" and self.source != "metric":
            raise ValueError("metric bucket must use source='metric'")
        if self.source_bucket.startswith("trace") and self.source != "trace":
            raise ValueError("trace buckets must use source='trace'")
        if self.source_bucket == "log" and self.source != "log":
            raise ValueError("log bucket must use source='log'")
        if self.source_bucket == "topology" and self.source != "topology":
            raise ValueError("topology bucket must use source='topology'")
        return self


class ShadowEdgeCandidate(BaseModel):
    case_id: str
    edge_key: str
    caller: str
    callee: str
    raw_caller: str
    raw_callee: str
    source_score: float = Field(ge=0.0)
    evidence_summary: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    rank: int = Field(ge=1)

    @field_validator("edge_key")
    @classmethod
    def edge_key_has_prefix(cls, value: str) -> str:
        if not value.startswith("edge:"):
            raise ValueError("edge_key must start with edge:")
        return value

    @model_validator(mode="after")
    def validate_edge_identity(self) -> ShadowEdgeCandidate:
        expected_key = f"edge:{self.caller}->{self.callee}"
        if self.edge_key != expected_key:
            raise ValueError(f"edge_key must be {expected_key}")
        return self


class SourceQuotaConfig(BaseModel, frozen=True):
    metric: int = 3
    trace_service: int = 2
    trace_edge_projected_service: int = 2
    log: int = 2
    topology: int = 2
    max_final_candidates: int = 8

    def quota_for(self, bucket: SourceBucket) -> int:
        return int(getattr(self, bucket))


class TopologyExpansionConfig(BaseModel, frozen=True):
    expand_hops: int = 1
    max_neighbors_per_candidate: int = 2
    max_total_topology_candidates: int = 5


class MultiSourceVariantResult(BaseModel):
    case_id: str
    variant: VariantName
    candidates: list[RcaCandidate]
    shadow_edges: list[ShadowEdgeCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_budget_and_ranks(self) -> MultiSourceVariantResult:
        if len(self.candidates) > MAX_FINAL_CANDIDATES:
            raise ValueError(f"final candidate budget is {MAX_FINAL_CANDIDATES}")
        ranks = [candidate.rank for candidate in self.candidates]
        if ranks != list(range(1, len(self.candidates) + 1)):
            raise ValueError("candidate ranks must be consecutive from 1")
        for candidate in self.candidates:
            if candidate.case_id != self.case_id:
                raise ValueError("All candidates must use the result case_id")
            if candidate.variant != self.variant:
                raise ValueError("All candidates must use the result variant")
        edge_ranks = [edge.rank for edge in self.shadow_edges]
        if edge_ranks != list(range(1, len(self.shadow_edges) + 1)):
            raise ValueError("shadow edge ranks must be consecutive from 1")
        return self
