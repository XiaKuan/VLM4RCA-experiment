import pytest

from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import (
    MultiSourceVariantResult,
    ShadowEdgeCandidate,
    SourceCandidate,
    SourceQuotaConfig,
    TopologyExpansionConfig,
)


def _candidate(rank: int, target: str) -> RcaCandidate:
    return RcaCandidate(
        case_id="case_001",
        variant="M+T",
        candidate_key=f"service:{target}",
        variant_candidate_id=f"cand:case_001:M+T:{rank}:service:{target}",
        target_type="service",
        canonical_target=target,
        raw_target=target,
        introduced_by="trace",
        metric_only_present=False,
        present_in_variants=["M+T", "M+T+L", "M+T+L+Topo"],
        sources=["trace"],
        source_scores={"trace": 1.0},
        evidence_summary=["trace evidence"],
        rank=rank,
        selected_for_rendering=True,
    )


def test_source_candidate_validates_key_and_bucket() -> None:
    candidate = SourceCandidate(
        case_id="case_001",
        candidate_key="service:checkout",
        target_type="service",
        canonical_target="checkout",
        raw_target="Checkout",
        source="trace",
        source_bucket="trace_service",
        score=4.25,
        evidence_summary=["span duration p95 increased from 10.00 to 30.00"],
    )

    assert candidate.candidate_key == "service:checkout"
    assert candidate.source == "trace"
    assert candidate.source_bucket == "trace_service"


def test_source_candidate_rejects_mismatched_candidate_key() -> None:
    with pytest.raises(ValueError, match="candidate_key must be service:checkout"):
        SourceCandidate(
            case_id="case_001",
            candidate_key="service:cart",
            target_type="service",
            canonical_target="checkout",
            raw_target="Checkout",
            source="trace",
            source_bucket="trace_service",
            score=4.25,
            evidence_summary=["span duration p95 increased"],
        )


def test_shadow_edge_candidate_uses_stable_edge_key() -> None:
    edge = ShadowEdgeCandidate(
        case_id="case_001",
        edge_key="edge:gateway->checkout",
        caller="gateway",
        callee="checkout",
        raw_caller="Gateway",
        raw_callee="Checkout",
        source_score=2.5,
        evidence_summary=["edge latency p95 increased from 5.00 to 20.00"],
        rank=1,
    )

    assert edge.edge_key == "edge:gateway->checkout"
    assert edge.rank == 1


def test_multi_source_variant_result_caps_final_candidates_at_eight() -> None:
    result = MultiSourceVariantResult(
        case_id="case_001",
        variant="M+T",
        candidates=[_candidate(rank, f"svc-{rank}") for rank in range(1, 9)],
        shadow_edges=[],
        warnings=[],
    )

    assert len(result.candidates) == 8


def test_multi_source_variant_result_rejects_more_than_eight_candidates() -> None:
    with pytest.raises(ValueError, match="final candidate budget is 8"):
        MultiSourceVariantResult(
            case_id="case_001",
            variant="M+T",
            candidates=[_candidate(rank, f"svc-{rank}") for rank in range(1, 10)],
            shadow_edges=[],
            warnings=[],
        )


def test_default_phase3_configs_match_spec() -> None:
    quotas = SourceQuotaConfig()
    topology = TopologyExpansionConfig()

    assert quotas.metric == 3
    assert quotas.trace_service == 2
    assert quotas.trace_edge_projected_service == 2
    assert quotas.log == 2
    assert quotas.topology == 2
    assert topology.expand_hops == 1
    assert topology.max_neighbors_per_candidate == 2
    assert topology.max_total_topology_candidates == 5


def test_metric_candidate_can_record_cross_variant_presence() -> None:
    candidate = RcaCandidate(
        case_id="case_001",
        variant="M",
        candidate_key="service:checkout",
        variant_candidate_id="cand:case_001:M:1:service:checkout",
        target_type="service",
        canonical_target="checkout",
        raw_target="Checkout",
        introduced_by="metric",
        metric_only_present=True,
        present_in_variants=["M", "M+T", "M+T+L", "M+T+L+Topo"],
        sources=["metric"],
        source_scores={"metric": 1.0},
        evidence_summary=["metric evidence"],
        rank=1,
        selected_for_rendering=True,
    )

    assert candidate.present_in_variants == ("M", "M+T", "M+T+L", "M+T+L+Topo")
