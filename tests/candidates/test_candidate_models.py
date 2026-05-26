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
    assert candidate.present_in_variants == ("M",)
    assert candidate.sources == ("metric",)


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
