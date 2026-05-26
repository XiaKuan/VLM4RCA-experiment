from vlm4rca.candidates.merge_rank import (
    build_variant_result,
    convert_metric_candidates_to_source_candidates,
    update_present_in_variants,
)
from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import SourceCandidate, SourceQuotaConfig


def _metric_candidate(rank: int, target: str, score: float) -> RcaCandidate:
    return RcaCandidate(
        case_id="case_001",
        variant="M",
        candidate_key=f"service:{target}",
        variant_candidate_id=f"cand:case_001:M:{rank}:service:{target}",
        target_type="service",
        canonical_target=target,
        raw_target=target,
        introduced_by="metric",
        metric_only_present=True,
        present_in_variants=("M",),
        sources=("metric",),
        source_scores={"metric": score},
        evidence_summary=["metric evidence"],
        rank=rank,
        selected_for_rendering=True,
    )


def _source(target: str, bucket: str, score: float, source: str | None = None) -> SourceCandidate:
    actual_source = source or ("trace" if bucket.startswith("trace") else bucket)
    return SourceCandidate(
        case_id="case_001",
        candidate_key=f"service:{target}",
        target_type="service",
        canonical_target=target,
        raw_target=target,
        source=actual_source,
        source_bucket=bucket,
        score=score,
        evidence_summary=[f"{bucket} evidence"],
    )


def test_convert_metric_candidates_to_source_candidates() -> None:
    converted = convert_metric_candidates_to_source_candidates([_metric_candidate(1, "checkout", 8.0)])

    assert converted[0].candidate_key == "service:checkout"
    assert converted[0].source == "metric"
    assert converted[0].source_bucket == "metric"
    assert converted[0].score == 8.0


def test_build_variant_result_merges_duplicate_services_and_keeps_top_eight() -> None:
    metric = [_metric_candidate(1, "checkout", 8.0), _metric_candidate(2, "redis", 4.0)]
    sources = [
        _source("checkout", "trace_service", 5.0, "trace"),
        _source("payment", "trace_edge_projected_service", 4.0, "trace"),
        _source("gateway", "log", 3.0, "log"),
        _source("inventory", "topology", 2.0, "topology"),
    ]

    result = build_variant_result(
        case_id="case_001",
        variant="M+T+L+Topo",
        metric_candidates=metric,
        source_candidates=sources,
        shadow_edges=[],
        quotas=SourceQuotaConfig(max_final_candidates=8),
    )

    checkout = result.candidates[0]
    assert checkout.canonical_target == "checkout"
    assert checkout.sources == ("metric", "trace")
    assert checkout.introduced_by == "metric"
    assert checkout.metric_only_present is True
    assert checkout.source_scores["metric"] == 1.0
    assert checkout.source_scores["trace"] == 1.0
    assert len(result.candidates) == 5
    assert [candidate.rank for candidate in result.candidates] == [1, 2, 3, 4, 5]
    assert all(candidate.variant_candidate_id.startswith("cand:case_001:M+T+L+Topo:") for candidate in result.candidates)


def test_protected_candidates_are_ranked_before_non_protected_candidates() -> None:
    metric = [_metric_candidate(1, "metric-a", 10.0), _metric_candidate(2, "metric-b", 9.0), _metric_candidate(3, "metric-c", 8.0)]
    sources = [
        _source("trace-a", "trace_service", 10.0, "trace"),
        _source("trace-b", "trace_service", 9.0, "trace"),
        _source("log-a", "log", 1.0, "log"),
        _source("log-b", "log", 0.9, "log"),
        _source("trace-low", "trace_service", 100.0, "trace"),
    ]

    result = build_variant_result(
        case_id="case_001",
        variant="M+T+L",
        metric_candidates=metric,
        source_candidates=sources,
        shadow_edges=[],
        quotas=SourceQuotaConfig(metric=3, trace_service=2, log=2, max_final_candidates=8),
    )

    protected_targets = [candidate.canonical_target for candidate in result.candidates[:7]]

    assert set(protected_targets) == {"metric-a", "metric-b", "metric-c", "trace-low", "trace-a", "log-a", "log-b"}


def test_variant_filters_disabled_source_buckets() -> None:
    metric = [_metric_candidate(1, "checkout", 8.0)]
    sources = [_source("payment", "trace_service", 5.0, "trace"), _source("gateway", "log", 3.0, "log")]

    result = build_variant_result(
        case_id="case_001",
        variant="M+T",
        metric_candidates=metric,
        source_candidates=sources,
        shadow_edges=[],
    )

    assert [candidate.canonical_target for candidate in result.candidates] == ["checkout", "payment"]


def test_update_present_in_variants_records_cross_variant_presence() -> None:
    m = build_variant_result("case_001", "M", [_metric_candidate(1, "checkout", 8.0)], [], [])
    mt = build_variant_result("case_001", "M+T", [_metric_candidate(1, "checkout", 8.0)], [_source("payment", "trace_service", 5.0, "trace")], [])

    updated = update_present_in_variants({"M": m, "M+T": mt})

    assert updated["M"].candidates[0].present_in_variants == ("M", "M+T")
    assert updated["M+T"].candidates[0].present_in_variants == ("M", "M+T")
    assert updated["M+T"].candidates[1].present_in_variants == ("M+T",)
