from vlm4rca.candidates.models import RcaCandidate, VariantRecallSummary
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
        "case_2": [
            _candidate("case_2", 1, "redis01", 9.0),
            _candidate("case_2", 2, "mysql02", 8.0),
        ],
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

    row = summarize_variant_recall("M", sidecars, candidates)

    assert isinstance(row, VariantRecallSummary)
    assert row.variant == "M"
    assert row.recall_at_3 == 0.5
    assert row.recall_at_5 == 0.5
    assert row.recall_at_8 == 0.5
    assert row.soft_recall_at_8 == 0.0
    assert row.avg_candidates == 1.0
