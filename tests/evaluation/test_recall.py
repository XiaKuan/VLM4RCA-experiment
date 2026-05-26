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
