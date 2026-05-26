from vlm4rca.candidates.models import RcaCandidate
from vlm4rca.candidates.multisource_models import ShadowEdgeCandidate
from vlm4rca.evaluation.ablation import (
    build_candidate_recall_ablation,
    build_checkpoint_decision,
    build_first_hit_rows,
    summarize_shadow_edges,
)
from vlm4rca.openrca.models import (
    GroundTruthMapping,
    IncidentWindows,
    ModalityAvailability,
    Phase1CaseSidecar,
)


def _sidecar(
    case_id: str,
    group: str,
    target: str,
    *,
    metrics: bool = True,
    traces: bool = True,
    logs: bool = True,
    topology: bool = True,
) -> Phase1CaseSidecar:
    return Phase1CaseSidecar(
        case_id=case_id,
        case_group=group,
        task_index="task_3",
        inject_time=1000,
        case_path=f"/tmp/{case_id}",
        raw_ground_truth=[target],
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
            metrics_available=metrics,
            traces_available=traces,
            logs_available=logs,
            topology_available=topology,
            topology_source="trace_derived" if topology else "unavailable",
        ),
        gt_mapping=[
            GroundTruthMapping(
                raw_ground_truth=target,
                mapped_component=target,
                mapping_type="component_exact",
                mapping_confidence="exact",
                source="test",
            )
        ],
    )


def _candidate(
    case_id: str, variant: str, rank: int, target: str, introduced_by: str
) -> RcaCandidate:
    return RcaCandidate(
        case_id=case_id,
        variant=variant,
        candidate_key=f"service:{target}",
        variant_candidate_id=f"cand:{case_id}:{variant}:{rank}:service:{target}",
        target_type="service",
        canonical_target=target,
        raw_target=target,
        introduced_by=introduced_by,
        metric_only_present=introduced_by == "metric",
        present_in_variants=(variant,),
        sources=(introduced_by,),
        source_scores={introduced_by: 1.0},
        evidence_summary=[f"{introduced_by} evidence"],
        rank=rank,
        selected_for_rendering=True,
    )


def test_candidate_recall_ablation_separates_primary_and_secondary() -> None:
    sidecars = [
        _sidecar("case_1", "metric_obvious", "checkout"),
        _sidecar("case_2", "soft_latency", "payment", logs=False),
    ]
    candidates = {
        "M": {
            "case_1": [_candidate("case_1", "M", 1, "checkout", "metric")],
            "case_2": [],
        },
        "M+T": {
            "case_1": [_candidate("case_1", "M+T", 1, "checkout", "metric")],
            "case_2": [_candidate("case_2", "M+T", 1, "payment", "trace")],
        },
        "M+T+L": {
            "case_1": [_candidate("case_1", "M+T+L", 1, "checkout", "metric")],
            "case_2": [
                _candidate("case_2", "M+T+L", 1, "payment", "trace")
            ],
        },
        "M+T+L+Topo": {
            "case_1": [
                _candidate("case_1", "M+T+L+Topo", 1, "checkout", "metric")
            ],
            "case_2": [
                _candidate("case_2", "M+T+L+Topo", 1, "payment", "trace")
            ],
        },
    }

    report = build_candidate_recall_ablation(sidecars, candidates)

    assert report["secondary"]["M"]["recall_at_8"] == 0.5
    assert report["secondary"]["M+T"]["recall_at_8"] == 1.0
    assert report["primary"]["M+T+L"]["n_cases"] == 1
    assert report["primary"]["M+T+L"]["recall_at_8"] == 1.0


def test_first_hit_rows_compute_first_and_new_hit_sources() -> None:
    sidecars = [_sidecar("case_1", "soft_latency", "payment")]
    candidates = {
        "M": {"case_1": [_candidate("case_1", "M", 1, "checkout", "metric")]},
        "M+T": {
            "case_1": [_candidate("case_1", "M+T", 1, "payment", "trace")]
        },
        "M+T+L": {
            "case_1": [_candidate("case_1", "M+T+L", 1, "payment", "trace")]
        },
        "M+T+L+Topo": {
            "case_1": [
                _candidate("case_1", "M+T+L+Topo", 1, "payment", "trace")
            ]
        },
    }

    rows = build_first_hit_rows(sidecars, candidates)

    assert rows[0]["first_hit_source"] == "Trace"
    assert rows[0]["new_hit_source"] == "Trace"
    assert rows[0]["metric_hit_at_8"] is False
    assert rows[0]["metric_trace_hit_at_8"] is True


def test_checkpoint_uses_full_eligible_cases_for_metric_vs_full_comparison() -> None:
    sidecars = [
        _sidecar("case_1", "metric_obvious", "checkout"),
        _sidecar("case_2", "soft_latency", "payment"),
    ]
    candidates = {
        "M": {
            "case_1": [_candidate("case_1", "M", 1, "checkout", "metric")],
            "case_2": [],
        },
        "M+T+L+Topo": {
            "case_1": [
                _candidate("case_1", "M+T+L+Topo", 1, "checkout", "metric")
            ],
            "case_2": [
                _candidate("case_2", "M+T+L+Topo", 1, "payment", "trace")
            ],
        },
    }

    decision = build_checkpoint_decision(sidecars, candidates)

    assert decision["eligible_cases"] == 2
    assert decision["metric_recall_at_8"] == 0.5
    assert decision["full_recall_at_8"] == 1.0
    assert decision["proceed_to_vlm_and_rendering"] is True


def test_shadow_edge_metrics_are_diagnostic() -> None:
    shadow_edges = {
        "case_1": [
            ShadowEdgeCandidate(
                case_id="case_1",
                edge_key="edge:gateway->checkout",
                caller="gateway",
                callee="checkout",
                raw_caller="gateway",
                raw_callee="checkout",
                source_score=1.0,
                evidence_summary=["edge evidence"],
                rank=1,
            )
        ]
    }

    summary = summarize_shadow_edges(
        {"case_1": ["edge:gateway->checkout"]}, shadow_edges
    )

    assert summary["diagnostic_only"] is True
    assert summary["edge_recall_at_3"] == 1.0
    assert summary["edge_recall_at_8"] == 1.0
