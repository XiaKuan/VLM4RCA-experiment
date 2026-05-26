from pathlib import Path

import pandas as pd

from vlm4rca.candidates.trace_builder import (
    build_trace_candidates_for_case,
    build_trace_candidates_from_dataframe,
)
from vlm4rca.openrca.models import IncidentWindows


def _windows() -> IncidentWindows:
    return IncidentWindows(
        inject_time=1_000,
        baseline_start=600,
        baseline_end=900,
        incident_start=900,
        incident_end=1_200,
        pre_window_seconds=100,
        post_window_seconds=200,
        baseline_window_seconds=300,
    )


def _trace_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1_000, 1_100, 600, 700, 800, 900, 1_000, 1_100],
            "trace_id": ["t1", "t2", "t3", "t4", "t5", "t6", "u1", "u2", "u3", "u4", "u5", "u6"],
            "span_id": ["a", "a", "a", "a", "a", "a", "b", "b", "b", "b", "b", "b"],
            "parent_span_id": ["", "", "", "", "", "", "a", "a", "a", "a", "a", "a"],
            "service": ["Gateway", "Gateway", "Gateway", "Gateway", "Gateway", "Gateway", "Checkout", "Checkout", "Checkout", "Checkout", "Checkout", "Checkout"],
            "caller": ["Client", "Client", "Client", "Client", "Client", "Client", "Gateway", "Gateway", "Gateway", "Gateway", "Gateway", "Gateway"],
            "callee": ["Gateway", "Gateway", "Gateway", "Gateway", "Gateway", "Gateway", "Checkout", "Checkout", "Checkout", "Checkout", "Checkout", "Checkout"],
            "duration": [10, 10, 10, 30, 31, 32, 5, 5, 5, 25, 26, 27],
        }
    )


def test_build_trace_candidates_scores_service_duration_shift() -> None:
    result = build_trace_candidates_from_dataframe("case_001", _trace_frame(), _windows())

    service_candidates = [candidate for candidate in result.source_candidates if candidate.source_bucket == "trace_service"]

    assert [candidate.canonical_target for candidate in service_candidates[:2]] == ["checkout", "gateway"]
    assert service_candidates[0].candidate_key == "service:checkout"
    assert service_candidates[0].source == "trace"
    assert service_candidates[0].score > service_candidates[1].score
    assert "span duration p95 increased" in service_candidates[0].evidence_summary[0]


def test_build_trace_candidates_emits_shadow_edges_and_projected_services() -> None:
    result = build_trace_candidates_from_dataframe("case_001", _trace_frame(), _windows())

    assert result.shadow_edges[0].edge_key == "edge:gateway->checkout"
    assert result.shadow_edges[0].source_score > 0.0

    projected = [
        candidate
        for candidate in result.source_candidates
        if candidate.source_bucket == "trace_edge_projected_service"
    ]

    assert [candidate.canonical_target for candidate in projected[:2]] == ["checkout", "gateway"]
    assert projected[0].projection_from_edge == "edge:gateway->checkout"
    assert projected[0].source == "trace"


def test_build_trace_candidates_reconstructs_edges_from_parent_span_ids() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [600, 600, 700, 700, 800, 800, 900, 900, 1_000, 1_000, 1_100, 1_100],
            "trace_id": ["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4", "t5", "t5", "t6", "t6"],
            "span_id": ["root", "child", "root", "child", "root", "child", "root", "child", "root", "child", "root", "child"],
            "parent_span_id": ["", "root", "", "root", "", "root", "", "root", "", "root", "", "root"],
            "service": ["Gateway", "Checkout", "Gateway", "Checkout", "Gateway", "Checkout", "Gateway", "Checkout", "Gateway", "Checkout", "Gateway", "Checkout"],
            "duration": [10, 5, 10, 5, 10, 5, 30, 25, 31, 26, 32, 27],
        }
    )

    result = build_trace_candidates_from_dataframe("case_001", frame, _windows())

    assert any(edge.edge_key == "edge:gateway->checkout" for edge in result.shadow_edges)


def test_build_trace_candidates_for_case_reads_traces_csv(tmp_path: Path) -> None:
    traces_path = tmp_path / "traces.csv"
    _trace_frame().to_csv(traces_path, index=False)

    result = build_trace_candidates_for_case("case_001", traces_path, _windows())

    assert result.source_candidates
    assert result.shadow_edges
    assert result.warnings == []


def test_missing_trace_columns_returns_warning_without_candidates() -> None:
    frame = pd.DataFrame({"timestamp": [1, 2], "message": ["a", "b"]})

    result = build_trace_candidates_from_dataframe("case_001", frame, _windows())

    assert result.source_candidates == []
    assert result.shadow_edges == []
    assert "missing required trace columns" in result.warnings[0]
