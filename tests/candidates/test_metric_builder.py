from pathlib import Path

import pandas as pd

from vlm4rca.candidates.metric_builder import (
    build_metric_candidates_for_case,
    build_metric_candidates_from_dataframe,
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


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1_000, 1_100],
            "Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 40, 41, 42],
            "Tomcat01__container__JVM-Memory_HeapMemoryUsage": [50, 50, 50, 75, 76, 75],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [20, 20, 20, 20, 20, 20],
            "Mysql02__container__OSLinux_LOCALDISK_LOCALDISK-sda_DSKBps": [5, 5, 5, 30, 31, 32],
        }
    )


def test_build_metric_candidates_groups_evidence_by_component() -> None:
    result = build_metric_candidates_from_dataframe(
        "case_001", _frame(), _windows(), max_candidates=8
    )

    assert result.case_id == "case_001"
    assert result.variant == "M"
    assert [candidate.canonical_target for candidate in result.candidates] == [
        "tomcat01",
        "mysql02",
        "redis01",
    ]
    assert result.candidates[0].candidate_key == "service:tomcat01"
    assert result.candidates[0].variant_candidate_id == "cand:case_001:M:1:service:tomcat01"
    assert (
        result.candidates[0].source_scores["metric"] > result.candidates[1].source_scores["metric"]
    )
    assert result.candidates[0].introduced_by == "metric"
    assert result.candidates[0].metric_only_present is True
    assert result.candidates[0].present_in_variants == ("M",)
    assert result.candidates[0].sources == ("metric",)
    assert result.candidates[0].selected_for_rendering is True
    assert len(result.candidates[0].evidence_summary) == 2


def test_build_metric_candidates_caps_output_to_top_eight() -> None:
    frame = pd.DataFrame({"timestamp": [600, 700, 800, 900, 1_000, 1_100]})
    for index in range(10):
        frame[f"Svc{index:02d}__container__OSLinux-CPU_CPU_CPUCpuUtil"] = [
            10,
            10,
            10,
            20 + index,
            21 + index,
            22 + index,
        ]

    result = build_metric_candidates_from_dataframe("case_001", frame, _windows(), max_candidates=8)

    assert len(result.candidates) == 8
    assert [candidate.rank for candidate in result.candidates] == list(range(1, 9))
    assert all(
        candidate.variant_candidate_id.startswith("cand:case_001:M:")
        for candidate in result.candidates
    )


def test_build_metric_candidates_for_case_reads_metrics_csv(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.csv"
    _frame().to_csv(metrics_path, index=False)

    result = build_metric_candidates_for_case("case_001", metrics_path, _windows())

    assert result.candidates[0].canonical_target == "tomcat01"
    assert result.warnings == ()
