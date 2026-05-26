import pandas as pd

from vlm4rca.candidates.metric_features import (
    MetricColumn,
    categorize_metric,
    parse_metric_column,
    score_metric_series,
    score_metrics_dataframe,
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


def test_parse_metric_column_extracts_component_and_category() -> None:
    parsed = parse_metric_column("Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil")

    assert parsed == MetricColumn(
        metric_column="Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil",
        raw_target="Tomcat01",
        canonical_target="tomcat01",
        metric_name="container__OSLinux-CPU_CPU_CPUCpuUtil",
        metric_category="cpu",
    )


def test_categorize_metric_uses_openrca_metric_name_heuristics() -> None:
    assert categorize_metric("JVM_Memory_HeapMemoryUsage") == "memory"
    assert categorize_metric("OSLinux_LOCALDISK_LOCALDISK-sda_DSKBps") == "disk"
    assert categorize_metric("NETWORK_ens160_NETOutErr") == "error"
    assert categorize_metric("NETWORK_ens160_NETKBTotalPerSec") == "network"
    assert categorize_metric("request_count") == "traffic"
    assert categorize_metric("span_duration_p95") == "latency"
    assert categorize_metric("unrecognized_signal") == "unknown"


def test_score_metric_series_computes_required_features() -> None:
    baseline = pd.Series([10.0, 10.0, 11.0, 9.0, 10.0, 10.0])
    incident = pd.Series([20.0, 21.0, 19.0, 20.0])

    evidence = score_metric_series(
        MetricColumn(
            metric_column="Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil",
            raw_target="Tomcat01",
            canonical_target="tomcat01",
            metric_name="container__OSLinux-CPU_CPU_CPUCpuUtil",
            metric_category="cpu",
        ),
        baseline,
        incident,
    )

    assert evidence.robust_z_score > 6.0
    assert round(evidence.relative_change, 2) == 1.0
    assert round(evidence.p95_shift, 2) == 0.94
    assert evidence.score > evidence.robust_z_score
    assert evidence.baseline_points == 6
    assert evidence.incident_points == 4


def test_score_metric_series_returns_zero_score_for_constant_no_change() -> None:
    baseline = pd.Series([0.0, 0.0, 0.0])
    incident = pd.Series([0.0, 0.0, 0.0])

    evidence = score_metric_series(
        MetricColumn(
            metric_column="Redis01__container__JVM-Memory_HeapMemoryUsage",
            raw_target="Redis01",
            canonical_target="redis01",
            metric_name="container__JVM-Memory_HeapMemoryUsage",
            metric_category="memory",
        ),
        baseline,
        incident,
    )

    assert evidence.robust_z_score == 0.0
    assert evidence.relative_change == 0.0
    assert evidence.p95_shift == 0.0
    assert evidence.score == 0.0


def test_score_metrics_dataframe_filters_to_baseline_and_incident_windows() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": [600, 700, 800, 900, 1_000, 1_100],
            "Tomcat01__container__OSLinux-CPU_CPU_CPUCpuUtil": [10, 10, 10, 20, 21, 20],
            "Redis01__container__JVM-Memory_HeapMemoryUsage": [5, 5, 5, 5, 5, 5],
        }
    )

    evidence, warnings = score_metrics_dataframe(frame, _windows())

    assert warnings == []
    assert [item.canonical_target for item in evidence] == ["tomcat01", "redis01"]
    assert evidence[0].metric_category == "cpu"
    assert evidence[0].baseline_points == 3
    assert evidence[0].incident_points == 3
