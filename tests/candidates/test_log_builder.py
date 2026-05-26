from pathlib import Path

import pandas as pd

from vlm4rca.candidates.log_builder import (
    DEFAULT_LOG_KEYWORDS,
    build_log_candidates_for_case,
    build_log_candidates_from_dataframe,
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


def _logs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [610, 700, 910, 930, 950, 980, 620, 940],
            "service": [
                "Checkout",
                "Checkout",
                "Checkout",
                "Checkout",
                "Checkout",
                "Checkout",
                "Gateway",
                "Gateway",
            ],
            "message": [
                "ok",
                "retry once",
                "timeout while calling payment",
                "deadline exceeded from payment",
                "connection refused",
                "retry backoff",
                "ok",
                "slow latency observed",
            ],
        }
    )


def test_default_keywords_include_phase0_required_terms() -> None:
    assert "timeout" in DEFAULT_LOG_KEYWORDS
    assert "deadline exceeded" in DEFAULT_LOG_KEYWORDS
    assert "connection refused" in DEFAULT_LOG_KEYWORDS
    assert "circuit breaker" in DEFAULT_LOG_KEYWORDS


def test_build_log_candidates_scores_keyword_rate_delta() -> None:
    result = build_log_candidates_from_dataframe("case_001", _logs(), _windows())

    assert [candidate.canonical_target for candidate in result.source_candidates] == [
        "checkout",
        "gateway",
    ]
    assert result.source_candidates[0].candidate_key == "service:checkout"
    assert result.source_candidates[0].source == "log"
    assert result.source_candidates[0].source_bucket == "log"
    assert result.source_candidates[0].score > result.source_candidates[1].score
    assert "keyword_rate_delta" in result.source_candidates[0].evidence_summary[0]


def test_build_log_candidates_uses_rate_normalization() -> None:
    result = build_log_candidates_from_dataframe("case_001", _logs(), _windows())

    checkout = result.source_candidates[0]

    assert round(checkout.score, 2) == 0.80


def test_build_log_candidates_for_case_reads_logs_csv(tmp_path: Path) -> None:
    logs_path = tmp_path / "logs.csv"
    _logs().to_csv(logs_path, index=False)

    result = build_log_candidates_for_case("case_001", logs_path, _windows())

    assert result.source_candidates[0].canonical_target == "checkout"
    assert result.warnings == []


def test_missing_log_columns_returns_warning_without_candidates() -> None:
    frame = pd.DataFrame({"timestamp": [1, 2], "body": ["timeout", "error"]})

    result = build_log_candidates_from_dataframe("case_001", frame, _windows())

    assert result.source_candidates == []
    assert "missing required log columns" in result.warnings[0]
